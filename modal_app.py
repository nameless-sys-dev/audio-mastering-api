import json
import time
import traceback
from pathlib import Path

import modal


APP_NAME = "audio-mastering"
ROOT = "/data"

# Transitioned from debian_slim to an Ubuntu base image with Python 3.11 injected.
image = (
    modal.Image.from_registry("ubuntu:22.04", add_python="3.11")
    # Make sure ca-certificates is installed
    .apt_install("ffmpeg", "libsndfile1", "ca-certificates")
    .pip_install(
        "fastapi==0.115.12",
        "pydantic==2.11.4",
        "python-multipart==0.0.20",
        "httpx==0.28.1",
        "numpy==2.2.5",
        "scipy==1.15.2",
        "soundfile==0.13.1",
    )
    .add_local_python_source("mastering")
)

app = modal.App(APP_NAME)

volume = modal.Volume.from_name(
    "audio-mastering-files", create_if_missing=True
)
jobs = modal.Dict.from_name(
    "audio-mastering-jobs", create_if_missing=True
)
secrets = modal.Secret.from_name("mastering-secrets")


@app.function(
    image=image,
    secrets=[secrets],
    volumes={ROOT: volume},
    cpu=4,
    memory=8192,
    timeout=3600,
    max_containers=4,
)
def process_job(job_id: str):
    from mastering import master_file

    record = jobs[job_id]
    record["status"] = "running"
    record["started_at"] = time.time()
    jobs[job_id] = record

    try:
        # Uploads are committed before the worker is spawned.
        volume.reload()

        source = (
            Path(ROOT)
            / "sources"
            / record["source_id"]
            / f"source{record['source_suffix']}"
        )
        output_directory = Path(ROOT) / "outputs" / job_id

        parent = (
            jobs[record["parent_id"]]
            if record["parent_id"] is not None
            else None
        )

        result = master_file(
            source,
            record,
            parent,
            output_directory,
        )

        record["result"] = result
        record["status"] = "ready"
        record["finished_at"] = time.time()

        (output_directory / "report.json").write_text(
            json.dumps(record, indent=2, allow_nan=False)
        )

        # Publish ready status only after the artifact is committed.
        volume.commit()
        jobs[job_id] = record

    except Exception as exc:
        traceback.print_exc()

        record.pop("result", None)
        record["status"] = "failed"
        record["finished_at"] = time.time()
        record["error"] = (
            str(exc)[:1200]
            if isinstance(exc, ValueError)
            else "Processing failed. Inspect the Modal worker logs."
        )
        jobs[job_id] = record


@app.function(
    image=image,
    secrets=[secrets],
    volumes={ROOT: volume},
    memory=1024,
    timeout=180,
    max_containers=8,
)
# Important: avoids Volume.reload() while another request in the same
# container has an artifact open for a streaming FileResponse.
@modal.concurrent(max_inputs=1)
@modal.asgi_app()
def api():
    import hashlib
    import hmac
    import os
    import shutil
    from typing import get_args
    from uuid import UUID, uuid4

    from fastapi import (
        Depends,
        FastAPI,
        File,
        Form,
        HTTPException,
        UploadFile,
    )
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.security import HTTPBearer

    from mastering import (
        DEMUXERS,
        EXPORTS,
        MAX_SECONDS,
        MAX_UPLOAD_BYTES,
        MIN_SECONDS,
        TARGET_MAX_LUFS,
        TARGET_MIN_LUFS,
        TRUE_PEAK_CEILING,
        ExportFormat,
        Genre,
        RevisionRequest,
    )

    web = FastAPI(
        title="Audio Mastering API",
        version="1.0.0",
        description=(
            "Upload a mix, choose a genre and LUFS target, "
            "then request revisions using plain-language feedback."
        ),
        dependencies=[Depends(HTTPBearer())],
    )

    public_paths = {
        "/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"
    }

    @web.middleware("http")
    async def guard(request, call_next):
        if request.url.path not in public_paths:
            # Authenticate before FastAPI parses a potentially large multipart body.
            header = request.headers.get("authorization", "")
            scheme, _, token = header.partition(" ")

            authorized = (
                scheme.lower() == "bearer"
                and hmac.compare_digest(
                    token.encode(),
                    os.environ["MASTER_API_KEY"].encode(),
                )
            )

            if not authorized:
                return JSONResponse(
                    {"detail": "Unauthorized"},
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )

            if request.method == "POST" and request.url.path == "/jobs":
                # Require a bounded upload before multipart parsing starts.
                raw_length = request.headers.get("content-length")
                if raw_length is None:
                    return JSONResponse(
                        {"detail": "Content-Length is required for uploads."},
                        status_code=411,
                    )

                try:
                    length = int(raw_length)
                except ValueError:
                    return JSONResponse(
                        {"detail": "Invalid Content-Length."},
                        status_code=400,
                    )

                # Extra allowance for multipart boundaries and form fields.
                if length < 0 or length > MAX_UPLOAD_BYTES + 1024 * 1024:
                    return JSONResponse(
                        {"detail": "Upload exceeds the size limit."},
                        status_code=413,
                    )

        response = await call_next(request)

        if request.url.path not in public_paths:
            response.headers["Cache-Control"] = "private, no-store"

        return response

    async def fetch_job(job_id: str):
        record = await jobs.get.aio(job_id, None)
        if record is None:
            raise HTTPException(404, "Job not found.")
        return record

    async def enqueue(record: dict):
        job_id = record["id"]
        await jobs.put.aio(job_id, record)

        try:
            call = await process_job.spawn.aio(job_id)
        except Exception:
            record["status"] = "failed"
            record["error"] = "Could not enqueue the worker."
            await jobs.put.aio(job_id, record)
            raise HTTPException(503, "Could not enqueue the worker.")

        return {
            "id": job_id,
            "status": "queued",
            "modal_call_id": call.object_id,
            "status_url": f"/jobs/{job_id}",
            "download_url": f"/jobs/{job_id}/download",
        }

    @web.get("/options")
    async def options():
        return {
            "genres": list(get_args(Genre)),
            "export_formats": EXPORTS,
            "input_extensions": sorted(DEMUXERS),
            "target_lufs_range": [TARGET_MIN_LUFS, TARGET_MAX_LUFS],
            "true_peak_ceiling_dbtp": TRUE_PEAK_CEILING,
            "minimum_duration_seconds": MIN_SECONDS,
            "maximum_duration_seconds": MAX_SECONDS,
            "maximum_upload_bytes": MAX_UPLOAD_BYTES,
            "channels": [1, 2],
            "lossless_export_bit_depth": 24,
            "lossless_sample_rate": "Preserve source rate",
            "opus_sample_rate_hz": 48000,
            "maximum_revision_depth": 10,
        }

    @web.post("/jobs", status_code=202)
    async def create_job(
        audio: UploadFile = File(...),
        genre: Genre = Form(...),
        target_lufs: float = Form(
            ..., ge=TARGET_MIN_LUFS, le=TARGET_MAX_LUFS
        ),
        export_format: ExportFormat = Form(...),
        description: str = Form("", max_length=4000),
    ):
        suffix = Path(audio.filename or "").suffix.lower()

        if suffix not in DEMUXERS:
            raise HTTPException(
                415,
                "Supported inputs: MP3, WAV, M4A AAC/ALAC, FLAC, "
                "and Opus in .opus/.ogg/.webm.",
            )

        job_id = str(uuid4())
        source_directory = Path(ROOT) / "sources" / job_id

        await volume.reload.aio()
        source_directory.mkdir(parents=True, exist_ok=False)

        source_path = source_directory / f"source{suffix}"
        size = 0
        digest = hashlib.sha256()

        try:
            with source_path.open("wb") as destination:
                while chunk := await audio.read(1024 * 1024):
                    size += len(chunk)

                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "Audio file is too large.")

                    digest.update(chunk)
                    destination.write(chunk)

            if size == 0:
                raise HTTPException(400, "The uploaded file is empty.")

            await volume.commit.aio()

        except BaseException:
            shutil.rmtree(source_directory, ignore_errors=True)
            raise
        finally:
            await audio.close()

        record = {
            "id": job_id,
            "source_id": job_id,
            "source_suffix": suffix,
            "source_sha256": digest.hexdigest(),
            "original_filename": audio.filename,
            "parent_id": None,
            "revision": 0,
            "status": "queued",
            "created_at": time.time(),
            "genre": genre,
            "target_lufs": target_lufs,
            "export_format": export_format,
            "description": description,
            "feedback_history": [],
        }

        return await enqueue(record)

    @web.get("/jobs/{job_id}")
    async def get_job(job_id: UUID):
        return await fetch_job(str(job_id))

    @web.post("/jobs/{job_id}/revisions", status_code=202)
    async def create_revision(job_id: UUID, request: RevisionRequest):
        parent = await fetch_job(str(job_id))

        if parent["status"] != "ready":
            raise HTTPException(409, "The parent master is not ready.")

        if parent["revision"] >= 10:
            raise HTTPException(409, "Maximum revision depth reached.")

        feedback = request.feedback.strip()
        if not feedback:
            raise HTTPException(422, "Feedback must not be blank.")

        new_id = str(uuid4())

        record = {
            "id": new_id,
            "source_id": parent["source_id"],
            "source_suffix": parent["source_suffix"],
            "source_sha256": parent["source_sha256"],
            "original_filename": parent["original_filename"],
            "parent_id": parent["id"],
            "revision": parent["revision"] + 1,
            "status": "queued",
            "created_at": time.time(),
            "genre": (
                request.genre if request.genre is not None
                else parent["genre"]
            ),
            "target_lufs": (
                request.target_lufs if request.target_lufs is not None
                else parent["target_lufs"]
            ),
            "export_format": (
                request.export_format if request.export_format is not None
                else parent["export_format"]
            ),
            "description": parent["description"],
            "feedback_history": [
                *parent["feedback_history"],
                feedback,
            ],
        }

        return await enqueue(record)

    @web.get("/jobs/{job_id}/download")
    async def download(job_id: UUID):
        job_id = str(job_id)
        record = await fetch_job(job_id)

        if record["status"] != "ready":
            raise HTTPException(409, "The master is not ready.")

        await volume.reload.aio()

        artifact = record["result"]["artifact"]
        path = Path(ROOT) / "outputs" / job_id / artifact["filename"]

        if not path.is_file():
            raise HTTPException(503, "Artifact is not visible yet; retry shortly.")

        return FileResponse(
            path,
            media_type=artifact["mime"],
            filename=f"master-{job_id}{path.suffix}",
            headers={"Cache-Control": "private, no-store"},
        )

    return web