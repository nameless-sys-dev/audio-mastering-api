import os
import sys
import requests
from typing import Optional, Dict, Any
from rich.console import Console
from rich.prompt import Prompt, FloatPrompt, Confirm
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

# Initialize Rich console
console = Console()

# Defined from openapi.json schemas
GENRE_CHOICES = [
    "pop", "rock", "metal", "hip_hop", "rnb", "edm", "electronic", 
    "jazz", "classical", "acoustic", "folk", "country", "ambient", 
    "cinematic", "reggae", "latin", "other"
]
FORMAT_CHOICES = ["wav", "m4a_aac", "m4a_alac", "flac", "opus"]

class MasteringAPIClient:
    def __init__(self, base_url: str, api_key: str):
        """Initializes the API client with the base URL and authorization key."""
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}"
        })

    def _handle_response(self, response: requests.Response) -> Optional[Dict[str, Any]]:
        """A helper method to parse the JSON response and handle HTTP errors nicely."""
        try:
            response.raise_for_status()
            # If the response has no content (e.g., 204 No Content), return an empty dict
            if not response.text.strip():
                return {}
            return response.json()
        except requests.exceptions.HTTPError as err:
            console.print(f"[bold red]HTTP Error:[/bold red] {err}")
            try:
                # Try to extract detailed validation errors from FastAPI (HTTPValidationError)
                error_details = response.json()
                if "detail" in error_details:
                    console.print("[red]API Details:[/red]")
                    if isinstance(error_details["detail"], list):
                        for detail in error_details["detail"]:
                            loc = " -> ".join([str(l) for l in detail.get("loc", [])])
                            console.print(f"  - [yellow]{loc}[/yellow]: {detail.get('msg')}")
                    else:
                        console.print(f"  - {error_details['detail']}")
            except ValueError:
                console.print(f"[red]Raw Response:[/red] {response.text}")
            return None
        except Exception as err:
            console.print(f"[bold red]Unexpected Error:[/bold red] {err}")
            return None

    def get_options(self):
        """GET /options - Fetches options from the server."""
        with console.status("[cyan]Fetching options..."):
            response = self.session.get(f"{self.base_url}/options")
            return self._handle_response(response)

    def create_job(self, file_path: str, genre: str, target_lufs: float, export_format: str, description: str):
        """POST /jobs - Uploads a mix to be mastered."""
        if not os.path.exists(file_path):
            console.print(f"[bold red]Error:[/bold red] File '{file_path}' does not exist.")
            return None

        url = f"{self.base_url}/jobs"
        
        # Open file in binary mode for multipart/form-data upload
        with open(file_path, "rb") as f:
            files = {"audio": (os.path.basename(file_path), f)}
            data = {
                "genre": genre,
                "target_lufs": target_lufs,
                "export_format": export_format,
                "description": description
            }
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                progress.add_task("[cyan]Uploading file and creating job...", total=None)
                response = self.session.post(url, files=files, data=data)
                
            return self._handle_response(response)

    def get_job(self, job_id: str):
        """GET /jobs/{job_id} - Retrieves the status and details of a specific job."""
        with console.status(f"[cyan]Fetching job '{job_id}'..."):
            response = self.session.get(f"{self.base_url}/jobs/{job_id}")
            return self._handle_response(response)

    def create_revision(self, job_id: str, feedback: str, target_lufs: Optional[float] = None, 
                        export_format: Optional[str] = None, genre: Optional[str] = None):
        """POST /jobs/{job_id}/revisions - Requests a revision for a completed job."""
        payload = {"feedback": feedback}
        
        if target_lufs is not None:
            payload["target_lufs"] = target_lufs
        if export_format:
            payload["export_format"] = export_format
        if genre:
            payload["genre"] = genre

        with console.status(f"[cyan]Submitting revision for job '{job_id}'..."):
            # This endpoint requires JSON application payload as per OpenAPI schema
            response = self.session.post(f"{self.base_url}/jobs/{job_id}/revisions", json=payload)
            return self._handle_response(response)

    def download_job(self, job_id: str):
        """GET /jobs/{job_id}/download - Gets download metadata for a job."""
        with console.status(f"[cyan]Fetching download details for job '{job_id}'..."):
            response = self.session.get(f"{self.base_url}/jobs/{job_id}/download")
            return self._handle_response(response)


def display_menu():
    """Displays the main interactive menu for the CLI tool."""
    table = Table(title="Audio Mastering API - Main Menu", style="blue")
    table.add_column("Option", style="cyan", justify="center")
    table.add_column("Action", style="white")
    
    table.add_row("1", "View Server Options")
    table.add_row("2", "Create New Mastering Job")
    table.add_row("3", "Check Job Status")
    table.add_row("4", "Request Revision")
    table.add_row("5", "Download Mastered Audio")
    table.add_row("6", "Exit")
    
    console.print(table)


def main():
    """Main execution block containing the setup and interactive loop."""
    console.print(Panel.fit("[bold magenta]Welcome to the Interactive Audio Mastering CLI[/bold magenta]"))
    
    # Load environment variables or ask the user for them
    api_url = os.environ.get("MASTERING_API_URL")
    if not api_url:
        api_url = Prompt.ask("[yellow]Enter the API Base URL (e.g., https://api.example.com)[/yellow]")
        
    api_key = os.environ.get("MASTERING_API_KEY")
    if not api_key:
        api_key = Prompt.ask("[yellow]Enter your Master Key (Bearer Token)[/yellow]", password=True)
        
    client = MasteringAPIClient(base_url=api_url, api_key=api_key)
    console.print(f"\n[green]Connected configured for:[/green] {api_url}\n")
    
    while True:
        display_menu()
        choice = Prompt.ask("Select an option", choices=["1", "2", "3", "4", "5", "6"])
        console.print()  # Spacer
        
        if choice == "1":
            # View Options
            result = client.get_options()
            if result is not None:
                console.print(Panel("Server Options Retrieved Successfully", style="green"))
                console.print_json(data=result)
                
        elif choice == "2":
            # Create Job
            console.print(Panel("Create New Mastering Job", style="blue"))
            file_path = Prompt.ask("Path to audio file (e.g., ./mix.wav)")
            genre = Prompt.ask("Select Genre", choices=GENRE_CHOICES, default="pop")
            
            # Request LUFS (between -30.0 and -6.0 as per schema)
            while True:
                target_lufs = FloatPrompt.ask("Target LUFS (between -30.0 and -6.0)", default=-14.0)
                if -30.0 <= target_lufs <= -6.0:
                    break
                console.print("[red]LUFS must be between -30.0 and -6.0[/red]")
                
            export_format = Prompt.ask("Export Format", choices=FORMAT_CHOICES, default="wav")
            description = Prompt.ask("Description (optional)", default="")
            
            result = client.create_job(file_path, genre, target_lufs, export_format, description)
            if result is not None:
                console.print(Panel("[bold green]Job Created Successfully![/bold green]"))
                console.print_json(data=result)
                
        elif choice == "3":
            # Check Job Status
            console.print(Panel("Check Job Status", style="blue"))
            job_id = Prompt.ask("Enter Job ID (UUID)")
            if job_id:
                result = client.get_job(job_id)
                if result is not None:
                    console.print_json(data=result)
                    
        elif choice == "4":
            # Request Revision
            console.print(Panel("Request Revision", style="blue"))
            job_id = Prompt.ask("Enter Job ID (UUID)")
            feedback = Prompt.ask("Enter your plain-language feedback (e.g., 'Make the bass punchier')")
            
            # Optional attributes
            new_lufs = None
            if Confirm.ask("Do you want to change the target LUFS?"):
                while True:
                    val = FloatPrompt.ask("New Target LUFS (between -30.0 and -6.0)")
                    if -30.0 <= val <= -6.0:
                        new_lufs = val
                        break
                    console.print("[red]LUFS must be between -30.0 and -6.0[/red]")
            
            new_format = None
            if Confirm.ask("Do you want to change the export format?"):
                new_format = Prompt.ask("New Export Format", choices=FORMAT_CHOICES)
                
            new_genre = None
            if Confirm.ask("Do you want to change the genre profiling?"):
                new_genre = Prompt.ask("New Genre", choices=GENRE_CHOICES)
                
            result = client.create_revision(
                job_id=job_id, 
                feedback=feedback, 
                target_lufs=new_lufs, 
                export_format=new_format, 
                genre=new_genre
            )
            
            if result is not None:
                console.print(Panel("[bold green]Revision Requested Successfully![/bold green]"))
                console.print_json(data=result)
                
        elif choice == "5":
            # Download Master
            console.print(Panel("Download Mastered Audio", style="blue"))
            job_id = Prompt.ask("Enter Job ID (UUID)")
            if job_id:
                result = client.download_job(job_id)
                if result is not None:
                    console.print(Panel("Download Details", style="green"))
                    console.print_json(data=result)
                    # Note: Since the schema returns JSON for the download endpoint (likely containing a signed URL),
                    # we print the JSON. If it returned raw bytes, we would handle streaming it to a file here.
                    
        elif choice == "6":
            # Exit
            console.print("[bold cyan]Goodbye![/bold cyan]")
            sys.exit(0)
            
        # Pause before clearing/showing menu again
        Prompt.ask("\n[dim]Press Enter to continue...[/dim]", default="")
        console.clear()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold red]Operation cancelled by user. Exiting...[/bold red]")
        sys.exit(1)