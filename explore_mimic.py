#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["requests", "rich", "python-dotenv"]
# ///
"""
MIMIC-IV v2.1 dataset explorer — lists all files and fetches CSV headers
without downloading the full dataset.

Usage:
    uv run explore_mimic.py

Prerequisites:
    Create a .env file at the repo root with:
        KAGGLE_API_TOKEN=<your token>

    Get your token at: https://www.kaggle.com/settings → API → "Create New Token"
    The token starts with "KGAT_".
"""

import csv
import io
import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich import print as rprint

DATASET_OWNER = "mangeshwagle"
DATASET_SLUG  = "mimic-iv-2-1"
API_BASE      = "https://www.kaggle.com/api/v1"

console = Console()


# ── credentials ──────────────────────────────────────────────────────────────

def load_token() -> str:
    """Load KAGGLE_API_TOKEN from .env, then fall back to the environment."""
    load_dotenv()
    token = os.getenv("KAGGLE_API_TOKEN")
    if not token:
        console.print(
            "[bold red]KAGGLE_API_TOKEN not found.[/bold red]\n"
            "Add it to your [cyan].env[/cyan] file:\n\n"
            "    KAGGLE_API_TOKEN=KGAT_<your token>\n\n"
            "Get one at: https://www.kaggle.com/settings → API → 'Create New Token'"
        )
        sys.exit(1)
    return token


# ── API helpers ───────────────────────────────────────────────────────────────

def list_files(session: requests.Session) -> list[dict]:
    """Return all file metadata objects for the dataset (no file data)."""
    url = f"{API_BASE}/datasets/{DATASET_OWNER}/{DATASET_SLUG}/files"
    files = []
    page = 1
    while True:
        r = session.get(url, params={"page": page, "pageSize": 200})
        r.raise_for_status()
        data = r.json()
        batch = data.get("datasetFiles", [])
        if not batch:
            break
        files.extend(batch)
        # stop if we got fewer than the page size — we're on the last page
        if len(batch) < 200:
            break
        page += 1
    return files


def fetch_csv_header(session: requests.Session, file_path: str) -> list[str] | None:
    """Download only the first 4 KB of a file and parse the CSV header row."""
    url = (
        f"{API_BASE}/datasets/{DATASET_OWNER}/{DATASET_SLUG}"
        f"/download/{file_path}"
    )
    # Range request: grab first 4096 bytes — always enough for a header row
    r = session.get(url, headers={"Range": "bytes=0-4095"}, allow_redirects=True)

    if r.status_code not in (200, 206):
        return None

    # The file may be .csv or .csv.gz (Kaggle sometimes recompresses)
    content_type = r.headers.get("Content-Type", "")
    raw = r.content

    if "gzip" in content_type or file_path.endswith(".gz"):
        import gzip
        try:
            raw = gzip.decompress(raw)
        except Exception:
            pass  # not actually gzip, fall through

    text = raw.decode("utf-8", errors="replace")
    first_line = text.splitlines()[0] if text.splitlines() else ""
    reader = csv.reader(io.StringIO(first_line))
    try:
        columns = next(reader)
    except StopIteration:
        return None

    return [c.strip() for c in columns if c.strip()]


# ── display ───────────────────────────────────────────────────────────────────

def human_size(n_bytes: int) -> str:
    value: float = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def print_file_table(files: list[dict]) -> None:
    table = Table(title="MIMIC-IV v2.1 — Dataset files", show_lines=True)
    table.add_column("#",          style="dim",  width=4)
    table.add_column("Path",       style="cyan", no_wrap=False)
    table.add_column("Size",       style="green", justify="right")
    table.add_column("Type",       style="yellow")

    for i, f in enumerate(files, 1):
        path  = f.get("name", "?")
        size  = human_size(f.get("totalBytes", 0))
        ftype = Path(path).suffix.lstrip(".").upper() or "—"
        table.add_row(str(i), path, size, ftype)

    console.print(table)


def print_schema(file_path: str, columns: list[str]) -> None:
    rprint(f"\n[bold cyan]{file_path}[/bold cyan]")
    for col in columns:
        rprint(f"  • {col}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    token = load_token()
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {token}"

    console.rule("[bold blue]MIMIC-IV v2.1 — File Explorer[/bold blue]")

    # 1. list all files (metadata only)
    console.print("\n[bold]Fetching file list …[/bold]")
    files = list_files(session)
    console.print(f"Found [bold green]{len(files)}[/bold green] files.\n")
    print_file_table(files)

    # 2. for every CSV, fetch only its header row
    csv_files = [
        f for f in files
        if f.get("name", "").lower().endswith((".csv", ".csv.gz"))
    ]

    if not csv_files:
        console.print("\n[yellow]No CSV files found.[/yellow]")
        return

    console.rule(f"\n[bold blue]CSV column headers ({len(csv_files)} files)[/bold blue]")
    console.print("Fetching first 4 KB of each CSV (Range requests — no full download)…\n")

    schemas: dict[str, list[str]] = {}
    for f in csv_files:
        path = f["name"]
        console.print(f"  → {path}", end=" ")
        cols = fetch_csv_header(session, path)
        if cols:
            schemas[path] = cols
            console.print(f"[green]({len(cols)} columns)[/green]")
        else:
            console.print("[red](failed)[/red]")

    # 3. pretty-print all schemas
    console.rule("\n[bold blue]Schema summary[/bold blue]")
    for path, cols in schemas.items():
        print_schema(path, cols)

    # 4. save as JSON for later use
    out = Path("mimic_iv_schema.json")
    out.write_text(json.dumps(schemas, indent=2))
    console.print(f"\n[bold green]Schemas saved to {out}[/bold green]")


if __name__ == "__main__":
    main()
