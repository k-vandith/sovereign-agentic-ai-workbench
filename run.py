"""Start the UI or the API without shell-specific paths.

    python run.py
    python run.py --api
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Sovereign Agentic AI Workbench")
    parser.add_argument("--api", action="store_true", help="Start the FastAPI server")
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    if args.api:
        port = str(args.port or 8000)
        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "src.api.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            port,
        ]
    else:
        port = str(args.port or 8501)
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(ROOT / "app.py"),
            "--server.port",
            port,
            "--server.headless",
            "true",
        ]
    raise SystemExit(subprocess.call(cmd, cwd=ROOT))


if __name__ == "__main__":
    main()
