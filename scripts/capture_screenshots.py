"""Capture real UI screenshots for each Streamlit page using Playwright.

Install the optional tooling first:
    python -m pip install -e ".[screenshots]"
    python -m playwright install chromium

Then run:
    python scripts/capture_screenshots.py
"""
from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT = 8510
PAGES = [
    ("workspace", "Workspace"),
    ("documents", "Documents"),
    ("settings", "Settings"),
]


def _wait_for_port(port: int, process: subprocess.Popen, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Streamlit stopped before it became ready. Run python run.py --port 8510 to inspect the error.")
        with socket.socket() as sock:
            sock.settimeout(0.5)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.25)
    raise TimeoutError("Streamlit did not start within 60 seconds.")


def main() -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print('Playwright is missing. Run: python -m pip install -e ".[screenshots]"')
        return 2

    process = subprocess.Popen(
        [sys.executable, "run.py", "--port", str(PORT)],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    output_dir = ROOT / "docs" / "screenshots"
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        _wait_for_port(PORT, process)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
            page.goto(f"http://127.0.0.1:{PORT}", wait_until="domcontentloaded", timeout=60_000)
            page.locator(".wb-wordmark").wait_for(timeout=30_000)
            for filename, label in PAGES:
                if label != "Workspace":
                    page.get_by_role("radio", name=label).check(timeout=15_000)
                    page.wait_for_timeout(700)
                page.screenshot(path=str(output_dir / f"{filename}.png"), full_page=True, animations="disabled")
                print(f"Saved {output_dir / f'{filename}.png'}")
            browser.close()
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
