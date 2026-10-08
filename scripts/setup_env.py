#!/usr/bin/env python3
"""Cross-platform virtualenv bootstrap for this project.

Handles:
- missing python3-venv / ensurepip
- symlink restrictions (uses --copies first)
- Windows vs POSIX paths

Usage (from project root):
    python3 scripts/setup_env.py
    python  scripts/setup_env.py
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
REQ = ROOT / "requirements.txt"
IS_WIN = platform.system() == "Windows"


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, check=True, **kwargs)


def create_venv() -> None:
    if VENV.exists():
        print(f"[ok] Virtualenv already exists: {VENV}")
        return

    print(f"[..] Creating virtualenv at {VENV}")
    strategies = [
        [sys.executable, "-m", "venv", str(VENV), "--copies"],
        [sys.executable, "-m", "venv", str(VENV)],
        [sys.executable, "-m", "venv", str(VENV), "--copies", "--without-pip"],
        [sys.executable, "-m", "venv", str(VENV), "--without-pip"],
    ]
    last_err: Exception | None = None
    for cmd in strategies:
        try:
            _run(cmd)
            print(f"[ok] Created with: {' '.join(cmd[3:])}")
            return
        except (subprocess.CalledProcessError, OSError) as exc:
            last_err = exc
            if VENV.exists():
                shutil.rmtree(VENV, ignore_errors=True)
            print(f"[warn] Strategy failed: {exc}")

    raise SystemExit(
        "Could not create a virtual environment.\n"
        "  Debian/Ubuntu:  sudo apt install python3-venv python3-pip\n"
        "  macOS Homebrew: brew install python\n"
        "  Windows:        install Python from python.org and re-run\n"
        f"Last error: {last_err}"
    )


def venv_python() -> Path:
    if IS_WIN:
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def ensure_pip(py: Path) -> None:
    try:
        _run([str(py), "-m", "pip", "--version"], capture_output=True)
        print("[ok] pip is available in the venv")
        return
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    print("[..] Bootstrapping pip …")
    try:
        _run([str(py), "-m", "ensurepip", "--upgrade"])
        return
    except subprocess.CalledProcessError:
        pass

    get_pip = ROOT / "scripts" / "_get_pip.py"
    try:
        import urllib.request

        print("[..] Downloading get-pip.py …")
        urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", str(get_pip))
        _run([str(py), str(get_pip)])
    except Exception as exc:
        raise SystemExit(
            "pip is not available and could not be bootstrapped.\n"
            "Install system packages: python3-pip / python3-venv, then re-run.\n"
            f"Detail: {exc}"
        ) from exc
    finally:
        if get_pip.exists():
            get_pip.unlink(missing_ok=True)


def install_requirements(py: Path) -> None:
    if not REQ.exists():
        raise SystemExit(f"Missing {REQ}")
    print(f"[..] Installing dependencies from {REQ.name}")
    _run([str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    _run([str(py), "-m", "pip", "install", "-r", str(REQ)])
    print("[ok] Dependencies installed")


def print_help() -> None:
    print()
    print("=" * 60)
    print("Setup complete. Activate the environment:")
    print()
    if IS_WIN:
        print(r"  .venv\Scripts\Activate.ps1")
        print(r"  # cmd.exe:  .venv\Scripts\activate.bat")
    else:
        print("  source .venv/bin/activate")
    print()
    print("Then:")
    print("  pytest -v")
    print("  python scripts/generate_demo_data.py   # if present")
    print("=" * 60)


def main() -> None:
    os.chdir(ROOT)
    print(f"Project root: {ROOT}")
    print(f"Python:       {sys.executable} ({sys.version.split()[0]})")
    print(f"Platform:     {platform.system()} {platform.machine()}")
    print()
    create_venv()
    py = venv_python()
    if not py.exists():
        raise SystemExit(f"Expected venv interpreter not found: {py}")
    ensure_pip(py)
    install_requirements(py)
    print_help()


if __name__ == "__main__":
    main()
