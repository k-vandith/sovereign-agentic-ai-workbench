from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import run
from scripts import capture_screenshots, setup_env
from src.ui.app import PAGES as APP_PAGES


def test_setup_bootstrap_installs_the_documented_test_dependencies():
    assert setup_env.REQ.name == "requirements-dev.txt"
    assert setup_env.REQ.is_file()
    assert "pytest" in setup_env.REQ.read_text(encoding="utf-8")


def test_screenshot_script_targets_current_ui_navigation():
    assert [label for _filename, label in capture_screenshots.PAGES] == APP_PAGES
    assert len({filename for filename, _label in capture_screenshots.PAGES}) == len(APP_PAGES)


def test_run_api_uses_configured_host_and_port(monkeypatch):
    settings = SimpleNamespace(api_host="127.0.0.2", api_port=8123, streamlit_port=9123)
    captured = {}
    monkeypatch.setattr(run, "Settings", lambda **_kwargs: settings)
    monkeypatch.setattr(sys, "argv", ["run.py", "--api"])
    monkeypatch.setattr(
        run.subprocess,
        "call",
        lambda command, cwd: captured.update(command=command, cwd=cwd) or 0,
    )

    with pytest.raises(SystemExit) as result:
        run.main()

    command = captured["command"]
    assert result.value.code == 0
    assert command[command.index("--host") + 1] == "127.0.0.2"
    assert command[command.index("--port") + 1] == "8123"


def test_run_ui_uses_configured_streamlit_port(monkeypatch):
    settings = SimpleNamespace(api_host="127.0.0.1", api_port=8000, streamlit_port=9123)
    captured = {}
    monkeypatch.setattr(run, "Settings", lambda **_kwargs: settings)
    monkeypatch.setattr(sys, "argv", ["run.py"])
    monkeypatch.setattr(
        run.subprocess,
        "call",
        lambda command, cwd: captured.update(command=command, cwd=cwd) or 0,
    )

    with pytest.raises(SystemExit) as result:
        run.main()

    command = captured["command"]
    assert result.value.code == 0
    assert command[command.index("--server.port") + 1] == "9123"
