"""Streamlit smoke tests for all primary workbench pages."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "src" / "ui" / "app.py"
PAGES = [
    "Overview",
    "Chat & Evidence",
    "Document Library",
    "Tool Trace",
    "Audit Log",
    "Settings & Glossary",
]


@pytest.mark.parametrize("page", PAGES)
def test_primary_page_renders_without_exception(page):
    app = AppTest.from_file(str(APP), default_timeout=45).run()
    assert not app.exception
    app.radio[0].set_value(page).run()
    assert not app.exception
