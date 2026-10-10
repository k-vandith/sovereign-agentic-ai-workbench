"""Shared LinkLens-derived styling primitives for the Sovereign Workbench."""
from __future__ import annotations

from pathlib import Path

_STYLE_PATH = Path(__file__).with_name("style.css")
_BASE_CSS = _STYLE_PATH.read_text(encoding="utf-8")

_DARK = """
:root {
  --wb-bg:#0e100d; --wb-surface:#161a14; --wb-surface-2:#1d2319;
  --wb-text:#f0f2e8; --wb-muted:#a6ae9d; --wb-border:#30382a;
  --wb-accent:#d4ff59; --wb-accent-strong:#eaffb0; --wb-accent-soft:#28341a;
  --wb-success:#9ad9a2; --wb-warning:#f5c36a; --wb-danger:#ff929a;
}
"""

_LIGHT = """
:root {
  --wb-bg:#f4f7fa; --wb-surface:#ffffff; --wb-surface-2:#eaf2f6;
  --wb-text:#172532; --wb-muted:#53697a; --wb-border:#cfdae2;
  --wb-accent:#087f91; --wb-accent-strong:#075d6a; --wb-accent-soft:#d7f0f2;
  --wb-success:#08774a; --wb-warning:#805400; --wb-danger:#a51f30;
}
html, body, [data-testid="stAppViewContainer"], .stApp { background:var(--wb-bg); color:var(--wb-text); }
[data-testid="stSidebar"] { background:#eaf0f5; }
.wb-hero { background:radial-gradient(circle at 94% 4%,rgba(8,127,145,.13),transparent 30%),linear-gradient(130deg,#fff,#eaf2f6); }
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] { color:#fff; }
"""

def get_css(mode: str = "dark") -> str:
    """Return the shared style sheet with this product's light or dark tokens."""
    override = _LIGHT if str(mode).lower() == "light" else _DARK
    return f"<style>\n{_BASE_CSS}\n{override}\n</style>"

def plotly_template(mode: str = "dark") -> dict:
    """Return common chart tokens for projects that render Plotly charts."""
    light = str(mode).lower() == "light"
    return {
        "layout": {
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "font": {"family": 'Inter, "Segoe UI", sans-serif', "color": "#172532" if light else "#eaf2f8"},
            "title": {"font": {"size": 17}},
            "colorway": ["#087f91", "#7c5ce7", "#2e9a72", "#c17b21", "#d45d79"] if light else ["#d4ff59", "#9ad9a2", "#f5c36a", "#b4a4ff", "#ff929a"],
            "xaxis": {"gridcolor": "#dce5ec" if light else "#253545", "zerolinecolor": "#c8d6df" if light else "#253545"},
            "yaxis": {"gridcolor": "#dce5ec" if light else "#253545", "zerolinecolor": "#c8d6df" if light else "#253545"},
            "margin": {"l": 12, "r": 12, "t": 42, "b": 12},
        }
    }

CSS = get_css("dark")
PLOTLY_TEMPLATE = plotly_template("dark")
