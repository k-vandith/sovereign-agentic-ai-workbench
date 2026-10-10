"""Small reusable layout helpers for the Sovereign Workbench."""
from __future__ import annotations

import html
import streamlit as st


def page_intro(title: str, purpose: str, how_to: str) -> None:
    st.markdown(f"## {html.escape(title)}")
    st.markdown(f'<div class="wb-purpose">{html.escape(purpose)}</div>', unsafe_allow_html=True)


def section(title: str, explanation: str = "") -> None:
    st.markdown(
        f'<div class="wb-section"><h3>{html.escape(title)}</h3>'
        f'<div class="wb-card-copy">{html.escape(explanation)}</div></div>',
        unsafe_allow_html=True,
    )


def kpi(label: str, value: object, explanation: str) -> None:
    st.markdown(
        f'<div class="wb-kpi"><div class="wb-kpi-label">{html.escape(str(label))}</div>'
        f'<div class="wb-kpi-value">{html.escape(str(value))}</div>'
        f'<div class="wb-kpi-help">{html.escape(str(explanation))}</div></div>',
        unsafe_allow_html=True,
    )


def flow_steps() -> None:
    steps = [
        ("1", "Add files", "Choose the documents you need."),
        ("2", "Connect an API", "Use an API for stronger answers."),
        ("3", "Ask & verify", "Check the source passages behind each answer."),
    ]
    cells = "".join(
        f'<div class="wb-step"><div class="wb-step-number">{number}</div>'
        f'<div class="wb-step-title">{title}</div><div class="wb-step-help">{description}</div></div>'
        for number, title, description in steps
    )
    st.markdown(f'<div class="wb-flow">{cells}</div>', unsafe_allow_html=True)


def brand_header(subtitle: str, pills_html: str) -> None:
    st.markdown(
        '<div class="wb-top"><div>'
        '<div class="wb-kicker">SOVEREIGN / KNOWLEDGE</div>'
        f'<div class="wb-subtitle">{html.escape(subtitle)}</div></div>'
        f'<div class="wb-pills">{pills_html}</div></div>',
        unsafe_allow_html=True,
    )
