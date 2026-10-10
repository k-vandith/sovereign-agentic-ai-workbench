"""Reusable layout components shared by the workbench and sibling projects."""
from __future__ import annotations

import html
import streamlit as st


def page_intro(title: str, purpose: str, how_to: str) -> None:
    """Place a consistent purpose statement and a collapsible reading guide."""
    st.markdown(f"## {html.escape(title)}")
    st.markdown(f'<div class="wb-purpose">{html.escape(purpose)}</div>', unsafe_allow_html=True)
    with st.expander("How to read this page"):
        st.write(how_to)


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
        f'<div class="wb-kpi-help">{html.escape(explanation)}</div></div>',
        unsafe_allow_html=True,
    )


def flow_steps() -> None:
    steps = [
        ("1", "Upload", "Add PDFs, Office documents, text or images."),
        ("2", "Configure", "Choose the Demo or local Ollama model."),
        ("3", "Ask & inspect", "Get an answer, source excerpts and tool activity."),
        ("4", "Review", "Check the audit trail and export your notes."),
    ]
    cells = "".join(
        f'<div class="wb-step"><div class="wb-step-number">{number}</div>'
        f'<div class="wb-step-title">{title}</div><div class="wb-step-help">{description}</div></div>'
        for number, title, description in steps
    )
    st.markdown(f'<div class="wb-flow">{cells}</div>', unsafe_allow_html=True)


def brand_header(subtitle: str, pills_html: str) -> None:
    st.markdown(
        '<div class="wb-top"><div class="wb-brand">'
        '<div class="wb-mark"><svg viewBox="0 0 64 64" role="img" aria-label="Sovereign Workbench mark">'
        '<path d="M9 17 32 7 55 18 55 45 32 57 9 45Z" fill="#173944" stroke="#55c7d9" stroke-width="2"/>'
        '<path d="M18 25 32 17 46 25 46 39 32 47 18 39Z" fill="none" stroke="#9beaf4" stroke-width="2.5"/>'
        '<circle cx="32" cy="32" r="5" fill="#55c7d9"/></svg></div>'
        '<div><div class="wb-wordmark">SOVEREIGN WORKBENCH'
        '<small>Private knowledge · Local AI</small></div></div></div>'
        f'<div><div class="wb-kicker">WORKSPACE</div><div class="wb-subtitle">{html.escape(subtitle)}</div></div>'
        f'<div class="wb-pills">{pills_html}</div></div>',
        unsafe_allow_html=True,
    )
