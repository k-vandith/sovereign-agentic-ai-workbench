"""Sovereign workbench — chat, sources, tools, and document workspace."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.agent import RAGAgent
from src.config import get_settings
from src.paths import safe_filename
from src.ui.theme import CSS

logger = logging.getLogger(__name__)


def _agent() -> RAGAgent:
    @st.cache_resource
    def _build() -> RAGAgent:
        return RAGAgent()

    return _build()


def _status_pills(agent: RAGAgent) -> str:
    settings = agent.settings
    backend = type(agent.llm).__name__
    mode = "Offline" if settings.offline_mode else "Network allowed"
    ready = "Ready" if agent.llm.is_available() else "Unavailable"
    tone = "ok" if backend == "DemoLLM" or agent.llm.is_available() else "warn"
    return (
        '<div class="wb-pills">'
        f'<span class="pill {tone}"><strong>● {ready}</strong> {backend}</span>'
        f'<span class="pill"><strong>{mode}</strong></span>'
        f'<span class="pill"><strong>CPU</strong></span>'
        f'<span class="pill"><strong>{agent.store.count()}</strong> chunks</span>'
        f'<span class="pill"><strong>{agent.principal.role.value}</strong></span>'
        "</div>"
    )


def main() -> None:
    logging.basicConfig(level=get_settings().log_level)
    st.set_page_config(
        page_title="Sovereign Workbench",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)
    agent = _agent()

    st.markdown(
        '<div class="wb-top"><div>'
        '<div class="wb-kicker">On-premise workspace</div>'
        '<div class="wb-title">Sovereign Agentic AI</div>'
        "</div>"
        f"{_status_pills(agent)}</div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Documents")
        st.caption("Files stay on this machine. Demo mode does not call a cloud model.")
        uploaded = st.file_uploader(
            "Add a document",
            type=["pdf", "docx", "txt", "md", "csv", "png", "jpg", "jpeg"],
            label_visibility="collapsed",
        )
        if uploaded is not None and st.session_state.get("last_upload") != uploaded.name:
            try:
                name = safe_filename(uploaded.name)
                dest = agent.settings.upload_dir / name
                dest.write_bytes(uploaded.getvalue())
                with st.spinner("Indexing…"):
                    result = agent.ingest_file(dest)
                st.session_state["last_upload"] = uploaded.name
                st.success(f"{result['chunks_added']} chunks from {result['file']}")
            except Exception as exc:
                logger.exception("Ingest failed")
                st.error("That file could not be indexed. Use PDF, DOCX, TXT, or MD.")
                st.caption(str(exc))
        if st.button("Load sample documents", use_container_width=True):
            sample = ROOT / "data" / "sample"
            if not sample.exists():
                st.warning("Sample folder is missing. Run scripts/generate_demo_data.py")
            else:
                summary = agent.ingest_directory(sample)
                st.success(f"Processed {summary['files_processed']} files.")
        if st.button("Clear knowledge base", use_container_width=True):
            agent.clear_knowledge_base()
            st.session_state.messages = []
            st.session_state.conversation_id = None
            st.rerun()
        st.divider()
        st.markdown("### Model")
        st.write(f"Backend `{type(agent.llm).__name__}`")
        st.caption(
            "Set LLM_BACKEND=ollama in .env after `ollama pull llama3.2:1b`. "
            "Offline mode blocks remote OpenAI-compatible endpoints."
        )

    chat_col, side_col = st.columns([1.7, 1], gap="large")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None
    if "last_trace" not in st.session_state:
        st.session_state.last_trace = []
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []

    with chat_col:
        st.markdown('<div class="panel"><h3>Conversation</h3>', unsafe_allow_html=True)
        if not st.session_state.messages:
            st.markdown(
                '<div class="empty">No questions yet.<br><br>'
                "Load the sample manuals, then ask about a limit, a procedure, or a calculation. "
                "Answers cite local chunks. Tool calls appear on the right.</div>",
                unsafe_allow_html=True,
            )
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        st.markdown("</div>", unsafe_allow_html=True)

        prompt = st.chat_input("Ask about the local documents")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            try:
                result = agent.query(
                    question=prompt,
                    conversation_id=st.session_state.conversation_id,
                )
            except Exception as exc:
                logger.exception("Query failed")
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "The question could not be answered. Check the knowledge base and try a shorter prompt.",
                    }
                )
                st.caption(str(exc))
            else:
                st.session_state.conversation_id = result["conversation_id"]
                st.session_state.last_sources = result.get("sources") or []
                st.session_state.last_trace = result.get("tool_trace") or []
                st.session_state.messages.append(
                    {"role": "assistant", "content": result["answer"]}
                )
            st.rerun()

    with side_col:
        st.markdown('<div class="panel"><h3>Evidence</h3>', unsafe_allow_html=True)
        sources = st.session_state.last_sources
        if not sources:
            st.markdown(
                '<div class="empty">Citations appear after a question that matches the knowledge base.</div>',
                unsafe_allow_html=True,
            )
        for src in sources[:6]:
            st.markdown(
                f'<div class="src"><div class="name">{src.get("source", "source")}</div>'
                f'<div class="meta">similarity {src.get("score", 0)}</div></div>',
                unsafe_allow_html=True,
            )
            st.caption((src.get("snippet") or "")[:280])
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="panel"><h3>Tool activity</h3>', unsafe_allow_html=True)
        trace = st.session_state.last_trace
        if not trace:
            st.markdown(
                '<div class="empty">No tools ran on the last turn. The demo model answers from retrieved text unless a TOOL line is requested.</div>',
                unsafe_allow_html=True,
            )
        for step in trace:
            label = "ok" if step.get("ok") else "failed"
            st.markdown(f"**{step.get('tool')}** · {label}")
            st.caption(str(step.get("output", ""))[:400])
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
