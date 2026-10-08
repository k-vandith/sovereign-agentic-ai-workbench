"""Streamlit chat interface for the Sovereign Agentic AI Workbench."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.config import get_settings
from src.agent import RAGAgent

st.set_page_config(
    page_title="Sovereign Agentic AI Workbench",
    page_icon="🔒",
    layout="wide",
)

settings = get_settings()


@st.cache_resource
def get_agent() -> RAGAgent:
    return RAGAgent()


agent = get_agent()

st.title("🔒 Sovereign Agentic AI Workbench")
st.caption(
    "On-premise • Privacy-first • Open-weight models • Local RAG knowledge base"
)

with st.sidebar:
    st.header("Configuration")
    st.write(f"**LLM Backend:** `{type(agent.llm).__name__}`")
    st.write(f"**Available:** {agent.llm.is_available()}")
    st.write(f"**KB Chunks:** {agent.store.count()}")
    st.divider()
    st.subheader("Upload Document")
    uploaded = st.file_uploader(
        "Supported: PDF, DOCX, TXT, MD, images",
        type=["pdf", "docx", "txt", "md", "csv", "png", "jpg", "jpeg"],
    )
    if uploaded is not None:
        dest = settings.upload_dir / uploaded.name
        dest.write_bytes(uploaded.getvalue())
        with st.spinner("Ingesting…"):
            try:
                result = agent.ingest_file(dest)
                st.success(
                    f"Ingested **{result['chunks_added']}** chunks from `{result['file']}`"
                )
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    st.divider()
    if st.button("Clear Knowledge Base", type="secondary"):
        agent.clear_knowledge_base()
        st.success("Knowledge base cleared.")
        st.rerun()
    st.divider()
    st.markdown(
        """
        **Demo mode** works without GPU or Ollama.
        To use a real model:
        1. Install [Ollama](https://ollama.com)
        2. `ollama pull llama3.2:1b`
        3. Set `LLM_BACKEND=ollama` in `.env`
        """
    )

if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- **{s['source']}** (score: {s['score']})")
                    st.caption(s["snippet"])

if prompt := st.chat_input("Ask a question about your local documents…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = agent.query(
                question=prompt,
                conversation_id=st.session_state.conversation_id,
            )
            st.session_state.conversation_id = result["conversation_id"]
            st.markdown(result["answer"])
            if result["sources"]:
                with st.expander("Sources"):
                    for s in result["sources"]:
                        st.markdown(f"- **{s['source']}** (score: {s['score']})")
                        st.caption(s["snippet"])
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                }
            )
