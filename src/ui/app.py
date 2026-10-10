"""Sovereign Workbench: a guided, local-first document Q&A workspace."""
from __future__ import annotations

import html
import importlib.util
import json
import logging
from pathlib import Path

import pandas as pd
import streamlit as st

from src.agent import RAGAgent
from src.config import get_settings
from src.document.io import build_answer_pdf, build_answer_report, build_sample_archive
from src.document.uploads import ingest_uploaded_files
from src.llm import OpenAICompatibleLLM
from src.ui.components import brand_header, page_intro, section
from src.ui.theme import get_css

ROOT = Path(__file__).resolve().parents[2]
PAGES = ["Workspace", "Documents", "Settings"]
UPLOAD_TYPES = [
    "pdf", "docx", "txt", "md", "csv", "json", "log",
    "png", "jpg", "jpeg", "webp", "bmp",
]

logger = logging.getLogger(__name__)


def _agent() -> RAGAgent:
    @st.cache_resource
    def _build() -> RAGAgent:
        return RAGAgent()

    return _build()


def _init_state(agent: RAGAgent) -> None:
    st.session_state.setdefault("wb_page", "Workspace")
    if st.session_state.get("wb_page") not in PAGES:
        st.session_state["wb_page"] = "Workspace"
    st.session_state.setdefault("wb_messages", [])
    st.session_state.setdefault("wb_last_result", None)
    st.session_state.setdefault("wb_last_trace", [])
    st.session_state.setdefault("wb_last_upload_results", [])
    st.session_state.setdefault("wb_ingest_results", [])
    st.session_state.setdefault("wb_tool_history", [])


def _switch_page(page: str) -> None:
    st.session_state["wb_page"] = page


def _go_documents() -> None:
    _switch_page("Documents")


def _go_chat() -> None:
    _switch_page("Workspace")


def _status_pills(agent: RAGAgent, model_available: bool) -> str:
    if isinstance(agent.llm, OpenAICompatibleLLM) and model_available:
        return '<span class="wb-pill ok"><strong>API configured</strong></span>'
    return '<span class="wb-pill"><strong>Preview mode</strong></span>'


def _sample_summary(agent: RAGAgent) -> dict:
    sample_dir = ROOT / "data" / "sample"
    if not sample_dir.is_dir():
        return {
            "files_processed": 0,
            "details": [{
                "file": "Sample pack",
                "error": "Sample files are not present. Run python scripts/generate_demo_data.py and try again.",
            }],
        }
    return agent.ingest_directory(sample_dir)


def _render_load_sample(agent: RAGAgent, button_key: str = "wb_load_sample") -> None:
    if st.button("Load sample documents", type="primary", key=button_key, help="Index the fictional industrial safety, process and equipment notes included with this project."):
        with st.spinner("Indexing fictional sample documents locally…"):
            summary = _sample_summary(agent)
        st.session_state["wb_ingest_results"] = summary.get("details", [])
        successful = sum(1 for item in summary.get("details", []) if "error" not in item)
        if successful:
            st.session_state["wb_sample_notice"] = f"Indexed {successful} sample document(s). You can now ask a question in Workspace."
        else:
            st.session_state["wb_sample_notice"] = "The sample pack could not be loaded. Check the result details."
        st.rerun()


def _source_preview(source: dict) -> None:
    name = str(source.get("source", "source"))
    score = source.get("score")
    label = f"Open source excerpt · {name}"
    if score is not None:
        label += f" · match {float(score):.3f}"
    with st.expander(label):
        st.caption("Retrieval similarity helps locate relevant text; it is not a confidence score or proof.")
        st.write(source.get("snippet") or "No excerpt was returned.")
        st.caption("Check the original document before relying on safety-critical details.")


def _answer_downloads(result: dict) -> None:
    question = str(result.get("question", ""))
    answer = str(result.get("answer", ""))
    sources = result.get("sources") or []
    trace = result.get("tool_trace") or []
    backend = str(result.get("backend", "Unknown"))
    st.markdown(
        '<div class="wb-note"><strong>Answer summary</strong><br>'
        f'Analysed question: {html.escape(question[:260])}<br>'
        f'Retrieved source excerpts: {len(sources)} · Tool calls: {len(trace)}<br>'
        'Meaning: this answer is generated from retrieved local text and the selected backend. '
        'Verify key details in the original source document before acting.</div>',
        unsafe_allow_html=True,
    )
    report = build_answer_report(question, answer, sources, trace, backend)
    left, right = st.columns(2)
    left.download_button(
        "Download HTML answer report",
        data=report.encode("utf-8"),
        file_name="sovereign_workbench_answer_report.html",
        mime="text/html",
        use_container_width=True,
        key="wb_answer_report_html",
    )
    if importlib.util.find_spec("reportlab") is not None:
        pdf = build_answer_pdf(question, answer, sources, trace, backend)
        if pdf:
            right.download_button(
                "Download PDF answer report",
                data=pdf,
                file_name="sovereign_workbench_answer_report.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="wb_answer_report_pdf",
            )
        else:
            right.caption("PDF generation failed. HTML export remains available.")
    else:
        right.caption("PDF export is optional. Enable it with: pip install '.[reports]'.")


def _render_overview(agent: RAGAgent, model_available: bool) -> None:
    sources = agent.store.list_sources()
    st.markdown(
        '<div class="wb-simple-hero">'
        '<div class="wb-kicker">SOVEREIGN WORKBENCH / PRIVATE KNOWLEDGE</div>'
        '<h1>Answers with <em>receipts.</em></h1>'
        '<p>Ask a question. Get a focused answer. Open the source passages that support it.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.markdown("#### Start with your documents")
        st.caption("Use the fictional sample pack or index files from your own machine.")
        if not sources:
            _render_load_sample(agent, "wb_load_sample_home")
            st.button("Add your documents", key="wb_add_docs_home", on_click=_go_documents, use_container_width=True)
        else:
            st.markdown(f"**{len(sources)} documents** · **{agent.store.count()} passages** ready to search")
            st.button("Manage documents", key="wb_manage_docs_home", on_click=_go_documents, use_container_width=True)
    with right:
        st.markdown("#### Better answers with an API")
        st.caption("Connect an API provider in Settings for stronger, more useful answers. Preview mode is for trying the workflow.")
        if not isinstance(agent.llm, OpenAICompatibleLLM):
            st.button("Set up API", key="wb_setup_api_home", on_click=lambda: _switch_page("Settings"), type="primary", use_container_width=True)
        else:
            st.caption("API settings detected. Questions and relevant document passages are sent to your configured provider for answer generation.")
    st.divider()


def _render_documents(agent: RAGAgent) -> None:
    page_intro(
        "Documents",
        "Add the files you want to search, preview them, and index their content.",
        "Choose files, check the preview and per-file status, then ask questions from Workspace.",
    )
    notice = st.session_state.pop("wb_sample_notice", None)
    if notice:
        st.success(notice)
    sample_cols = st.columns(2)
    with sample_cols[0]:
        st.download_button(
            "Download sample document pack (.zip)",
            data=build_sample_archive(ROOT / "data" / "sample"),
            file_name="sovereign_workbench_sample_documents.zip",
            mime="application/zip",
            help="A ZIP of fictional industrial notes you can load as a sample case.",
            use_container_width=True,
        )
    with sample_cols[1]:
        _render_load_sample(agent, "wb_load_sample_docs")
    uploads = st.file_uploader(
        "Choose one or more documents",
        type=UPLOAD_TYPES,
        accept_multiple_files=True,
        key="wb_upload_files",
        help="PDF, DOCX, TXT, MD, CSV, JSON, LOG, PNG, JPG, WEBP and BMP. Maximum 25 MB per file and 100 MB per batch.",
    )
    if uploads:
        preview_rows = []
        total_bytes = 0
        for upload in uploads:
            size = int(getattr(upload, "size", 0) or 0)
            total_bytes += size
            preview_rows.append({
                "File": upload.name,
                "Type": Path(upload.name).suffix.lower().lstrip(".").upper(),
                "Size (MB)": round(size / (1024 * 1024), 2),
                "Pre-check": "Within limit" if size <= 25 * 1024 * 1024 else "Over 25 MB limit",
            })
        st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)
        st.caption(f"{len(uploads)} file(s) selected · {total_bytes / (1024 * 1024):.2f} MB total · Batch limit 100 MB.")
        with st.expander("Preview selected files"):
            for upload in uploads[:5]:
                suffix = Path(upload.name).suffix.lower()
                st.markdown(f"**{html.escape(upload.name)}**")
                if suffix in {".txt", ".md", ".csv", ".json", ".log"}:
                    preview = upload.getvalue().decode("utf-8", errors="replace")[:1800]
                    st.code(preview or "The file has no readable text.", language="text")
                elif suffix in {".pdf", ".docx"}:
                    st.caption("Text preview is available in the indexed library after successful extraction.")
                else:
                    st.caption("Image upload is supported. Text extraction from images requires optional local OCR.")
        signature = tuple((upload.name, int(getattr(upload, "size", 0) or 0)) for upload in uploads)
        already_processed = signature == st.session_state.get("wb_processed_signature")
        if already_processed:
            st.info("This exact selection was already processed. Use Re-index same selection to run it again.")
        if already_processed:
            st.button(
                "Re-index same selection",
                key="wb_reindex_selection",
                on_click=lambda: st.session_state.pop("wb_processed_signature", None),
            )
        if st.button("Index selected documents", type="primary", disabled=already_processed, key="wb_index_uploads"):
            with st.spinner("Validating and indexing documents in a temporary session folder…"):
                results = ingest_uploaded_files(agent, uploads)
            st.session_state["wb_last_upload_results"] = results
            st.session_state["wb_processed_signature"] = signature
            succeeded = sum(1 for item in results if item.get("status") == "Indexed")
            if succeeded:
                st.success(f"Indexed {succeeded} document(s). Raw temporary files have been removed.")
            if succeeded < len(uploads):
                st.warning("Some files were not indexed. Check the per-file result table below.")
            st.rerun()
    with st.expander("What happens to uploaded data?"):
        st.write(
            "Each selected file is checked against supported extensions, a 25 MB per-file limit and a 100 MB batch limit. "
            "Files are written only to a temporary folder while extraction runs, and the folder is deleted after processing. "
            "Extracted chunks and local embeddings are persisted in the configured vector-store folder so the library remains searchable."
        )
        st.warning("Images are registered as indexed image metadata only by default; this does not perform OCR or semantic image understanding.")
    recent_results = st.session_state.get("wb_last_upload_results", [])
    ingest_results = st.session_state.get("wb_ingest_results", [])
    combined_results = recent_results or ingest_results
    if combined_results:
        section("Latest ingestion results", "A per-file status table explains which sources are ready to search.")
        st.dataframe(pd.DataFrame(combined_results), use_container_width=True, hide_index=True)
    sources = agent.store.list_sources()
    section("Indexed document library", "Each row represents a source document with passages available to local retrieval.")
    if sources:
        st.dataframe(
            pd.DataFrame(sources).rename(columns={"source": "Document", "chunks": "Passages", "preview": "Example excerpt", "pages": "Pages represented"}),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(f"{len(sources)} source document(s) · {agent.store.count()} searchable passage(s).")
        if st.button("Clear all indexed content", help="Removes the current searchable index. It does not delete the repository sample files."):
            try:
                agent.clear_knowledge_base()
            except PermissionError:
                st.error("Your configured role cannot clear the knowledge base. Ask the local administrator to enable the operator role.")
            else:
                st.session_state["wb_messages"] = []
                st.session_state["wb_last_result"] = None
                st.success("The local search index was cleared.")
                st.rerun()
    else:
        st.info("No indexed documents yet. Load the sample pack or select files above.")


def _render_chat(agent: RAGAgent, model_available: bool) -> None:
    st.markdown('<div class="wb-chat-title"><div class="wb-kicker">ASK YOUR DOCUMENTS</div><h2>What do you need to know?</h2></div>', unsafe_allow_html=True)
    if not agent.store.count():
        st.info("Load the sample pack or add a document before asking a question.")
    elif not isinstance(agent.llm, OpenAICompatibleLLM):
        st.caption("Preview mode is limited. For better answers, configure an API in Settings.")
    else:
        st.caption("Answers use retrieved passages. Your question and relevant excerpts are sent to the configured API provider.")
    for message in st.session_state["wb_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            for source in message.get("sources", [])[:6]:
                _source_preview(source)
            trace = message.get("tool_trace", [])
            if trace:
                with st.expander(f"Tool activity · {len(trace)} call(s)"):
                    for step in trace:
                        st.markdown(f"**{html.escape(str(step.get('tool', 'tool')))}** · {'OK' if step.get('ok') else 'Needs review'}")
                        st.caption(str(step.get("output", ""))[:400])
    prompt = st.chat_input(
        "Ask about a procedure, limit, maintenance task or note…",
        disabled=not bool(agent.store.count()),
    )
    if prompt:
        st.session_state["wb_messages"].append({"role": "user", "content": prompt})
        conversation_id = st.session_state.get("wb_conversation_id")
        try:
            with st.spinner("Searching sources and preparing an answer…"):
                result = agent.query(question=prompt, conversation_id=conversation_id)
        except Exception:
            logger.exception("Document question failed")
            st.session_state["wb_messages"].append({
                "role": "assistant",
                "content": "I couldn't complete that question. Check your API setup and indexed sources, then try again.",
            })
        else:
            result["question"] = prompt
            st.session_state["wb_conversation_id"] = result.get("conversation_id")
            st.session_state["wb_last_result"] = result
            st.session_state["wb_last_trace"] = result.get("tool_trace") or []
            st.session_state["wb_messages"].append({
                "role": "assistant",
                "content": result.get("answer", "No answer was returned."),
                "sources": result.get("sources") or [],
                "tool_trace": result.get("tool_trace") or [],
            })
        st.rerun()
    last_result = st.session_state.get("wb_last_result")
    if last_result:
        with st.expander("Export the latest answer"):
            _answer_downloads(last_result)


def _render_tools(agent: RAGAgent) -> None:
    page_intro(
        "Tool Trace",
        "Inspect the most recent tool calls and run a small local utility directly when you need to validate a calculation or search a source.",
        "Each trace shows the tool name, input, output and whether it succeeded. Calculator accepts basic arithmetic only. Document search retrieves local passages. The image tool reports local file metadata and is not semantic vision.",
    )
    trace_items = list(st.session_state.get("wb_last_trace", []))
    for message in st.session_state.get("wb_messages", []):
        trace_items.extend(message.get("tool_trace", []))
    trace_items.extend(st.session_state.get("wb_tool_history", []))
    unique = []
    seen = set()
    for item in trace_items:
        marker = (item.get("tool"), item.get("input"), item.get("output"), item.get("ok"))
        if marker not in seen:
            seen.add(marker)
            unique.append(item)
    section("Recent tool activity", "An auditable, human-readable trace from the last chat and local manual runs.")
    if unique:
        st.dataframe(
            pd.DataFrame([{
                "Tool": item.get("tool"),
                "Status": "OK" if item.get("ok") else "Needs review",
                "Input": item.get("input", ""),
                "Output": item.get("output", ""),
            } for item in unique[-30:]]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No tools have run yet. Ask a question that needs a calculation or use the local utility runner below.")
    section("Run a local utility", "This runner uses the same built-in tools and does not contact a hosted model.")
    with st.form("wb_tool_form"):
        tool = st.selectbox(
            "Utility",
            agent.tools.names(),
            help="Calculator supports basic arithmetic; document search queries the current index; summarizer returns a compact text summary; image analysis reads metadata from a local path.",
        )
        argument = st.text_area("Input", placeholder="Example: 2 * (14 + 3)", help="For image analysis, enter a path to a local image file.")
        submitted = st.form_submit_button("Run utility", type="primary")
    if submitted:
        try:
            if not agent.principal.can("use_tools"):
                raise PermissionError("The configured role cannot run tools.")
            with st.spinner("Running the local utility…"):
                result = agent.tools.run(tool, argument)
            record = {"tool": result.name, "input": result.input, "output": result.output, "ok": result.ok}
            st.session_state["wb_tool_history"].append(record)
            agent.audit.record(
                "tool_call", user=agent.principal.name, tool=result.name,
                input=result.input[:500], output=result.output[:1000], ok=result.ok,
            )
            if result.ok:
                st.success("Utility completed.")
            else:
                st.warning("The utility returned a problem. Review the result below.")
            st.write(result.output)
        except PermissionError as exc:
            st.error(str(exc))


def _render_audit(agent: RAGAgent) -> None:
    page_intro(
        "Audit Log",
        "Review local records of document ingestion, user questions, generated answers and tool activity.",
        "The core agent restricts audit access to an administrator role. Configure the local role in .env and restart the app; this simple local workbench does not provide user authentication or multi-user identity management.",
    )
    try:
        rows = agent.view_audit(limit=200)
    except PermissionError:
        st.warning("Audit access is restricted for the current role.")
        st.markdown(
            '<div class="wb-card"><div class="wb-card-title">Enable local administrator access</div>'
            '<div class="wb-card-copy">Set DEFAULT_ROLE=admin in your local .env file, restart the workbench, and open Audit Log again. This setting is a local role configuration, not an identity provider.</div></div>',
            unsafe_allow_html=True,
        )
        st.code("DEFAULT_ROLE=admin", language="dotenv")
        return
    section("Recent events", "Latest local audit entries; question and answer text may be included, so protect this machine and its files.")
    if not rows:
        st.info("No audit events recorded yet. Ingest a document or ask a question to create the first event.")
        return
    frame = pd.DataFrame(rows)
    preferred = [column for column in ("ts", "event", "user", "role", "file", "chunks", "tool", "ok", "question") if column in frame.columns]
    remainder = [column for column in frame.columns if column not in preferred]
    st.dataframe(frame[preferred + remainder], use_container_width=True, hide_index=True)
    st.download_button(
        "Download visible audit events (.json)",
        data=json.dumps(rows, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="sovereign_workbench_audit.json",
        mime="application/json",
    )
    st.caption(f"{len(rows)} event(s) shown. Audit records are local files under the configured data directory.")


def _render_settings(agent: RAGAgent, model_available: bool) -> None:
    st.markdown("## Settings")
    st.markdown("### Answer quality")
    if isinstance(agent.llm, OpenAICompatibleLLM) and model_available:
        st.success("API configuration detected. The app will use it to generate answers from relevant document passages.")
    else:
        st.info("Recommended: connect an API provider for better answers. Preview mode lets you test uploads, search and source excerpts but is not a full reasoning model.")
    st.markdown("Create a local `.env` file in the project root, add your provider details, and restart the app.")
    st.code(
        "LLM_BACKEND=openai_compatible\n"
        "OPENAI_COMPATIBLE_BASE_URL=https://api.openai.com\n"
        "OPENAI_COMPATIBLE_API_KEY=your_api_key_here\n"
        "OPENAI_COMPATIBLE_MODEL=your_model_name\n"
        "OFFLINE_MODE=false",
        language="dotenv",
    )
    st.caption("Use an API provider that supports the OpenAI chat-completions format. Keep your API key in .env, never in source code or a committed file.")
    st.markdown("### Data handling")
    st.write("Files are indexed locally. When API mode is enabled, your question and the relevant retrieved passages are sent to the configured provider to generate the answer. Review your provider's data policy before uploading sensitive material.")
    st.write("Raw upload copies are temporary and removed after ingestion. The local search index and audit records remain on this machine until you clear or delete them.")
    with st.expander("Advanced: tool activity"):
        _render_tools(agent)
    with st.expander("Advanced: audit log"):
        _render_audit(agent)
    with st.expander("Glossary"):
        glossary = {
            "Source passage": "A section of a document retrieved because it may help answer your question.",
            "RAG": "Retrieval-augmented generation: search for relevant text first, then ask a language model to answer using that context.",
            "Embedding": "A numeric representation of text used by the search index to find related passages.",
            "Audit log": "A local record of document and tool activity. It may contain questions and answers.",
        }
        for term, definition in glossary.items():
            st.markdown(f"**{term}** — {definition}")
    st.caption("This app does not provide multi-user authentication or automatically encrypt the local data directory. Protect the device and files with operating-system permissions.")


def main() -> None:
    st.set_page_config(
        page_title="Sovereign Workbench",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    logging.basicConfig(level=get_settings().log_level)
    agent = _agent()
    _init_state(agent)
    st.markdown(get_css("dark"), unsafe_allow_html=True)
    try:
        model_available = bool(agent.llm.is_available())
    except Exception:
        model_available = False

    with st.sidebar:
        st.markdown(
            '<div class="wb-brand"><div class="wb-mark"><svg viewBox="0 0 64 64" aria-label="Sovereign Workbench">'
            '<path d="M9 17 32 7 55 18 55 45 32 57 9 45Z" fill="#27321a" stroke="#d4ff59" stroke-width="2"/>'
            '<path d="M18 25 32 17 46 25 46 39 32 47 18 39Z" fill="none" stroke="#efffaf" stroke-width="2.5"/>'
            '<circle cx="32" cy="32" r="5" fill="#d4ff59"/></svg></div>'
            '<div class="wb-wordmark">SOVEREIGN<small>Knowledge, with sources</small></div></div>',
            unsafe_allow_html=True,
        )
        st.radio("Go to", PAGES, key="wb_page")
        st.divider()
        st.caption(f"{len(agent.store.list_sources())} documents · {agent.store.count()} passages")

    page = st.session_state["wb_page"]
    brand_header(page, _status_pills(agent, model_available))
    if page == "Workspace":
        _render_overview(agent, model_available)
        _render_chat(agent, model_available)
    elif page == "Documents":
        _render_documents(agent)
    else:
        _render_settings(agent, model_available)




if __name__ == "__main__":
    main()
