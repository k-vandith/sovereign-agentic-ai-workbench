"""Sovereign Workbench: a guided, local-first document Q&A workspace."""
from __future__ import annotations

import html
import importlib.util
import logging
import shutil
from pathlib import Path

import pandas as pd
import streamlit as st

from src.agent import RAGAgent
from src.agent.rbac import Role
from src.config import get_settings
from src.document.io import build_answer_pdf, build_answer_report, build_sample_archive
from src.document.uploads import ingest_uploaded_files
from src.llm import DemoLLM, OllamaLLM
from src.ui.components import brand_header, flow_steps, kpi, page_intro, section
from src.ui.theme import get_css

ROOT = Path(__file__).resolve().parents[2]
PAGES = [
    "Overview",
    "Chat & Evidence",
    "Document Library",
    "Tool Trace",
    "Audit Log",
    "Settings & Glossary",
]
BACKENDS = ["Demo · offline", "Ollama · local model"]
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
    default_backend = "Ollama · local model" if isinstance(agent.llm, OllamaLLM) else "Demo · offline"
    st.session_state.setdefault("wb_page", "Overview")
    st.session_state.setdefault("wb_theme", "Dark")
    st.session_state.setdefault("wb_backend", default_backend)
    st.session_state.setdefault("wb_messages", [])
    st.session_state.setdefault("wb_last_result", None)
    st.session_state.setdefault("wb_last_trace", [])
    st.session_state.setdefault("wb_last_upload_results", [])
    st.session_state.setdefault("wb_ingest_results", [])
    st.session_state.setdefault("wb_tool_history", [])


def _switch_page(page: str) -> None:
    st.session_state["wb_page"] = page


def _go_documents() -> None:
    _switch_page("Document Library")


def _go_chat() -> None:
    _switch_page("Chat & Evidence")


def _status_pills(agent: RAGAgent, model_available: bool) -> str:
    backend = type(agent.llm).__name__
    ready_label = "Ready" if model_available else "Not reachable"
    readiness = "ok" if model_available else "warn"
    mode = "Offline policy ON" if agent.settings.offline_mode else "Offline policy OFF"
    return (
        f'<span class="wb-pill {readiness}"><strong>{html.escape(ready_label)}</strong> '
        f'{html.escape(backend)}</span>'
        f'<span class="wb-pill"><strong>{html.escape(mode)}</strong></span>'
        '<span class="wb-pill"><strong>CPU</strong> processing</span>'
        f'<span class="wb-pill"><strong>{agent.store.count()}</strong> indexed chunks</span>'
        f'<span class="wb-pill"><strong>{html.escape(agent.principal.role.value)}</strong> role</span>'
    )


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
            st.success(f"Indexed {successful} sample document(s). You can now ask a question in Chat & Evidence.")
        else:
            st.warning("The sample pack could not be loaded. Follow the instruction shown in the result details.")
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
    page_intro(
        "Overview",
        "Ask questions about local manuals and procedures, inspect the source evidence, and keep a readable record of tool activity.",
        "Start by loading the fictional sample pack or uploading your own files. Ask one concrete question. Open each source excerpt to inspect the evidence behind the answer.",
    )
    st.markdown(
        '<div class="wb-hero"><div class="wb-eyebrow">PRIVATE KNOWLEDGE · LOCAL AI</div>'
        '<h1>Ask your documents. Keep control of your data.</h1>'
        '<p>A practical workbench for trainers, engineers and operational teams who need answers from confidential manuals, safety protocols and process notes without sending document content to a hosted AI service.</p>'
        '<span class="wb-tag">Local-first workflow</span><span class="wb-tag">Demo or local Ollama</span>'
        '<span class="wb-tag">Source excerpts included</span></div>',
        unsafe_allow_html=True,
    )
    flow_steps()
    sources = agent.store.list_sources()
    columns = st.columns(4)
    with columns[0]:
        kpi("Documents indexed", len(sources), "Distinct source files in the local index")
    with columns[1]:
        kpi("Text chunks", agent.store.count(), "Small passages used to find relevant evidence")
    with columns[2]:
        kpi("Model backend", "Demo" if isinstance(agent.llm, DemoLLM) else "Ollama", "Selected answer-generation mode")
    with columns[3]:
        kpi("Model status", "Ready" if model_available else "Unavailable", "Whether the selected backend can respond now")

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        section("Who is this for?", "A guided starting point for people who need to find facts in local documents.")
        st.markdown(
            '<div class="wb-card"><div class="wb-card-title">Industrial trainers & students</div>'
            '<div class="wb-card-copy">Explore procedures and learn how to trace an answer back to its source.</div></div>'
            '<div class="wb-card"><div class="wb-card-title">Operators & engineers</div>'
            '<div class="wb-card-copy">Find relevant operating limits, maintenance guidance and documented process notes.</div></div>'
            '<div class="wb-card"><div class="wb-card-title">Privacy-conscious teams</div>'
            '<div class="wb-card-copy">Keep source documents in a local folder and select Demo or a local Ollama model.</div></div>',
            unsafe_allow_html=True,
        )
    with right:
        section("Start in two ways", "Choose a safe fictional pack or your own files.")
        st.write("**1. Try with sample data**")
        st.caption("The included pack contains fictional equipment, process and safety notes.")
        _render_load_sample(agent, "wb_load_sample_overview")
        st.divider()
        st.write("**2. Upload your own data**")
        st.caption("PDF, DOCX, TXT, Markdown, CSV, JSON and common image formats are accepted.")
        st.button("Upload your own documents", key="wb_upload_cta", on_click=_go_documents, use_container_width=True)
        st.button("Ask a question", key="wb_chat_cta", on_click=_go_chat, use_container_width=True)

    if not sources:
        st.info("Your local library is empty. Load the sample pack or add a document to begin.")
    else:
        section("Current library", "These source names and chunk counts describe what the local index can search.")
        st.dataframe(
            pd.DataFrame(sources)[["source", "chunks", "preview"]].rename(
                columns={"source": "Document", "chunks": "Indexed passages", "preview": "Example excerpt"}
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("The raw uploaded copy is cleaned up after ingestion; searchable text and embeddings remain in the local vector store.")


def _render_documents(agent: RAGAgent) -> None:
    page_intro(
        "Document Library",
        "Upload several documents at once, validate their file types and size, and index their content for later questions.",
        "Choose files, review the supported types, then select Index selected documents. The app reports status for each file. Raw upload copies are temporary; indexed text and embeddings stay in the local vector store until you clear it.",
    )
    st.download_button(
        "Download sample document pack (.zip)",
        data=build_sample_archive(ROOT / "data" / "sample"),
        file_name="sovereign_workbench_sample_documents.zip",
        mime="application/zip",
        help="A ZIP of fictional industrial notes you can load as a sample case.",
    )
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
        signature = tuple((upload.name, int(getattr(upload, "size", 0) or 0)) for upload in uploads)
        already_processed = signature == st.session_state.get("wb_processed_signature")
        if already_processed:
            st.info("This exact selection was already processed. Clear the selection below to choose another batch.")
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
        st.button(
            "Clear selected files",
            on_click=lambda: st.session_state.update({"wb_upload_files": [], "wb_processed_signature": None}),
            key="wb_clear_uploads",
        )
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
    page_intro(
        "Chat & Evidence",
        "Ask a specific question about the indexed documents and open the source excerpts that support the answer.",
        "Questions are matched against local document passages. The selected model drafts the reply. Open the source evidence panels to check names, measurements and context against the excerpt.",
    )
    if not agent.store.count():
        st.info("The library is empty. Load the fictional sample pack or upload documents first.")
        st.button("Open document library", on_click=_go_documents, type="primary")
    if not model_available:
        st.warning("The selected model backend is not reachable. Choose Demo mode or start the local Ollama service, then recheck model status.")
    for message in st.session_state["wb_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            for source in message.get("sources", [])[:8]:
                _source_preview(source)
            trace = message.get("tool_trace", [])
            if trace:
                with st.expander(f"Tool activity · {len(trace)} call(s)"):
                    for step in trace:
                        st.markdown(f"**{html.escape(str(step.get('tool', 'tool')))}** · {'OK' if step.get('ok') else 'Needs review'}")
                        st.caption(str(step.get("output", ""))[:500])
    prompt = st.chat_input("Ask about a safety limit, procedure, maintenance task or process note")
    if prompt:
        st.session_state["wb_messages"].append({"role": "user", "content": prompt})
        conversation_id = st.session_state.get("wb_conversation_id")
        try:
            with st.spinner("Searching local passages and drafting an answer…"):
                result = agent.query(question=prompt, conversation_id=conversation_id)
        except Exception:
            logger.exception("Local question answering failed")
            st.session_state["wb_messages"].append({
                "role": "assistant",
                "content": "I could not complete this question. Check that documents are indexed and the selected backend is ready, then try a shorter question.",
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
        section("Answer summary & report", "A compact record of the latest question, evidence and tool activity.")
        _answer_downloads(last_result)
    st.caption("Prompt ideas: What is the operating pressure? Summarise the safety protocol. Which maintenance interval is stated in the manual?")


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
        data=__import__("json").dumps(rows, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="sovereign_workbench_audit.json",
        mime="application/json",
    )
    st.caption(f"{len(rows)} event(s) shown. Audit records are local files under the configured data directory.")


def _render_settings(agent: RAGAgent, model_available: bool) -> None:
    page_intro(
        "Settings & Glossary",
        "Understand which local capabilities are active, how the selected backend works, and what the workbench's technical terms mean.",
        "Use Demo for an immediate no-model service test or Ollama for local open-weight generation. Embedding weights may need to be downloaded the first time they are used; document content and prompts are processed by this local app.",
    )
    section("Runtime configuration", "Visible settings and backend readiness for this application session.")
    cols = st.columns(3)
    with cols[0]:
        kpi("Backend", "Demo" if isinstance(agent.llm, DemoLLM) else "Ollama", "Answers use the selected generator")
    with cols[1]:
        kpi("Offline policy", "On" if agent.settings.offline_mode else "Off", "The environment setting for network policy")
    with cols[2]:
        kpi("Current role", agent.principal.role.value, "Core permissions applied to actions")
    if isinstance(agent.llm, OllamaLLM) and not model_available:
        st.warning("Ollama is not reachable at the configured local URL. Start Ollama, pull the configured model, and use Recheck model status in the sidebar.")
        st.code("ollama pull llama3.2:1b", language="bash")
    if isinstance(agent.llm, DemoLLM):
        st.info("Demo mode returns a template-based response that can quote retrieved local passages. Select Ollama for a real local language model.")
    with st.expander("How to change the configured model"):
        st.write("The sidebar backend selector switches this process between Demo and the local Ollama client. To change the default after restart, set LLM_BACKEND in .env. The UI intentionally does not expose paid API credentials.")
        st.code("LLM_BACKEND=demo\n# or\nLLM_BACKEND=ollama\nOLLAMA_MODEL=llama3.2:1b", language="dotenv")

    section("Optional capability status", "Some features need extra packages or local system tools. Missing features are described rather than silently substituted.")
    capabilities = [
        ("CPU sentence embeddings", importlib.util.find_spec("sentence_transformers") is not None, "Installed with the base requirements; first use may download the configured model weights."),
        ("FAISS vector index", importlib.util.find_spec("faiss") is not None, "When unavailable, the vector store uses its NumPy similarity fallback."),
        ("PDF answer reports", importlib.util.find_spec("reportlab") is not None, "Enable with pip install '.[reports]' from the repository root."),
        ("Image OCR", importlib.util.find_spec("pytesseract") is not None and shutil.which("tesseract") is not None, "Not active by default. Install Tesseract OCR and pytesseract to extract text from image files."),
        ("Ollama local generation", isinstance(agent.llm, OllamaLLM) and model_available, "Install Ollama locally, pull the configured model, then select it from the sidebar."),
    ]
    st.dataframe(
        pd.DataFrame([{
            "Capability": name,
            "Status": "Active" if active else "Optional / not active",
            "What it means / how to enable": description,
        } for name, active, description in capabilities]),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("Images are currently registered as metadata only. OCR and visual interpretation are not implied by a successful image upload.")
    section("Plain-English glossary", "Short definitions for the terms used throughout the workbench.")
    glossary = {
        "RAG (retrieval-augmented generation)": "The app finds related text passages first, then gives those passages to the language model as context.",
        "Chunk / passage": "A short piece of a document that can be matched against a question.",
        "Embedding": "A numeric representation of text used to compare its meaning with a question.",
        "Vector store": "The local index that keeps document passages and their numeric representations for search.",
        "Citation / source excerpt": "A file name and passage retrieved as evidence relevant to an answer; always verify it against the original.",
        "Tool trace": "A record of utility calls, their inputs, outputs and success status during a query.",
        "Audit trail": "A local log of ingestion, question, answer and tool events.",
        "Ollama": "A local runtime for running compatible open-weight language models on your own machine.",
        "Demo backend": "A lightweight template-based generator for trying the workflow without a language model service.",
        "RBAC (role-based access control)": "Core permissions that limit selected actions based on the configured local role.",
        "OCR (optical character recognition)": "Software that attempts to extract text from pixels in an image or scanned page.",
        "Offline mode": "A configuration flag intended to signal a no-remote-service policy; verify the runtime and model downloads before using in a disconnected environment.",
    }
    for term, definition in glossary.items():
        with st.expander(term):
            st.write(definition)
    st.warning("This app is not an identity provider and does not encrypt the local data directory automatically. Protect the device, index and audit log with operating-system permissions.")


def main() -> None:
    st.set_page_config(
        page_title="Sovereign Workbench · Private Knowledge",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    logging.basicConfig(level=get_settings().log_level)
    agent = _agent()
    _init_state(agent)
    st.markdown(get_css(st.session_state["wb_theme"]), unsafe_allow_html=True)

    backend = st.session_state["wb_backend"]
    if st.session_state.get("wb_active_backend") != backend:
        if backend == "Demo · offline":
            agent.llm = DemoLLM()
        else:
            agent.llm = OllamaLLM()
        st.session_state["wb_active_backend"] = backend
        st.session_state.pop("wb_model_available", None)
    if "wb_model_available" not in st.session_state:
        try:
            st.session_state["wb_model_available"] = bool(agent.llm.is_available())
        except Exception:
            st.session_state["wb_model_available"] = False
    model_available = bool(st.session_state["wb_model_available"])

    with st.sidebar:
        st.markdown(
            '<div class="wb-brand"><div class="wb-mark"><svg viewBox="0 0 64 64" aria-label="Sovereign Workbench">'
            '<path d="M9 17 32 7 55 18 55 45 32 57 9 45Z" fill="#173944" stroke="#55c7d9" stroke-width="2"/>'
            '<path d="M18 25 32 17 46 25 46 39 32 47 18 39Z" fill="none" stroke="#9beaf4" stroke-width="2.5"/>'
            '<circle cx="32" cy="32" r="5" fill="#55c7d9"/></svg></div>'
            '<div class="wb-wordmark">SOVEREIGN WORKBENCH<small>Private knowledge · Local AI</small></div></div>',
            unsafe_allow_html=True,
        )
        st.caption("Industrial documents · Local-first workflow")
        st.divider()
        st.radio("Workspace", PAGES, key="wb_page")
        st.divider()
        st.selectbox("Model backend", BACKENDS, key="wb_backend", help="Demo uses a template-based generator. Ollama sends prompts only to your configured local Ollama service.")
        st.selectbox("Appearance", ["Dark", "Light"], key="wb_theme", help="Change the workspace color contrast.")
        st.markdown(
            f'<span class="wb-pill {"ok" if model_available else "warn"}"><strong>{"Ready" if model_available else "Not reachable"}</strong> model status</span>',
            unsafe_allow_html=True,
        )
        st.caption("Offline policy: " + ("ON" if agent.settings.offline_mode else "OFF"))
        if st.button("Recheck model status", use_container_width=True, help="Check the selected local backend again."):
            try:
                st.session_state["wb_model_available"] = bool(agent.llm.is_available())
            except Exception:
                st.session_state["wb_model_available"] = False
            st.rerun()
        st.divider()
        st.caption(f"{agent.store.count()} indexed passage(s) · role: {agent.principal.role.value}")

    page = st.session_state["wb_page"]
    brand_header(page, _status_pills(agent, model_available))

    if page == "Overview":
        _render_overview(agent, model_available)
    elif page == "Chat & Evidence":
        _render_chat(agent, model_available)
    elif page == "Document Library":
        _render_documents(agent)
    elif page == "Tool Trace":
        _render_tools(agent)
    elif page == "Audit Log":
        _render_audit(agent)
    else:
        _render_settings(agent, model_available)


if __name__ == "__main__":
    main()
