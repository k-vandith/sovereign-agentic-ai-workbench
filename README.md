# Sovereign Agentic AI Workbench

![Sovereign Workbench product mark](docs/workbench-mark.svg)

**Private knowledge. Local AI. Source-backed answers.**

A guided document Q&A workbench for trainers, engineers and operational teams. Upload manuals and procedures, ask a question, open the source excerpts behind the answer, inspect local tool activity and export a report. The default Demo backend works without a hosted model API key; Ollama is available for local language-model generation.

## Quick start

Python 3.11 or 3.12 is recommended.

### Windows PowerShell

    git clone https://github.com/k-vandith/sovereign-agentic-ai-workbench.git
    cd sovereign-agentic-ai-workbench
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements-dev.txt
    python run.py

### Linux / macOS

    git clone https://github.com/k-vandith/sovereign-agentic-ai-workbench.git
    cd sovereign-agentic-ai-workbench
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install -r requirements-dev.txt
    python run.py

Open http://127.0.0.1:8501. The API can be started separately with python run.py --api and checked at http://127.0.0.1:8000/health.

After your virtual environment is activated, the three-command launch is:

    pip install -r requirements.txt
    python scripts/generate_demo_data.py
    python run.py

## What you can do

- Add multiple PDF, DOCX, TXT, Markdown, CSV, JSON, LOG and common image files in one upload batch.
- Preview the selection, validate file size/type and see a per-file indexing status.
- Try the included fictional safety protocol, equipment manual and process notes.
- Ask questions in a chat-style interface and open clickable source-excerpt panels.
- Inspect tool-call inputs/outputs and read the local audit trail (admin role required).
- Switch between the immediate Demo generator and an Ollama service on your machine.
- Download a self-contained HTML answer report. PDF and OCR are optional features with setup instructions in Settings & Glossary.

## Try it in five steps

1. Start the app and review the Overview. It explains the four-step workflow and shows the current document and model status.
2. Click **Load sample documents**. The fictional industrial safety, pump and process notes are indexed locally.
3. Open **Chat & Evidence** and ask: What is the maximum allowable temperature for Reactor R-12?
4. Open the source excerpt below the answer, then open **Tool Trace** to inspect whether a utility ran.
5. Download the HTML answer report. It includes the question, answer, retrieved source excerpts and tool activity.

To use your own data, open Document Library, select one or more files, inspect the preview, and click Index selected documents. The raw upload copies are deleted after ingestion; indexed text chunks and embeddings remain in the local vector-store folder until you clear the index.

## Supported inputs

| Format | Supported behavior | Notes |
|---|---|---|
| PDF | Extract selectable text | Scanned PDFs need an OCR workflow; embedded-page OCR is not automatic. |
| DOCX | Extract paragraph text | Tables may not be included by the current extractor. |
| TXT / MD / CSV / JSON / LOG | Read UTF-8 text and index it | Large batches are bounded by the upload limits. |
| PNG / JPG / JPEG / WEBP / BMP | Index metadata by default; optional local OCR | Image understanding is not implied. Install Tesseract and the optional OCR extra to extract visible text. |
| ZIP | Download the included sample pack | Upload extracted documents rather than the ZIP itself. |

Upload limits are 25 MB per file and 100 MB per batch. The app validates extensions, reports processing errors by file and uses a temporary session directory for raw uploaded bytes.

## Navigation

- **Overview** — product purpose, who it is for, four-step workflow, local index KPIs and sample shortcut.
- **Chat & Evidence** — conversational questions, clickable source excerpts, a compact answer summary and report downloads.
- **Document Library** — multiple uploads, preview, validation, per-file status, indexed source summaries and clear-index action.
- **Tool Trace** — recent tool activity and a manual local utility runner.
- **Audit Log** — local ingestion, prompt, answer and tool events. Access follows the configured agent role.
- **Settings & Glossary** — model status, optional feature availability and plain-English definitions.

## Model options

### Demo

The Demo backend produces template-based answers and is useful for checking the end-to-end interface. With indexed material, it can include passages retrieved from the local vector store. It is not a full reasoning model.

### Ollama (local model)

1. Install Ollama for your operating system.
2. Pull the configured model, for example: **ollama pull llama3.2:1b**.
3. Choose **Ollama · local model** in the sidebar.

The workbench sends prompts to the configured local Ollama endpoint, not a hosted OpenAI-compatible API through this selector. The first embedding run may download sentence-transformer weights into the local Hugging Face cache. For a disconnected environment, pre-cache the weights first.

## Optional features

Install from the repository root:

- PDF answer export: **python -m pip install -e ".[reports]"**
- Semantic embeddings (optional): **python -m pip install torch --index-url https://download.pytorch.org/whl/cpu**, then **python -m pip install -e ".[embeddings]"**. Cache the configured model while online before switching to a disconnected environment.
- OCR: **python -m pip install -e ".[ocr]"**, then install the Tesseract executable separately and ensure it is on PATH.
- Screenshot capture: **python -m pip install -e ".[screenshots]"**, then **python -m playwright install chromium**.

The Settings & Glossary page lists whether each feature is active and what is missing. Image uploads without OCR are indexed as image metadata only.

## Capture screenshots of every page

Run the following commands on a machine with Chromium support:

    python -m pip install -e ".[screenshots]"
    python -m playwright install chromium
    python scripts/capture_screenshots.py

The script starts the app on port 8510 and saves actual Playwright screenshots for each navigation page to docs/screenshots/. Screenshots should be regenerated after visual changes; they are machine-generated artifacts, not hand-drawn mockups.

## Architecture

~~~mermaid
flowchart LR
  UI[Streamlit workspace] --> Agent[RAG agent]
  Agent --> Retrieval[Local vector store]
  Agent --> LLM[Demo or local Ollama]
  Agent --> Tools[Local calculator and document utilities]
  Agent --> Audit[Local JSONL audit log]
  Retrieval --> Chunks[Document passages]
  Upload[Temporary uploads] --> Processor[PDF DOCX text and optional OCR]
  Processor --> Retrieval
~~~

Core agent, document processing, retrieval and audit logic stay outside the Streamlit UI. The vector store uses FAISS when available and a NumPy similarity fallback otherwise. The default embedding backend is dependency-light deterministic text hashing; semantic transformer embeddings are optional.

## Run tests

    python -m pip install -r requirements-dev.txt
    ruff check src tests app.py run.py
    bandit -q -r src -ll
    pip-audit -r requirements.txt --progress-spinner off
    pytest -v

## Privacy, roles and limitations

- Raw upload copies are placed in a temporary folder and removed after ingestion. Extracted chunks and numeric embeddings remain locally persisted until the index is cleared.
- The audit log can include filenames, questions, answers and tool outputs. Protect the data directory with operating-system permissions.
- The DEFAULT_ROLE setting controls the local agent role. Audit access requires admin; for example, set DEFAULT_ROLE=admin in a local .env file and restart.
- This app does not provide an identity provider or multi-user authentication. A role setting is not a substitute for authenticated access control.
- Demo responses are template-based, not a substitute for a real language model or source verification.
- OCR is optional. Without Tesseract and pytesseract, images are stored as metadata-only passages.
- The default text-hashing retrieval mode has no model downloads. Optional semantic embeddings use local transformer weights; install a CPU-only PyTorch wheel and pre-cache the model before using a disconnected machine.
- Safety-critical guidance must be checked against the original procedures and approved by the responsible person.

## Project layout

    src/
      agent/       RAG agent, tools, RBAC and audit
      document/    text extraction, chunking, temporary uploads and reports
      retrieval/   local vector store
      llm/         Demo and local model interfaces
      ui/          Streamlit pages, style sheet, theme and components
      api/         FastAPI endpoints
    tests/         core, ingestion, API and page smoke tests
    docs/          product mark and screenshot output directory

## License

MIT. See LICENSE.
