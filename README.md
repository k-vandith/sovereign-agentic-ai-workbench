<p align="center">
  <img src="docs/workbench-mark.svg" alt="Sovereign Workbench logo" width="88" />
</p>

# Sovereign Workbench

**A quieter workspace for answers grounded in your documents.**

Add manuals, notes and procedures. Ask a question in one place, inspect the source passages behind the answer, and export a short report. The UI is intentionally focused on three areas: **Workspace**, **Documents**, and **Settings**.

> For stronger answer generation, connect an API provider. The preview mode helps test uploading and source retrieval, but it is not a full reasoning model.

## Run locally

Python 3.11 or 3.12 is recommended.

### Windows PowerShell

```powershell
git clone https://github.com/k-vandith/sovereign-agentic-ai-workbench.git
cd sovereign-agentic-ai-workbench
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
python run.py
```

### Linux / macOS

```bash
git clone https://github.com/k-vandith/sovereign-agentic-ai-workbench.git
cd sovereign-agentic-ai-workbench
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
python run.py
```

Open http://127.0.0.1:8501.

## Get better answers with an API

1. Copy `.env.example` to `.env` in the repository root.
2. Set `LLM_BACKEND=openai_compatible` and `OFFLINE_MODE=false`.
3. Fill in the API endpoint, API key, and model name.
4. Restart the app.

Example settings:

```dotenv
LLM_BACKEND=openai_compatible
OFFLINE_MODE=false
OPENAI_COMPATIBLE_BASE_URL=https://api.openai.com/v1
OPENAI_COMPATIBLE_API_KEY=your_api_key_here
OPENAI_COMPATIBLE_MODEL=your_model_name
```

Use a provider that supports the OpenAI chat-completions format. Keep your key in the local `.env` file; do not commit that file or paste keys into source code.

Without API configuration, the app runs in preview mode so you can check imports, retrieval and source excerpts. Preview responses are illustrative and should not be treated as full model answers.

**Data flow:** files are indexed locally. When a real backend is enabled, the user's question and relevant retrieved passages are sent to the configured provider for answer generation. Review that provider's data policy before uploading sensitive documents.

### Use a local Ollama model

Start Ollama locally and make sure the configured model is available. Then set these values in `.env` and restart the workbench:

```dotenv
LLM_BACKEND=ollama
OFFLINE_MODE=true
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:1b
OLLAMA_TIMEOUT=120
```

With `OFFLINE_MODE=true`, Ollama endpoints must resolve to `localhost` or a loopback IP address. Non-local Ollama endpoints and hosted API calls are blocked in offline mode. The active backend is reflected in the UI and response metadata.

## API security and answer quality

Before starting the API with `python run.py --api`, create a local `.env` and set `API_TOKEN` to a unique random secret (generate one with `python -c "import secrets; print(secrets.token_urlsafe(32))"`). Send it as `Authorization: Bearer YOUR_TOKEN` for `POST /ingest`, `POST /query`, `DELETE /knowledge-base`, `GET /conversations`, `GET /conversations/{id}`, and `GET /stats`. These routes fail closed when `API_TOKEN` is missing. Requests with a browser `Origin` are accepted only from the configured local Streamlit origin; CORS alone is not relied on to block simple cross-origin form uploads. Keep the token private and do not commit `.env`.

Retrieval only sends passages scoring strictly above `RELEVANCE_THRESHOLD` (default `0.15`) to the model. If none clear the threshold, the workbench returns an explicit not-found answer without making a model call. `CONVERSATION_HISTORY_TOKENS` (default `2000`) limits prior dialogue included in follow-up prompts; it uses a local tokenizer when available with a conservative fallback.

## A simple workflow

1. Open **Workspace**.
2. Load the fictional sample pack or go to **Documents** to add your own files.
3. Ask a question in the workspace.
4. Open the source excerpts and verify key details against the original.
5. Export the answer as HTML or, when ReportLab is installed, PDF.

## Supported documents

PDF, DOCX, TXT, Markdown, CSV, JSON, LOG and common image formats (PNG, JPG, JPEG, WEBP, BMP). Text-based PDFs are extracted; scanned PDFs are not OCR-processed in this version. Image text extraction can use the optional OCR extra and a local Tesseract executable. Without OCR, images are indexed as metadata only; image upload alone does not imply visual understanding.

Upload limits are 25 MB per file and 100 MB per batch. Raw upload copies are deleted after extraction. Indexed passages, embeddings and audit records remain in the local data directory until cleared or deleted.

## Navigation

- **Workspace** — a short introduction and the main question-and-answer view.
- **Documents** — upload, preview, index, inspect and clear sources.
- **Settings** — configure API generation, understand data flow, and open advanced tool activity or the audit log when needed.

## Optional features

Install from the repository root:

- PDF answer export: `python -m pip install -e ".[reports]"`
- OCR: `python -m pip install -e ".[ocr]"`, then install the Tesseract executable separately.
- Browser screenshots: `python -m pip install -e ".[screenshots]"`, then `python -m playwright install chromium`.

The default retrieval index uses local CPU-friendly text hashing. Optional semantic embeddings may download transformer weights when enabled.

## Tests and checks

```powershell
python -m pip install -r requirements-dev.txt
ruff check src tests app.py run.py
bandit -q -r src -ll
pip-audit -r requirements.txt --progress-spinner off
pytest -v
```

## Security and limitations

- The local audit log may contain filenames, questions, answers and tool outputs. Protect it with operating-system permissions. New entries are SHA-256 hash-chained; use **Verify audit log integrity** in Settings to detect modifications. Legacy records created before hash-chaining can only be anchored as a prefix when the first chained event is appended, not retrospectively proven authentic.
- The application does not provide multi-user authentication or automatically encrypt the local data directory.
- Browser CORS is restricted to localhost/127.0.0.1 on the configured Streamlit port, but this is not a substitute for authenticated access control.
- A local role setting is not a substitute for authenticated access control.
- Retrieval and generated answers can be incomplete or incorrect. Confirm safety-critical guidance with the original document and responsible personnel.
- Do not commit sensitive source documents or API credentials.

## Project structure

```text
src/
  agent/       RAG, tools, permissions and audit
  api/         FastAPI endpoints
  config/      Environment settings
  document/    Extraction, uploads and reports
  llm/         API and preview generation
  retrieval/   Local vector store
  ui/          Streamlit workspace and theme
tests/         Retrieval, ingestion, API and UI smoke tests
scripts/       Local setup, sample-data and screenshot helpers
docs/          Product mark and optional screenshots
```

## License

MIT. See LICENSE.
