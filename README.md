# Sovereign Agentic AI Workbench

On-premise, privacy-first agentic AI workbench using open-weight multimodal-capable models for confidential industrial workflows.

## Problem Statement

Industrial environments handle highly confidential process data, safety protocols, and equipment manuals that cannot leave the local network. Existing cloud LLM solutions introduce data-exfiltration risk. This project provides a fully local RAG + agent workbench that runs on ordinary hardware and never sends documents or queries off-machine by default.

## Overview

The workbench lets operators:

- Upload industrial documents (PDF, DOCX, TXT, images)
- Build a local vector knowledge base
- Query the knowledge base via a chat UI or REST API
- Receive answers with source citations
- Keep full conversation history on disk
- Switch between a zero-dependency demo backend and a real Ollama open-weight model

## Features

- **Local document ingestion** – PDF, DOCX, plain text, basic image registration
- **Chunking + embedding** – sentence-transformers (CPU) + FAISS / numpy fallback
- **RAG agent** – retrieval + generation with source references
- **Conversation history** – in-memory + optional persistence
- **Model abstraction** – Demo / Ollama / OpenAI-compatible backends
- **FastAPI backend** + **Streamlit chat UI**
- **Demo mode** – works with zero external services and no GPU
- **Cross-platform** – Windows, Linux, macOS via `pathlib` and env vars

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Streamlit  │────▶│  RAG Agent   │────▶│  LLM Layer  │
│     UI      │     │              │     │ Demo/Ollama │
└─────────────┘     └──────┬───────┘     └─────────────┘
                           │
┌─────────────┐     ┌──────▼───────┐     ┌─────────────┐
│   FastAPI   │────▶│  Retrieval   │────▶│  FAISS /    │
│             │     │  (Vector KB) │     │  numpy      │
└─────────────┘     └──────┬───────┘     └─────────────┘
                           │
                    ┌──────▼───────┐
                    │  Document    │
                    │  Processing  │
                    └──────────────┘
```

## Tech Stack

- Python 3.11+
- FastAPI + Uvicorn
- Streamlit
- sentence-transformers + faiss-cpu
- pypdf, python-docx, Pillow
- pydantic-settings
- pytest

## Repository Structure

```
sovereign-agentic-ai-workbench/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── .env.example
├── src/
│   ├── config/          # Settings
│   ├── document/        # Extraction & chunking
│   ├── retrieval/       # Vector store
│   ├── llm/             # Model backends
│   ├── agent/           # RAG agent
│   ├── api/             # FastAPI app
│   └── ui/              # Streamlit app
├── tests/
├── data/
│   └── sample/
├── scripts/
│   └── generate_demo_data.py
├── models/
└── docs/
```

## System Requirements

| Mode            | CPU          | RAM   | Disk  | GPU        |
|-----------------|--------------|-------|-------|------------|
| Demo (default)  | Any modern   | 4 GB  | 2 GB  | Not needed |
| Ollama 1B–3B    | 4+ cores     | 8 GB  | 5 GB  | Optional   |
| Larger models   | 8+ cores     | 16 GB+| 10 GB+| Recommended|

First run of sentence-transformers downloads ~90 MB model weights (cached afterwards).

## Installation

### Windows (PowerShell)

```powershell
git clone https://github.com/k-vandith/sovereign-agentic-ai-workbench.git
cd sovereign-agentic-ai-workbench
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

### Linux / macOS

```bash
git clone https://github.com/k-vandith/sovereign-agentic-ai-workbench.git
cd sovereign-agentic-ai-workbench
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Environment Variables

See `.env.example`. Important keys:

| Variable        | Default              | Description                          |
|-----------------|----------------------|--------------------------------------|
| `LLM_BACKEND`   | `demo`               | `demo` \| `ollama` \| `openai_compatible` |
| `OLLAMA_MODEL`  | `llama3.2:1b`        | Model tag when using Ollama          |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embedding model                |

## Dataset / Demo Mode

```bash
python scripts/generate_demo_data.py
```

This writes three synthetic industrial documents into `data/sample/`. Upload them via the UI or API.

## Running the Application

### Streamlit UI (recommended for demos)

```bash
streamlit run src/ui/app.py
```

Open http://localhost:8501

### FastAPI backend

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs: http://localhost:8000/docs

### Optional: Ollama

```bash
# Install Ollama from https://ollama.com then:
ollama pull llama3.2:1b
# Set LLM_BACKEND=ollama in .env and restart
```

## API Usage

```bash
# Health
curl http://localhost:8000/health

# Ingest
curl -X POST -F "file=@data/sample/safety_protocol.txt" http://localhost:8000/ingest

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the max temperature for Reactor R-12?"}'
```

## Testing

```bash
pytest -v
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: src` | Run from project root; `pythonpath = ["."]` is set in pyproject.toml |
| Embedding model download slow | First run caches under `~/.cache/huggingface` |
| Ollama connection refused | Start `ollama serve` and confirm `OLLAMA_BASE_URL` |
| FAISS install fails on some platforms | Code falls back to pure-numpy cosine search automatically |

## Limitations

- Demo backend produces template answers; it is not a generative model.
- Image support is registration-only (no OCR unless Tesseract is added separately).
- Multimodal generation requires an Ollama vision model (e.g. `llava`) – not enabled by default.
- Conversation history is in-memory by default (restarts clear it).
- Not intended for production multi-user deployments without additional auth.

## Security / Privacy

- No data leaves the machine unless you deliberately point at a remote OpenAI-compatible endpoint.
- Uploaded files stay under `data/uploads/`.
- `.env` and secrets are git-ignored.
- Designed for air-gapped / confidential industrial networks.

## Future Improvements

- Persistent conversation store (SQLite)
- Tool-calling agents (file search, calculator, internal APIs)
- Optional Tesseract OCR path
- Role-based access control
- Streaming token responses

## License

MIT
