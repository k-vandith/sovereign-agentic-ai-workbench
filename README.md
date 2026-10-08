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
│   ├── config/
│   ├── document/
│   ├── retrieval/
│   ├── llm/
│   ├── agent/
│   ├── api/
│   └── ui/
├── tests/
├── data/sample/
├── scripts/
│   ├── setup_env.py
│   ├── setup.sh
│   ├── setup.ps1
│   └── generate_demo_data.py
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

### Recommended (all platforms) — automated bootstrap

Handles missing `ensurepip`, symlink restrictions, and installs dependencies into `.venv`:

```bash
git clone https://github.com/k-vandith/sovereign-agentic-ai-workbench.git
cd sovereign-agentic-ai-workbench
python3 scripts/setup_env.py    # or:  python scripts/setup_env.py
```

Then activate:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Copy environment file:

```bash
# Linux / macOS
cp .env.example .env

# Windows
copy .env.example .env
```

### Manual setup

#### Windows (PowerShell)

```powershell
git clone https://github.com/k-vandith/sovereign-agentic-ai-workbench.git
cd sovereign-agentic-ai-workbench
python -m venv .venv --copies
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```

#### Linux / macOS

```bash
git clone https://github.com/k-vandith/sovereign-agentic-ai-workbench.git
cd sovereign-agentic-ai-workbench
# If `python3 -m venv` fails with ensurepip errors:
#   sudo apt install python3-venv python3-pip   # Debian/Ubuntu
python3 -m venv .venv --copies
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

### Why `--copies`?

Some environments (restricted sandboxes, certain CI images) cannot create symlinks inside a venv (`Operation not permitted` on `lib64 → lib`). Using `--copies` avoids that. `scripts/setup_env.py` tries `--copies` first automatically.

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

## Running the Application

### Streamlit UI

```bash
streamlit run src/ui/app.py
```

### FastAPI backend

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
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
| `venv` / ensurepip fails | Run `python3 scripts/setup_env.py` or install `python3-venv` (Debian/Ubuntu) |
| `Operation not permitted` on lib64 | Use `python3 -m venv .venv --copies` (setup script does this) |
| `ModuleNotFoundError: pydantic_settings` | Activate `.venv` and re-run `pip install -r requirements.txt` |
| `ModuleNotFoundError: sentence_transformers` | Same — install from requirements inside the active venv |

## Limitations

- Demo backend produces template answers; it is not a generative model.
- Image support is registration-only (no OCR unless Tesseract is added separately).
- Multimodal generation requires an Ollama vision model (e.g. `llava`) – not enabled by default.
- Conversation history is in-memory by default (restarts clear it).

## Security / Privacy

- No data leaves the machine unless you deliberately point at a remote OpenAI-compatible endpoint.
- Uploaded files stay under `data/uploads/`.
- Designed for air-gapped / confidential industrial networks.

## License

MIT
