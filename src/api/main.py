"""FastAPI backend for the Sovereign Agentic AI Workbench."""
from __future__ import annotations

import hmac
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from src.config import get_settings
from src.agent import RAGAgent
from src.paths import safe_filename
from src.document.uploads import MAX_FILE_BYTES, SUPPORTED_UPLOAD_EXTENSIONS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()
agent = RAGAgent()

app = FastAPI(
    title="Sovereign Agentic AI Workbench",
    description="Local-document RAG workbench with preview, local Ollama, or OpenAI-compatible generation backends",
    version="1.0.0",
)
ALLOWED_BROWSER_ORIGINS = {
    f"http://127.0.0.1:{settings.streamlit_port}",
    f"http://localhost:{settings.streamlit_port}",
}
app.add_middleware(
    CORSMiddleware,
    allow_origins=sorted(ALLOWED_BROWSER_ORIGINS),
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)


TOKEN_PROTECTED_ROUTES = {
    ("/ingest", "POST"),
    ("/query", "POST"),
    ("/knowledge-base", "DELETE"),
    ("/conversations", "GET"),
    ("/stats", "GET"),
}


@app.middleware("http")
async def protect_state_changing_routes(request: Request, call_next):
    """Require a bearer token for private API routes and reject hostile browser writes."""
    method = request.method.upper()
    route_key = (request.url.path, method)
    private_conversation_read = method == "GET" and request.url.path.startswith("/conversations/")
    if route_key not in TOKEN_PROTECTED_ROUTES and not private_conversation_read:
        return await call_next(request)

    origin = request.headers.get("origin")
    # Check Origin independently of CORS: simple form posts do not preflight.
    if origin and method in {"POST", "PUT", "PATCH", "DELETE"} and origin not in ALLOWED_BROWSER_ORIGINS:
        return JSONResponse(
            status_code=403,
            content={"detail": "Cross-origin state-changing requests are not allowed."},
        )

    expected_token = str(settings.api_token or "").strip()
    if not expected_token:
        return JSONResponse(
            status_code=503,
            content={"detail": "API token authentication is not configured. Set API_TOKEN before enabling write/query routes."},
        )

    authorization = request.headers.get("authorization", "")
    scheme, separator, supplied_token = authorization.partition(" ")
    if (
        not separator
        or scheme.lower() != "bearer"
        or not supplied_token
        or not hmac.compare_digest(supplied_token, expected_token)
    ):
        return JSONResponse(
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
            content={"detail": "A valid bearer token is required."},
        )

    return await call_next(request)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=10000)
    conversation_id: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=50, strict=True)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question must not be blank.")
        return value


class QueryResponse(BaseModel):
    conversation_id: str
    answer: str
    sources: list[dict]
    backend: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "llm_backend": type(agent.llm).__name__,
        "llm_available": agent.llm.is_available(),
        "knowledge_base_chunks": agent.store.count(),
    }


@app.post("/ingest")
def ingest(file: UploadFile = File(...)):
    """Ingest one bounded upload from a temporary file, always cleaning it up."""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided.")

        name = safe_filename(file.filename)
        if Path(name).suffix.lower() not in SUPPORTED_UPLOAD_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Unsupported file type.")

        size = 0
        with TemporaryDirectory(prefix="sovereign-workbench-api-upload-") as folder:
            dest = Path(folder) / name
            with dest.open("wb") as output:
                while True:
                    chunk = file.file.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MAX_FILE_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=f"Uploaded file exceeds the {MAX_FILE_BYTES // (1024 * 1024)} MB per-file limit.",
                        )
                    output.write(chunk)

            if size == 0:
                raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            outcome = agent.ingest_file(dest)
            if not int(outcome.get("chunks_added", 0)):
                return {
                    **outcome,
                    "status": "no_searchable_text",
                    "message": "No searchable text was extracted; scanned PDFs are not OCR-processed in this version.",
                }
            return {**outcome, "status": "indexed"}
    except HTTPException:
        raise
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail="The configured role cannot ingest documents.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Ingest failed")
        raise HTTPException(
            status_code=500,
            detail="Document ingestion failed. Check the application logs.",
        ) from exc
    finally:
        file.file.close()


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    try:
        return agent.query(
            question=req.question,
            conversation_id=req.conversation_id,
            top_k=req.top_k,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail="The configured role cannot run this query.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(
            status_code=500,
            detail="Query failed. Check the backend configuration and application logs.",
        ) from exc


@app.get("/conversations")
def list_conversations():
    return agent.list_conversations()


@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str):
    conv = agent.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(404, "Conversation not found")
    return {
        "id": conv.id,
        "created_at": conv.created_at,
        "messages": [
            {
                "role": m.role,
                "content": m.content,
                "sources": m.sources,
                "timestamp": m.timestamp,
            }
            for m in conv.messages
        ],
    }


@app.delete("/knowledge-base")
def clear_kb():
    try:
        agent.clear_knowledge_base()
    except PermissionError as exc:
        raise HTTPException(
            status_code=403,
            detail="The configured role cannot clear the knowledge base.",
        ) from exc
    return {"status": "cleared", "chunks": 0}


@app.get("/stats")
def stats():
    return {
        "chunks": agent.store.count(),
        "conversations": len(agent.conversations),
        "upload_dir": str(settings.upload_dir),
    }
