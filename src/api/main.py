"""FastAPI backend for the Sovereign Agentic AI Workbench."""
from __future__ import annotations

import logging
import shutil

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import get_settings
from src.agent import RAGAgent
from src.paths import safe_filename

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()
agent = RAGAgent()

app = FastAPI(
    title="Sovereign Agentic AI Workbench",
    description="On-premise privacy-first RAG workbench with open-weight models",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    conversation_id: str | None = None
    top_k: int | None = None


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
async def ingest(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "No filename provided")
    try:
        dest = settings.upload_dir / safe_filename(file.filename)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    try:
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        result = agent.ingest_file(dest)
        return result
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.exception("Ingest failed")
        raise HTTPException(500, f"Ingest failed: {exc}") from exc


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    try:
        return agent.query(
            question=req.question,
            conversation_id=req.conversation_id,
            top_k=req.top_k,
        )
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(500, str(exc)) from exc


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
    agent.clear_knowledge_base()
    return {"status": "cleared", "chunks": 0}


@app.get("/stats")
def stats():
    return {
        "chunks": agent.store.count(),
        "conversations": len(agent.conversations),
        "upload_dir": str(settings.upload_dir),
    }
