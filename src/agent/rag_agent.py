"""Agent layer that orchestrates retrieval + generation."""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import get_settings
from src.document import extract_text_from_file, chunk_text
from src.llm import get_llm, BaseLLM
from src.retrieval import DocumentChunk, LocalVectorStore

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful industrial AI assistant running fully on-premise.
Answer strictly based on the provided context when available.
Cite sources by filename when you use retrieved information.
If the context is insufficient, say so clearly.
Do not invent confidential industrial data."""


@dataclass
class Message:
    role: str
    content: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class Conversation:
    id: str
    messages: list[Message] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class RAGAgent:
    """Retrieval-Augmented Generation agent with conversation history."""

    def __init__(
        self,
        vector_store: LocalVectorStore | None = None,
        llm: BaseLLM | None = None,
    ) -> None:
        settings = get_settings()
        self.settings = settings
        self.store = vector_store or LocalVectorStore(
            persist_dir=settings.vectorstore_dir,
            embedding_model_name=settings.embedding_model,
        )
        self.llm = llm or get_llm()
        self.conversations: dict[str, Conversation] = {}

    def ingest_file(self, path: Path) -> dict[str, Any]:
        """Ingest a single file into the knowledge base."""
        text = extract_text_from_file(path)
        chunks_raw = chunk_text(
            text,
            chunk_size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
        )
        doc_chunks = [
            DocumentChunk(
                id=f"{path.stem}_{i}",
                text=c,
                source=str(path.name),
                metadata={"path": str(path), "chunk_index": i},
            )
            for i, c in enumerate(chunks_raw)
        ]
        n = self.store.add_documents(doc_chunks)
        logger.info("Ingested %d chunks from %s", n, path.name)
        return {
            "file": path.name,
            "chunks_added": n,
            "total_chunks": self.store.count(),
        }

    def ingest_directory(self, directory: Path) -> dict[str, Any]:
        from src.document import iter_supported_files

        results = []
        for f in iter_supported_files(directory):
            try:
                results.append(self.ingest_file(f))
            except Exception as exc:
                logger.warning("Skipping %s: %s", f, exc)
                results.append({"file": f.name, "error": str(exc)})
        return {"files_processed": len(results), "details": results}

    def query(
        self,
        question: str,
        conversation_id: str | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Run a RAG query and optionally append to a conversation."""
        top_k = top_k or self.settings.top_k
        hits = self.store.search(question, top_k=top_k)

        context_parts = []
        sources = []
        for chunk, score in hits:
            context_parts.append(f"[Source: {chunk.source}]\n{chunk.text}")
            sources.append(
                {
                    "source": chunk.source,
                    "score": round(score, 4),
                    "snippet": chunk.text[:200],
                }
            )
        context = "\n\n".join(context_parts) if context_parts else "No relevant documents found."

        prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer based on the context above. Cite sources when used."
        )

        answer = self.llm.generate(prompt, system=SYSTEM_PROMPT)

        if conversation_id is None:
            conversation_id = str(uuid.uuid4())
        conv = self.conversations.get(conversation_id) or Conversation(id=conversation_id)
        conv.messages.append(Message(role="user", content=question))
        conv.messages.append(Message(role="assistant", content=answer, sources=sources))
        self.conversations[conversation_id] = conv

        return {
            "conversation_id": conversation_id,
            "answer": answer,
            "sources": sources,
            "backend": type(self.llm).__name__,
        }

    def list_conversations(self) -> list[dict[str, Any]]:
        return [
            {
                "id": c.id,
                "created_at": c.created_at,
                "message_count": len(c.messages),
            }
            for c in self.conversations.values()
        ]

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        return self.conversations.get(conversation_id)

    def clear_knowledge_base(self) -> None:
        self.store.clear()
