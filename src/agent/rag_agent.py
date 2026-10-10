"""Agent layer: RAG + tool loop + audit + RBAC."""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.agent.audit import AuditLog
from src.agent.rbac import Principal, Role, require
from src.agent.tools import ToolRegistry, parse_tool_calls
from src.config import get_settings
from src.document import extract_text_from_file, chunk_text
from src.llm import get_llm, BaseLLM
from src.retrieval import DocumentChunk, LocalVectorStore

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful industrial AI assistant running fully on-premise.
Answer strictly based on the provided context when available.
Cite sources by filename when you use retrieved information.
If the context is insufficient, say so clearly.
Do not invent confidential industrial data.

You may request tools by writing a line exactly like:
TOOL: calculator | 2 + 2 * 3
TOOL: document_search | reactor temperature limit
TOOL: summarizer | <text to summarise>
TOOL: image_analysis | path/to/image.png
After tool results are provided, give a final answer without further TOOL lines unless needed."""


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
    """Retrieval-Augmented Generation agent with tools, audit, and RBAC."""

    def __init__(
        self,
        vector_store: LocalVectorStore | None = None,
        llm: BaseLLM | None = None,
        principal: Principal | None = None,
        audit: AuditLog | None = None,
        max_tool_rounds: int = 3,
    ) -> None:
        settings = get_settings()
        self.settings = settings
        self.store = vector_store or LocalVectorStore(
            persist_dir=settings.vectorstore_dir,
            embedding_model_name=settings.embedding_model,
        )
        self.llm = llm or get_llm()
        self.tools = ToolRegistry(vector_store=self.store)
        try:
            configured_role = Role(str(settings.default_role).strip().lower())
        except ValueError:
            configured_role = Role.OPERATOR
        self.principal = principal or Principal(name="local-user", role=configured_role)
        self.audit = audit or AuditLog(path=settings.data_dir / "audit" / "audit.jsonl")
        self.max_tool_rounds = max_tool_rounds
        self.conversations: dict[str, Conversation] = {}

    def ingest_file(self, path: Path) -> dict[str, Any]:
        require(self.principal, "ingest")
        text = extract_text_from_file(path)
        chunks_raw = chunk_text(
            text, chunk_size=self.settings.chunk_size, overlap=self.settings.chunk_overlap
        )
        doc_chunks = [
            DocumentChunk(
                id=f"{path.stem}_{i}",
                text=c,
                source=str(path.name),
                metadata={"path": path.name, "chunk_index": i},
            )
            for i, c in enumerate(chunks_raw)
        ]
        n = self.store.add_documents(doc_chunks)
        self.audit.record("ingest", user=self.principal.name, file=path.name, chunks=n)
        return {"file": path.name, "chunks_added": n, "total_chunks": self.store.count()}

    def ingest_directory(self, directory: Path) -> dict[str, Any]:
        from src.document import iter_supported_files
        results = []
        for f in iter_supported_files(directory):
            try:
                results.append(self.ingest_file(f))
            except Exception as exc:
                results.append({"file": f.name, "error": str(exc)})
        return {"files_processed": len(results), "details": results}

    def query(
        self,
        question: str,
        conversation_id: str | None = None,
        top_k: int | None = None,
        use_tools: bool = True,
    ) -> dict[str, Any]:
        require(self.principal, "query")
        top_k = top_k or self.settings.top_k
        hits = self.store.search(question, top_k=top_k)
        context_parts, sources = [], []
        for chunk, score in hits:
            context_parts.append(f"[Source: {chunk.source}]\n{chunk.text}")
            sources.append({"source": chunk.source, "score": round(score, 4), "snippet": chunk.text[:200]})
        context = "\n\n".join(context_parts) if context_parts else "No relevant documents found."
        tool_trace: list[dict[str, Any]] = []
        prompt = (
            f"Context:\n{context}\n\nQuestion: {question}\n\n"
            "Answer based on the context above. Cite sources when used. Use TOOL: lines if a tool would help."
        )
        self.audit.record(
            "prompt", user=self.principal.name, role=self.principal.role.value,
            question=question[:2000], offline=self.settings.offline_mode,
        )
        answer = self.llm.generate(prompt, system=SYSTEM_PROMPT)
        if use_tools and self.principal.can("use_tools"):
            for _ in range(self.max_tool_rounds):
                calls = parse_tool_calls(answer)
                if not calls:
                    break
                tool_outputs = []
                for name, arg in calls:
                    result = self.tools.run(name, arg)
                    tool_trace.append({"tool": result.name, "input": result.input, "output": result.output, "ok": result.ok})
                    self.audit.record(
                        "tool_call", user=self.principal.name, tool=result.name,
                        input=result.input[:500], output=result.output[:1000], ok=result.ok,
                    )
                    tool_outputs.append(f"[{result.name}] {result.output}")
                follow = (
                    f"Context:\n{context}\n\nQuestion: {question}\n\n"
                    f"Tool results:\n" + "\n".join(tool_outputs) + "\n\n"
                    "Provide the final answer for the user. Do not emit TOOL lines unless still needed."
                )
                answer = self.llm.generate(follow, system=SYSTEM_PROMPT)
        self.audit.record(
            "answer", user=self.principal.name, question=question[:500],
            answer=answer[:2000], tools_used=len(tool_trace),
        )
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
            "tool_trace": tool_trace,
            "backend": type(self.llm).__name__,
            "user": self.principal.name,
            "role": self.principal.role.value,
        }

    def list_conversations(self) -> list[dict[str, Any]]:
        return [{"id": c.id, "created_at": c.created_at, "message_count": len(c.messages)} for c in self.conversations.values()]

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        return self.conversations.get(conversation_id)

    def clear_knowledge_base(self) -> None:
        """Drop indexed chunks. Requires ingest permission."""
        require(self.principal, "ingest")
        self.store.clear()
        self.audit.record("clear_kb", user=self.principal.name)

    def view_audit(self, limit: int = 50) -> list[dict[str, Any]]:
        require(self.principal, "view_audit")
        return self.audit.recent(limit=limit)
