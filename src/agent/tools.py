"""Built-in tools for the agent loop (CPU-only, offline-capable)."""
from __future__ import annotations

import ast
import logging
import math
import operator
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval_expr(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval_expr(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_eval_expr(node.left), _safe_eval_expr(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval_expr(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        if name in {"sqrt", "log", "sin", "cos", "abs", "round"} and len(node.args) == 1:
            fn = {"sqrt": math.sqrt, "log": math.log, "sin": math.sin, "cos": math.cos, "abs": abs, "round": round}[name]
            return float(fn(_safe_eval_expr(node.args[0])))
    raise ValueError("Only basic arithmetic expressions are allowed")


@dataclass
class ToolResult:
    name: str
    input: str
    output: str
    ok: bool = True


class ToolRegistry:
    def __init__(self, vector_store: Any | None = None) -> None:
        self._store = vector_store
        self._tools: dict[str, Callable[[str], str]] = {
            "calculator": self.calculator,
            "document_search": self.document_search,
            "summarizer": self.summarizer,
            "image_analysis": self.image_analysis,
        }

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def run(self, name: str, arg: str) -> ToolResult:
        fn = self._tools.get(name)
        if not fn:
            return ToolResult(name=name, input=arg, output=f"Unknown tool: {name}", ok=False)
        try:
            return ToolResult(name=name, input=arg, output=fn(arg), ok=True)
        except Exception as exc:
            logger.exception("Tool %s failed", name)
            return ToolResult(name=name, input=arg, output=str(exc), ok=False)

    def calculator(self, expression: str) -> str:
        tree = ast.parse(expression.strip(), mode="eval")
        value = _safe_eval_expr(tree)
        return f"{expression.strip()} = {value}"

    def document_search(self, query: str) -> str:
        if self._store is None or self._store.count() == 0:
            return "Knowledge base is empty. Ingest documents first."
        hits = self._store.search(query, top_k=3)
        if not hits:
            return "No matching documents found."
        parts = [f"[{c.source} score={s:.3f}] {c.text[:300]}" for c, s in hits]
        return "\n---\n".join(parts)

    def summarizer(self, text: str) -> str:
        text = text.strip()
        if not text:
            return "Nothing to summarise."
        sentences = re.split(r"(?<=[.!?])\s+", text)
        head = sentences[0] if sentences else text[:200]
        keywords = re.findall(r"\b[A-Z][A-Za-z0-9\-]{2,}\b|\b\d+(?:\.\d+)?\s*(?:°C|C|psi|bar|%)?\b", text)
        uniq = list(dict.fromkeys(keywords))
        key_line = ", ".join(uniq[:12]) if uniq else "(no key terms)"
        return f"Summary: {head}\nKey terms: {key_line}\nLength: {len(text)} chars, ~{len(sentences)} sentences."

    def image_analysis(self, path_or_desc: str) -> str:
        path = Path(path_or_desc.strip().strip("'\""))
        if not path.exists() or not path.is_file():
            return f"Image path not found: {path_or_desc}. Provide a local image path."
        try:
            from PIL import Image
            with Image.open(path) as img:
                w, h = img.size
                mode, fmt = img.mode, (img.format or path.suffix)
            return (
                f"Image file: {path.name}\nFormat: {fmt}, mode: {mode}, size: {w}x{h} px\n"
                "Content: local metadata analysis only. Semantic image understanding is not enabled in this build."
            )
        except Exception as exc:
            return f"Could not analyse image: {exc}"


TOOL_CALL_RE = re.compile(
    r"TOOL:\s*(?P<name>calculator|document_search|summarizer|image_analysis)\s*\|\s*(?P<arg>.+?)(?:\n|$)",
    re.IGNORECASE | re.DOTALL,
)


def parse_tool_calls(text: str) -> list[tuple[str, str]]:
    return [(m.group("name").lower(), m.group("arg").strip()) for m in TOOL_CALL_RE.finditer(text)]
