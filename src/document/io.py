"""Portable reports and sample-pack utilities for the local workbench."""
from __future__ import annotations

from html import escape
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from typing import Any


def build_answer_report(
    question: str,
    answer: str,
    sources: list[dict[str, Any]],
    tool_trace: list[dict[str, Any]],
    backend: str,
) -> str:
    """Build a self-contained HTML report without external assets."""
    source_rows = "".join(
        "<tr><td>" + escape(str(item.get("source", "source"))) + "</td><td>"
        + escape(str(item.get("score", "—"))) + "</td><td>"
        + escape(str(item.get("snippet", ""))) + "</td></tr>"
        for item in sources
    )
    trace_rows = "".join(
        "<tr><td>" + escape(str(item.get("tool", "tool"))) + "</td><td>"
        + escape("OK" if item.get("ok") else "Needs review") + "</td><td>"
        + escape(str(item.get("input", ""))) + "</td><td>"
        + escape(str(item.get("output", ""))) + "</td></tr>"
        for item in tool_trace
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Sovereign Workbench answer report</title>
<style>
body{{font:16px/1.6 Inter,Segoe UI,Arial,sans-serif;color:#182a37;background:#f1f6f8;margin:0}}
main{{max-width:1000px;margin:0 auto;padding:30px}}
header{{padding:28px;border-radius:18px;background:#10222e;color:#f0fbff}}
header small{{color:#9beaf4;letter-spacing:.12em;text-transform:uppercase}}
section{{margin:16px 0;padding:20px;background:#fff;border:1px solid #d8e3e9;border-radius:14px}}
h1{{margin:.4rem 0}}h2{{font-size:1.2rem}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit}}
table{{width:100%;border-collapse:collapse;overflow-wrap:anywhere}}th,td{{text-align:left;vertical-align:top;padding:10px;border-bottom:1px solid #e1e8ed}}
th{{background:#eaf4f7}}footer{{color:#5c7180;font-size:.82rem;margin-top:20px}}
</style></head><body><main>
<header><small>Private knowledge · Local AI</small><h1>Workbench answer report</h1>
<p>Backend: {escape(str(backend))}</p></header>
<section><h2>Question</h2><pre>{escape(str(question))}</pre></section>
<section><h2>Answer summary</h2><pre>{escape(str(answer))}</pre>
<p>Interpret answers in context. Verify safety-critical details against the original procedure or manual.</p></section>
<section><h2>Source evidence ({len(sources)})</h2>
<table><thead><tr><th>Document</th><th>Match score</th><th>Source excerpt</th></tr></thead>
<tbody>{source_rows or '<tr><td colspan="3">No source excerpts were returned for this question.</td></tr>'}</tbody></table></section>
<section><h2>Tool activity ({len(tool_trace)})</h2>
<table><thead><tr><th>Tool</th><th>Status</th><th>Input</th><th>Output</th></tr></thead>
<tbody>{trace_rows or '<tr><td colspan="4">No tools ran for this question.</td></tr>'}</tbody></table></section>
<footer>Generated locally by Sovereign Workbench. This report is not a substitute for reviewing the source document.</footer>
</main></body></html>"""


def build_answer_pdf(
    question: str,
    answer: str,
    sources: list[dict[str, Any]],
    tool_trace: list[dict[str, Any]],
    backend: str,
) -> bytes | None:
    """Return a local PDF report when ReportLab is installed."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
        from xml.sax.saxutils import escape as xml_escape

        buffer = BytesIO()
        document = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=16 * mm, leftMargin=16 * mm, topMargin=16 * mm, bottomMargin=16 * mm)
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name="WBBody", parent=styles["BodyText"], leading=14, spaceAfter=7))
        story = [
            Paragraph("Sovereign Workbench — Answer Report", styles["Title"]),
            Paragraph(f"Backend: {xml_escape(str(backend))}", styles["BodyText"]),
            Spacer(1, 10),
            Paragraph("Question", styles["Heading2"]),
            Paragraph(xml_escape(str(question)).replace("\\n", "<br/>"), styles["WBBody"]),
            Paragraph("Answer", styles["Heading2"]),
            Paragraph(xml_escape(str(answer)).replace("\\n", "<br/>"), styles["WBBody"]),
            Paragraph("Source evidence", styles["Heading2"]),
        ]
        source_rows = [[Paragraph("Document", styles["BodyText"]), Paragraph("Score", styles["BodyText"]), Paragraph("Excerpt", styles["BodyText"])]]
        for item in sources:
            source_rows.append([
                Paragraph(xml_escape(str(item.get("source", "source"))), styles["BodyText"]),
                Paragraph(xml_escape(str(item.get("score", "—"))), styles["BodyText"]),
                Paragraph(xml_escape(str(item.get("snippet", ""))[:700]), styles["BodyText"]),
            ])
        source_table = Table(source_rows, colWidths=[42 * mm, 20 * mm, 110 * mm], repeatRows=1)
        source_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d7f0f2")),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c7d6df")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.extend([source_table, Spacer(1, 10), Paragraph("Tool activity", styles["Heading2"])])
        if tool_trace:
            for step in tool_trace:
                story.append(Paragraph(
                    f"<b>{xml_escape(str(step.get('tool', 'tool')))}</b> — "
                    f"{'OK' if step.get('ok') else 'Needs review'}<br/>"
                    f"Input: {xml_escape(str(step.get('input', ''))[:300])}<br/>"
                    f"Output: {xml_escape(str(step.get('output', ''))[:800])}",
                    styles["WBBody"],
                ))
        else:
            story.append(Paragraph("No tools ran for this question.", styles["WBBody"]))
        story.append(Spacer(1, 10))
        story.append(Paragraph(
            "Generated locally. Verify safety-critical details against the original procedure or manual.",
            styles["Italic"],
        ))
        document.build(story)
        return buffer.getvalue()
    except Exception:
        return None


def build_sample_archive(sample_dir: Path) -> bytes:
    """Return a ZIP containing supported demo files, with no filesystem output."""
    buffer = BytesIO()
    files = []
    if sample_dir.is_dir():
        files = sorted(
            path for path in sample_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".pdf", ".docx", ".txt", ".md", ".csv", ".json", ".log"}
        )
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        if files:
            for path in files:
                archive.write(path, arcname=path.relative_to(sample_dir).as_posix())
        else:
            archive.writestr(
                "sample_process_note.txt",
                "Fictional sample note\\nThe pump unit should be inspected every 2,000 operating hours.\\n",
            )
    return buffer.getvalue()
