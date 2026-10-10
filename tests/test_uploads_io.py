from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile
from io import BytesIO

from src.document.io import build_answer_pdf, build_answer_report, build_sample_archive
from src.document.uploads import ingest_uploaded_files


class FakeUpload:
    def __init__(self, name: str, data: bytes):
        self.name = name
        self.data = data
        self.size = len(data)

    def getvalue(self) -> bytes:
        return self.data


class RecordingAgent:
    def __init__(self):
        self.paths = []
        self.contents = []

    def ingest_file(self, path: Path):
        self.paths.append(path)
        self.contents.append(path.read_bytes())
        return {"file": path.name, "chunks_added": 2, "total_chunks": 2}


def test_uploads_are_processed_from_temporary_paths_and_cleaned_afterward():
    agent = RecordingAgent()
    result = ingest_uploaded_files(agent, [FakeUpload("safety.txt", b"Safety information")])
    assert result[0]["status"] == "Indexed"
    assert result[0]["chunks_added"] == 2
    assert agent.contents == [b"Safety information"]
    assert not agent.paths[0].exists()


def test_unsupported_file_has_a_clear_result_and_does_not_abort_batch():
    agent = RecordingAgent()
    result = ingest_uploaded_files(
        agent,
        [FakeUpload("unsafe.exe", b"not supported"), FakeUpload("manual.md", b"Procedure text")],
    )
    assert result[0]["status"] == "Not loaded"
    assert "Unsupported file type" in result[0]["message"]
    assert result[1]["status"] == "Indexed"
    assert len(agent.paths) == 1


def test_file_and_batch_size_limits_are_checked():
    agent = RecordingAgent()
    too_large = ingest_uploaded_files(agent, [FakeUpload("large.txt", b"12345")], max_file_bytes=4)
    assert too_large[0]["status"] == "Not loaded"
    too_many = ingest_uploaded_files(
        agent,
        [FakeUpload("a.txt", b"123"), FakeUpload("b.txt", b"456")],
        max_file_bytes=4,
        max_batch_bytes=5,
    )
    assert len(too_many) == 1
    assert "total limit" in too_many[0]["message"]
    assert agent.paths == []


def test_duplicate_filenames_get_distinct_temporary_paths():
    agent = RecordingAgent()
    result = ingest_uploaded_files(
        agent,
        [FakeUpload("manual.txt", b"first"), FakeUpload("manual.txt", b"second")],
    )
    assert [item["status"] for item in result] == ["Indexed", "Indexed"]
    assert agent.paths[0].name != agent.paths[1].name
    assert all(not path.exists() for path in agent.paths)


def test_html_answer_report_escapes_uploaded_content():
    report = build_answer_report(
        question="<script>alert(1)</script>",
        answer="A safe answer",
        sources=[{"source": "manual.txt", "score": 0.9, "snippet": "<img src=x>"}],
        tool_trace=[],
        backend="DemoLLM",
    )
    assert "<script>alert(1)</script>" not in report
    assert "&lt;script&gt;" in report
    assert "&lt;img" in report


def test_sample_archive_is_created_in_memory(tmp_path: Path):
    sample_dir = tmp_path / "samples"
    sample_dir.mkdir()
    (sample_dir / "manual.txt").write_text("A fictional manual.", encoding="utf-8")
    (sample_dir / "ignored.exe").write_bytes(b"x")
    archive_data = build_sample_archive(sample_dir)
    with ZipFile(BytesIO(archive_data)) as archive:
        assert archive.namelist() == ["manual.txt"]
        assert archive.read("manual.txt") == b"A fictional manual."


def test_pdf_report_is_explicitly_optional():
    result = build_answer_pdf("Question", "Answer", [], [], "DemoLLM")
    assert result is None or result.startswith(b"%PDF")
