from __future__ import annotations

import asyncio
import base64
import io
import json
from pathlib import Path
from typing import Any, Dict, Optional

from pypdf import PdfReader, PdfWriter

from agent.tools import GmailFetchTool, PdfMergeTool


class _FakeResponse:
    def __init__(self, *, status_code: int = 200, payload: Optional[Dict[str, Any]] = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self) -> Dict[str, Any]:
        return dict(self._payload)


def _run(tool: Any, params: Dict[str, Any]):
    return asyncio.run(tool.execute("tc1", params))


def _write_token(path: Path) -> None:
    payload = {
        "token": "access-token",
        "refresh_token": "refresh-token",
        "token_uri": "https://oauth2.googleapis.com/token",
        "client_id": "client-id",
        "client_secret": "client-secret",
        "scopes": ["https://www.googleapis.com/auth/gmail.readonly"],
        "expiry": "2099-01-01T00:00:00Z",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _pdf_bytes() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _write_pdf(path: Path) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with path.open("wb") as handle:
        writer.write(handle)


def test_gmail_fetch_returns_error_when_token_missing(tmp_path: Path):
    tool = GmailFetchTool(workspace_root=str(tmp_path))
    result = _run(tool, {"token_path": "missing-token.json", "download_attachments": False})

    assert result.details["ok"] is False
    assert "token file not found" in result.content[0].text.lower()


def test_gmail_fetch_composes_query_from_sender_and_query(tmp_path: Path, monkeypatch):
    token_path = tmp_path / "token.json"
    _write_token(token_path)

    calls = []

    def _fake_get(url, headers=None, params=None, timeout=None):
        calls.append({"url": url, "params": dict(params or {})})
        return _FakeResponse(payload={"messages": []})

    monkeypatch.setattr("agent.tools.gmail_pdf_tools.requests.get", _fake_get)

    tool = GmailFetchTool(workspace_root=str(tmp_path))
    result = _run(
        tool,
        {
            "token_path": str(token_path),
            "sender": "billing@example.com",
            "query": "subject:invoice newer_than:30d",
            "max_results": 7,
            "download_attachments": False,
        },
    )

    assert result.details["ok"] is True
    assert result.details["query"] == "from:billing@example.com subject:invoice newer_than:30d"
    assert result.details["message_count"] == 0
    assert calls
    assert calls[0]["params"]["q"] == "from:billing@example.com subject:invoice newer_than:30d"
    assert calls[0]["params"]["maxResults"] == 7


def test_gmail_fetch_downloads_attachments_and_reports_pdf_paths(tmp_path: Path, monkeypatch):
    token_path = tmp_path / "token.json"
    _write_token(token_path)

    pdf_data = _pdf_bytes()
    pdf_b64 = base64.urlsafe_b64encode(pdf_data).decode("utf-8").rstrip("=")
    png_b64 = base64.urlsafe_b64encode(b"png-bytes").decode("utf-8").rstrip("=")

    def _fake_get(url, headers=None, params=None, timeout=None):
        if url.endswith("/messages"):
            return _FakeResponse(payload={"messages": [{"id": "m1"}]})
        if url.endswith("/messages/m1"):
            return _FakeResponse(
                payload={
                    "id": "m1",
                    "threadId": "t1",
                    "snippet": "snippet",
                    "payload": {
                        "headers": [
                            {"name": "From", "value": "billing@example.com"},
                            {"name": "Subject", "value": "Invoice"},
                        ],
                        "parts": [
                            {
                                "filename": "invoice.pdf",
                                "mimeType": "application/pdf",
                                "body": {"attachmentId": "a1", "size": len(pdf_data)},
                            },
                            {
                                "filename": "preview.png",
                                "mimeType": "image/png",
                                "body": {"data": png_b64, "size": 9},
                            },
                        ],
                    },
                }
            )
        if url.endswith("/messages/m1/attachments/a1"):
            return _FakeResponse(payload={"data": pdf_b64})
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr("agent.tools.gmail_pdf_tools.requests.get", _fake_get)

    tool = GmailFetchTool(workspace_root=str(tmp_path))
    result = _run(
        tool,
        {
            "token_path": str(token_path),
            "download_attachments": True,
            "attachment_dir": "downloads",
        },
    )

    assert result.details["ok"] is True
    assert len(result.details["downloaded_paths"]) == 2
    assert len(result.details["pdf_attachment_paths"]) == 1
    for path in result.details["downloaded_paths"]:
        assert Path(path).exists()


def test_gmail_fetch_rejects_attachment_dir_outside_workspace(tmp_path: Path):
    tool = GmailFetchTool(workspace_root=str(tmp_path))
    result = _run(
        tool,
        {
            "download_attachments": True,
            "attachment_dir": "/tmp",
        },
    )

    assert result.details["ok"] is False
    assert result.details["error"] == "path_outside_workspace"


def test_pdf_merge_merges_valid_pdfs(tmp_path: Path):
    _write_pdf(tmp_path / "a.pdf")
    _write_pdf(tmp_path / "b.pdf")

    tool = PdfMergeTool(workspace_root=str(tmp_path))
    result = _run(
        tool,
        {
            "input_paths": ["a.pdf", "b.pdf"],
            "output_path": "merged/output.pdf",
        },
    )

    assert result.details["ok"] is True
    assert result.details["merged_count"] == 2
    merged_path = Path(result.details["output_path"])
    assert merged_path.exists()
    assert len(PdfReader(str(merged_path)).pages) == 2


def test_pdf_merge_skips_non_pdf_inputs(tmp_path: Path):
    _write_pdf(tmp_path / "a.pdf")
    (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")

    tool = PdfMergeTool(workspace_root=str(tmp_path))
    result = _run(
        tool,
        {
            "input_paths": ["a.pdf", "notes.txt"],
            "output_path": "out.pdf",
        },
    )

    assert result.details["ok"] is True
    assert result.details["merged_count"] == 1
    assert result.details["skipped_count"] == 1
    assert result.details["skipped"][0]["reason"] == "not_pdf_extension"


def test_pdf_merge_rejects_output_outside_workspace(tmp_path: Path):
    _write_pdf(tmp_path / "a.pdf")

    tool = PdfMergeTool(workspace_root=str(tmp_path))
    result = _run(
        tool,
        {
            "input_paths": ["a.pdf"],
            "output_path": "/tmp/merged.pdf",
        },
    )

    assert result.details["ok"] is False
    assert result.details["error"] == "path_outside_workspace"
