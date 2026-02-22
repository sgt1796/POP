from __future__ import annotations

import base64
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import requests

from ..agent_types import AgentTool, AgentToolResult, TextContent

try:
    from google.auth.transport.requests import Request as GoogleAuthRequest
    from google.oauth2.credentials import Credentials
except Exception:  # pragma: no cover - exercised through runtime error path
    GoogleAuthRequest = None
    Credentials = None

try:
    from pypdf import PdfWriter
except Exception:  # pragma: no cover - exercised through runtime error path
    PdfWriter = None


GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
_GMAIL_API_ROOT = "https://gmail.googleapis.com/gmail/v1/users/me"


def _sanitize_filename(name: str) -> str:
    candidate = os.path.basename(str(name or "").strip())
    if not candidate:
        return "attachment"
    # Remove dangerous/surprising characters while keeping readable filenames.
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", candidate)
    safe = safe.strip("._")
    return safe or "attachment"


def _decode_b64url(data: str) -> bytes:
    text = str(data or "")
    padded = text + ("=" * (-len(text) % 4))
    return base64.urlsafe_b64decode(padded.encode("utf-8"))


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    key = str(value).strip().lower()
    if key in {"1", "true", "yes", "y", "on"}:
        return True
    if key in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _path_in_workspace(path: str, workspace_root: str) -> bool:
    try:
        return os.path.commonpath([workspace_root, path]) == workspace_root
    except ValueError:
        return False


def _resolve_workspace_path(path_value: str, workspace_root: str) -> str:
    candidate = str(path_value).strip()
    if not candidate:
        raise ValueError("path is required")
    resolved = os.path.realpath(candidate if os.path.isabs(candidate) else os.path.join(workspace_root, candidate))
    if not _path_in_workspace(resolved, workspace_root):
        raise ValueError("path_outside_workspace")
    return resolved


def _header_lookup(headers: Sequence[Dict[str, Any]], name: str) -> str:
    target = str(name).strip().lower()
    for header in headers:
        if str(header.get("name", "")).strip().lower() == target:
            return str(header.get("value", ""))
    return ""


def _iter_message_parts(payload: Optional[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    if not payload:
        return []
    parts = payload.get("parts") or []
    if not parts:
        return [payload]
    flattened: List[Dict[str, Any]] = []
    stack: List[Dict[str, Any]] = list(parts)
    while stack:
        part = stack.pop(0)
        nested = part.get("parts") or []
        if nested:
            stack = list(nested) + stack
            continue
        flattened.append(part)
    return flattened


def _extension_from_mime(mime_type: str) -> str:
    key = str(mime_type or "").strip().lower()
    if key == "application/pdf":
        return ".pdf"
    if key in {"image/jpeg", "image/jpg"}:
        return ".jpg"
    if key == "image/png":
        return ".png"
    if key == "text/plain":
        return ".txt"
    return ".bin"


class GmailFetchTool(AgentTool):
    name = "gmail_fetch"
    description = (
        "Fetch Gmail messages by sender/query and optionally download attachments "
        "(workspace-only attachment output path)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Gmail search query fragment"},
            "sender": {"type": "string", "description": "Sender email filter; mapped to from:<sender>"},
            "max_results": {"type": "integer", "description": "Maximum messages to fetch (1-50)"},
            "download_attachments": {
                "type": "boolean",
                "description": "Download attachment files for matched messages (default true)",
            },
            "attachment_dir": {
                "type": "string",
                "description": "Attachment output directory (must be inside workspace)",
            },
            "token_path": {
                "type": "string",
                "description": "Path to Gmail OAuth token JSON file (default token.json)",
            },
            "include_spam_trash": {
                "type": "boolean",
                "description": "Whether to include spam/trash in search",
            },
        },
    }
    label = "Gmail Fetch"

    def __init__(self, workspace_root: Optional[str] = None) -> None:
        self.workspace_root = os.path.realpath(str(workspace_root or os.getcwd()))

    @staticmethod
    def _error(text: str, details: Dict[str, Any]) -> AgentToolResult:
        return AgentToolResult(content=[TextContent(type="text", text=text)], details={"ok": False, **details})

    @staticmethod
    def _ok(text: str, details: Dict[str, Any]) -> AgentToolResult:
        return AgentToolResult(content=[TextContent(type="text", text=text)], details={"ok": True, **details})

    def _load_credentials(self, token_path: str) -> Any:
        if Credentials is None or GoogleAuthRequest is None:
            raise RuntimeError("gmail_fetch requires google-auth. Add google-auth to project dependencies.")

        if not os.path.exists(token_path):
            raise RuntimeError(
                "token file not found. Run a one-time Gmail OAuth bootstrap to create token.json, then retry."
            )

        try:
            creds = Credentials.from_authorized_user_file(token_path, GMAIL_SCOPES)
        except Exception as exc:
            raise RuntimeError(f"failed to parse token file: {exc}") from exc

        if not creds.valid:
            if creds.expired and creds.refresh_token:
                try:
                    creds.refresh(GoogleAuthRequest())
                except Exception as exc:
                    raise RuntimeError(f"failed to refresh Gmail token: {exc}") from exc
                try:
                    with open(token_path, "w", encoding="utf-8") as handle:
                        handle.write(creds.to_json())
                except Exception:
                    # Refresh succeeded even if writeback fails; continue using in-memory token.
                    pass
            else:
                raise RuntimeError(
                    "token is missing/expired and cannot be refreshed in non-interactive mode. "
                    "Generate or refresh token.json manually, then retry."
                )

        if not getattr(creds, "token", None):
            raise RuntimeError("Gmail credentials are missing an access token after refresh.")

        return creds

    def _request_json(self, token: str, url: str, *, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            params=params,
            timeout=30,
        )
        if response.status_code >= 400:
            message = response.text.strip() or "unknown error"
            raise RuntimeError(f"gmail api error {response.status_code}: {message[:200]}")
        try:
            return dict(response.json())
        except Exception as exc:
            raise RuntimeError(f"gmail api returned invalid json: {exc}") from exc

    def _build_query(self, sender: str, query: str) -> str:
        parts: List[str] = []
        sender_value = str(sender or "").strip()
        query_value = str(query or "").strip()
        if sender_value:
            parts.append(f"from:{sender_value}")
        if query_value:
            parts.append(query_value)
        return " ".join(parts).strip()

    def _next_attachment_path(
        self,
        attachment_dir: str,
        message_id: str,
        filename: str,
        seen_paths: Set[str],
    ) -> str:
        safe_base = _sanitize_filename(filename)
        prefix = _sanitize_filename(message_id) or "msg"
        base_name = f"{prefix}_{safe_base}"
        stem, ext = os.path.splitext(base_name)
        candidate = os.path.join(attachment_dir, base_name)
        index = 1
        while candidate in seen_paths or os.path.exists(candidate):
            suffix = f"_{index}"
            candidate = os.path.join(attachment_dir, f"{stem}{suffix}{ext}")
            index += 1
        seen_paths.add(candidate)
        return candidate

    def _fetch_attachment_bytes(self, token: str, message_id: str, part: Dict[str, Any]) -> bytes:
        body = part.get("body") or {}
        if body.get("data"):
            return _decode_b64url(str(body.get("data")))

        attachment_id = str(body.get("attachmentId", "")).strip()
        if not attachment_id:
            return b""

        payload = self._request_json(
            token,
            f"{_GMAIL_API_ROOT}/messages/{message_id}/attachments/{attachment_id}",
        )
        data = str(payload.get("data", "")).strip()
        return _decode_b64url(data) if data else b""

    def _fetch_impl(self, params: Dict[str, Any]) -> AgentToolResult:
        sender = str(params.get("sender", "")).strip()
        query_raw = str(params.get("query", "")).strip()
        try:
            max_results = int(params.get("max_results", 5) or 5)
        except Exception:
            max_results = 5
        max_results = max(1, min(max_results, 50))

        download_attachments = _coerce_bool(params.get("download_attachments"), True)
        include_spam_trash = _coerce_bool(params.get("include_spam_trash"), False)
        token_path_raw = str(params.get("token_path", "token.json") or "token.json").strip()
        token_path = os.path.realpath(
            token_path_raw if os.path.isabs(token_path_raw) else os.path.join(self.workspace_root, token_path_raw)
        )
        query = self._build_query(sender, query_raw)

        attachment_dir = ""
        if download_attachments:
            dir_raw = str(
                params.get("attachment_dir", os.path.join("agent", "mem", "gmail_attachments"))
                or os.path.join("agent", "mem", "gmail_attachments")
            ).strip()
            try:
                attachment_dir = _resolve_workspace_path(dir_raw, self.workspace_root)
            except ValueError:
                return self._error(
                    "gmail_fetch error: attachment_dir must be inside workspace.",
                    {
                        "error": "path_outside_workspace",
                        "attachment_dir": dir_raw,
                        "workspace_root": self.workspace_root,
                    },
                )
            os.makedirs(attachment_dir, exist_ok=True)

        try:
            creds = self._load_credentials(token_path)
        except Exception as exc:
            return self._error(
                f"gmail_fetch auth error: {exc}",
                {
                    "error": str(exc),
                    "token_path": token_path,
                },
            )

        list_params: Dict[str, Any] = {
            "maxResults": max_results,
            "includeSpamTrash": bool(include_spam_trash),
        }
        if query:
            list_params["q"] = query

        try:
            listing = self._request_json(f"{creds.token}", f"{_GMAIL_API_ROOT}/messages", params=list_params)
        except Exception as exc:
            return self._error(
                f"gmail_fetch api error: {exc}",
                {
                    "error": str(exc),
                    "query": query,
                    "max_results": max_results,
                },
            )

        ids = [str(item.get("id", "")).strip() for item in list(listing.get("messages") or [])]
        ids = [item for item in ids if item]

        messages_out: List[Dict[str, Any]] = []
        downloaded_paths: List[str] = []
        pdf_paths: List[str] = []
        seen_download_paths: Set[str] = set()
        errors: List[str] = []

        for message_id in ids:
            try:
                message = self._request_json(
                    f"{creds.token}",
                    f"{_GMAIL_API_ROOT}/messages/{message_id}",
                    params={"format": "full"},
                )
            except Exception as exc:
                errors.append(f"message {message_id}: {exc}")
                continue

            payload = dict(message.get("payload") or {})
            headers = list(payload.get("headers") or [])
            attachments_out: List[Dict[str, Any]] = []

            for index, part in enumerate(_iter_message_parts(payload), start=1):
                body = dict(part.get("body") or {})
                has_attachment_payload = bool(body.get("attachmentId") or body.get("data"))
                if not has_attachment_payload:
                    continue

                filename_raw = str(part.get("filename", "")).strip()
                mime_type = str(part.get("mimeType", "")).strip()
                if filename_raw:
                    filename = _sanitize_filename(filename_raw)
                else:
                    ext = _extension_from_mime(mime_type)
                    filename = f"attachment_{index}{ext}"

                attachment_item: Dict[str, Any] = {
                    "filename": filename,
                    "mime_type": mime_type,
                    "size": int(body.get("size", 0) or 0),
                    "attachment_id": str(body.get("attachmentId", "") or ""),
                    "downloaded": False,
                    "path": "",
                }

                if download_attachments:
                    try:
                        data = self._fetch_attachment_bytes(f"{creds.token}", message_id, part)
                        output_path = self._next_attachment_path(attachment_dir, message_id, filename, seen_download_paths)
                        with open(output_path, "wb") as handle:
                            handle.write(data)
                        attachment_item["downloaded"] = True
                        attachment_item["path"] = output_path
                        downloaded_paths.append(output_path)
                        if output_path.lower().endswith(".pdf"):
                            pdf_paths.append(output_path)
                    except Exception as exc:
                        attachment_item["error"] = str(exc)

                attachments_out.append(attachment_item)

            messages_out.append(
                {
                    "id": message_id,
                    "thread_id": str(message.get("threadId", "") or ""),
                    "snippet": str(message.get("snippet", "") or ""),
                    "from": _header_lookup(headers, "From"),
                    "subject": _header_lookup(headers, "Subject"),
                    "date": _header_lookup(headers, "Date"),
                    "attachment_count": len(attachments_out),
                    "attachments": attachments_out,
                }
            )

        summary_lines = [
            f"gmail_fetch: fetched {len(messages_out)} message(s) with query={query or '(none)'}",
            f"downloaded_attachments={len(downloaded_paths)}",
            f"pdf_attachments={len(pdf_paths)}",
        ]
        if errors:
            summary_lines.append(f"partial_errors={len(errors)}")

        return self._ok(
            "\n".join(summary_lines),
            {
                "query": query,
                "sender": sender,
                "max_results": max_results,
                "include_spam_trash": include_spam_trash,
                "token_path": token_path,
                "message_count": len(messages_out),
                "messages": messages_out,
                "download_attachments": download_attachments,
                "attachment_dir": attachment_dir,
                "downloaded_paths": downloaded_paths,
                "pdf_attachment_paths": pdf_paths,
                "errors": errors,
            },
        )

    async def execute(
        self,
        tool_call_id: str,
        params: Dict[str, Any],
        signal: Optional[Any] = None,
        on_update: Optional[Any] = None,
    ) -> AgentToolResult:
        del tool_call_id, signal, on_update
        return self._fetch_impl(params)


class PdfMergeTool(AgentTool):
    name = "pdf_merge"
    description = "Merge multiple PDFs into one output PDF (workspace-only read/write paths)."
    parameters = {
        "type": "object",
        "properties": {
            "input_paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Input PDF paths to merge",
            },
            "output_path": {
                "type": "string",
                "description": "Merged output PDF path",
            },
            "overwrite": {
                "type": "boolean",
                "description": "Overwrite output if it already exists (default false)",
            },
        },
        "required": ["input_paths", "output_path"],
    }
    label = "PDF Merge"

    def __init__(self, workspace_root: Optional[str] = None) -> None:
        self.workspace_root = os.path.realpath(str(workspace_root or os.getcwd()))

    @staticmethod
    def _error(text: str, details: Dict[str, Any]) -> AgentToolResult:
        return AgentToolResult(content=[TextContent(type="text", text=text)], details={"ok": False, **details})

    @staticmethod
    def _ok(text: str, details: Dict[str, Any]) -> AgentToolResult:
        return AgentToolResult(content=[TextContent(type="text", text=text)], details={"ok": True, **details})

    def _merge_impl(self, params: Dict[str, Any]) -> AgentToolResult:
        if PdfWriter is None:
            return self._error(
                "pdf_merge error: pypdf is not installed.",
                {"error": "missing_dependency:pypdf"},
            )

        raw_inputs = params.get("input_paths", [])
        if not isinstance(raw_inputs, list) or not raw_inputs:
            return self._error(
                "pdf_merge error: input_paths must be a non-empty array.",
                {"error": "invalid_input_paths"},
            )

        output_raw = str(params.get("output_path", "")).strip()
        if not output_raw:
            return self._error(
                "pdf_merge error: missing output_path.",
                {"error": "missing_output_path"},
            )

        overwrite = _coerce_bool(params.get("overwrite"), False)

        try:
            output_path = _resolve_workspace_path(output_raw, self.workspace_root)
        except ValueError:
            return self._error(
                "pdf_merge error: output_path must be inside workspace.",
                {
                    "error": "path_outside_workspace",
                    "output_path": output_raw,
                    "workspace_root": self.workspace_root,
                },
            )

        if os.path.exists(output_path) and not overwrite:
            return self._error(
                "pdf_merge error: output file already exists and overwrite=false.",
                {
                    "error": "output_exists",
                    "output_path": output_path,
                },
            )

        candidates: List[str] = []
        skipped: List[Dict[str, str]] = []
        for item in raw_inputs:
            raw = str(item).strip()
            if not raw:
                skipped.append({"input": raw, "reason": "empty_path"})
                continue
            try:
                resolved = _resolve_workspace_path(raw, self.workspace_root)
            except ValueError:
                skipped.append({"input": raw, "reason": "path_outside_workspace"})
                continue
            if not os.path.exists(resolved):
                skipped.append({"input": raw, "reason": "not_found"})
                continue
            if not os.path.isfile(resolved):
                skipped.append({"input": raw, "reason": "not_a_file"})
                continue
            if not resolved.lower().endswith(".pdf"):
                skipped.append({"input": raw, "reason": "not_pdf_extension"})
                continue
            candidates.append(resolved)

        if not candidates:
            return self._error(
                "pdf_merge error: no valid PDF inputs to merge.",
                {
                    "error": "no_valid_pdf_inputs",
                    "skipped": skipped,
                },
            )

        os.makedirs(os.path.dirname(output_path) or self.workspace_root, exist_ok=True)

        merged_inputs: List[str] = []
        writer = PdfWriter()
        try:
            for candidate in candidates:
                try:
                    writer.append(candidate)
                    merged_inputs.append(candidate)
                except Exception:
                    skipped.append({"input": candidate, "reason": "invalid_pdf_content"})

            if not merged_inputs:
                return self._error(
                    "pdf_merge error: no readable PDF inputs remained after validation.",
                    {
                        "error": "no_readable_pdf_inputs",
                        "skipped": skipped,
                    },
                )

            with open(output_path, "wb") as handle:
                writer.write(handle)
        except Exception as exc:
            return self._error(
                f"pdf_merge error: {exc}",
                {
                    "error": str(exc),
                    "output_path": output_path,
                },
            )
        finally:
            close_writer = getattr(writer, "close", None)
            if callable(close_writer):
                close_writer()

        summary = (
            f"pdf_merge: merged {len(merged_inputs)} PDF(s) -> {output_path} "
            f"(skipped={len(skipped)})"
        )
        return self._ok(
            summary,
            {
                "input_paths": [str(x) for x in raw_inputs],
                "merged_inputs": merged_inputs,
                "merged_count": len(merged_inputs),
                "skipped": skipped,
                "skipped_count": len(skipped),
                "output_path": output_path,
                "overwrite": overwrite,
            },
        )

    async def execute(
        self,
        tool_call_id: str,
        params: Dict[str, Any],
        signal: Optional[Any] = None,
        on_update: Optional[Any] = None,
    ) -> AgentToolResult:
        del tool_call_id, signal, on_update
        return self._merge_impl(params)
