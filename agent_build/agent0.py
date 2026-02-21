import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from POP.embedder import Embedder
from POP.stream import stream

from agent import Agent
from agent.agent_types import AgentTool, AgentToolResult, TextContent
from agent.tools import BashExecConfig, BashExecTool, FastTool, SlowTool, WebSnapshotTool

# Logging helpers
LOG_LEVELS = {
    "quiet": 0,
    "messages": 1,
    "stream": 2,
    "debug": 3,
}

USER_PROMPT_MARKER = "|Current user message|:\n"
DEFAULT_TOOLSMAKER_ALLOWED_CAPS = "fs_read,fs_write,http"
TOOL_CAPABILITIES = {"fs_read", "fs_write", "http", "secrets"}
BASH_READ_COMMANDS = {"pwd", "ls", "cat", "head", "tail", "wc", "find", "rg", "git", "echo", "df", "du"}
BASH_WRITE_COMMANDS = {"mkdir", "touch", "cp", "mv", "rm"}
BASH_GIT_READ_SUBCOMMANDS = {"status", "diff", "log", "show", "branch"}


def _parse_toolsmaker_allowed_capabilities(value: Optional[str]) -> List[str]:
    raw = str(value if value is not None else DEFAULT_TOOLSMAKER_ALLOWED_CAPS)
    caps: List[str] = []
    seen = set()
    for item in raw.split(","):
        cap = item.strip()
        if not cap or cap in seen:
            continue
        if cap in TOOL_CAPABILITIES:
            caps.append(cap)
            seen.add(cap)
    if not caps:
        return ["fs_read", "fs_write", "http"]
    return caps


def _parse_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    key = str(value).strip().lower()
    if key in {"1", "true", "yes", "y", "on"}:
        return True
    if key in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _parse_float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _parse_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def _parse_path_list_env(name: str, default_paths: Sequence[str], base_dir: str) -> List[str]:
    value = os.getenv(name)
    raw_items = []
    if value is None:
        raw_items = [str(item) for item in default_paths]
    else:
        raw_items = [x.strip() for x in str(value).split(",") if x.strip()]
        if not raw_items:
            raw_items = [str(item) for item in default_paths]

    normalized: List[str] = []
    seen = set()
    for item in raw_items:
        candidate = item if os.path.isabs(item) else os.path.join(base_dir, item)
        root = os.path.realpath(candidate)
        if root in seen:
            continue
        normalized.append(root)
        seen.add(root)
    return normalized


def _sorted_csv(values: Sequence[str]) -> str:
    cleaned = {str(item).strip() for item in values if str(item).strip()}
    return ", ".join(sorted(cleaned))


def _resolve_log_level(value: str) -> int:
    if not value:
        return LOG_LEVELS["quiet"]
    key = str(value).strip().lower()
    if key.isdigit():
        return int(key)
    return LOG_LEVELS.get(key, LOG_LEVELS["quiet"])


def _extract_texts(message: Any) -> List[str]:
    texts: List[str] = []
    if not message:
        return texts
    content = getattr(message, "content", None)
    if not content:
        return texts
    for item in content:
        if isinstance(item, TextContent):
            texts.append(item.text or "")
        elif isinstance(item, dict) and item.get("type") == "text":
            texts.append(str(item.get("text", "")))
    return texts


def _extract_latest_assistant_text(agent: Agent) -> str:
    for message in reversed(agent.state.messages):
        if getattr(message, "role", None) != "assistant":
            continue
        text = "\n".join([t for t in _extract_texts(message) if t.strip()]).strip()
        if text:
            return text
    return ""


def _extract_original_user_message(text: str) -> str:
    if USER_PROMPT_MARKER in text:
        return text.split(USER_PROMPT_MARKER, 1)[1].strip()
    return text.strip()


def _format_message_line(message: Any) -> str:
    role = getattr(message, "role", "unknown")
    text = "\n".join(_extract_texts(message)).strip()
    return f"[event] {role}: {text}"


def _extract_bash_exec_command(event: Dict[str, Any]) -> str:
    args = event.get("args")
    if isinstance(args, dict):
        cmd = args.get("cmd")
        if isinstance(cmd, str) and cmd.strip():
            return cmd.strip()

    result = event.get("result")
    details = getattr(result, "details", None)
    if isinstance(details, dict):
        cmd = details.get("command")
        if isinstance(cmd, str) and cmd.strip():
            return cmd.strip()
    if isinstance(result, dict):
        details = result.get("details")
        if isinstance(details, dict):
            cmd = details.get("command")
            if isinstance(cmd, str) and cmd.strip():
                return cmd.strip()
    return ""


def make_event_logger(level: str = "quiet"):
    """Create an event logger function for agent events.
    Levels:
    - quiet: no logging
    - messages: log completed messages
    - stream: log message updates/streams
    - debug: log all events
    """
    level_value = _resolve_log_level(level)

    def log(event: Dict[str, Any]) -> None:
        etype = event.get("type")
        if level_value <= LOG_LEVELS["quiet"]:
            if etype == "tool_execution_end" and str(event.get("toolName", "")).strip() == "bash_exec":
                command = _extract_bash_exec_command(event)
                if command:
                    print(f"Ran command {command}")
                else:
                    print("Ran command")
            return

        if etype == "tool_execution_start":
            print(f"[tool:start] {event.get('toolName')} args={event.get('args')}")
            return
        if etype == "tool_execution_end":
            print(f"[tool:end] {event.get('toolName')} error={event.get('isError')}")
            return
        if etype == "message_end" and level_value >= LOG_LEVELS["messages"]:
            message = event.get("message")
            if message:
                print(_format_message_line(message))
            return
        if etype == "message_update" and level_value >= LOG_LEVELS["stream"]:
            assistant_event = event.get("assistantMessageEvent") or {}
            if assistant_event.get("type") == "text_delta":
                delta = assistant_event.get("delta")
                if delta:
                    print(f"[stream] {delta}")
            return
        if level_value >= LOG_LEVELS["debug"]:
            print(f"[debug] {event}")

    return log


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


@dataclass
class MemoryEntry:
    text: str
    embedding: np.ndarray


class ConversationMemory:
    """Short-term in-memory vector store."""

    def __init__(self, embedder: Embedder, max_entries: int = 100) -> None:
        self.embedder = embedder
        self.max_entries = max_entries
        self._entries: List[MemoryEntry] = []

    def add(self, text: str) -> None:
        embedding = self.embedder.get_embedding([text])[0]
        self._entries.append(MemoryEntry(text=text, embedding=embedding))
        if len(self._entries) > self.max_entries:
            self._entries.pop(0)

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        if not self._entries:
            return []
        query_emb = self.embedder.get_embedding([query])[0]
        scores = [_cosine_similarity(query_emb, entry.embedding) for entry in self._entries]
        k = max(1, min(int(top_k or 1), len(self._entries)))
        top_indices = np.argsort(scores)[-k:][::-1]
        return [self._entries[i].text for i in top_indices]


class DiskMemory:
    """
    Persistent text+vector memory split across two files:
      - <base>.text.jsonl
      - <base>.embeddings.npy
    """

    def __init__(self, filepath: str, embedder: Embedder, max_entries: int = 1000) -> None:
        self.base = os.path.splitext(filepath)[0] if filepath.endswith(".jsonl") else filepath
        self.text_path = f"{self.base}.text.jsonl"
        self.emb_path = f"{self.base}.embeddings.npy"
        self.embedder = embedder
        self.max_entries = max_entries
        os.makedirs(os.path.dirname(self.text_path) or ".", exist_ok=True)
        self._n_text = self._count_lines(self.text_path)

    def add(self, text: str) -> None:
        with open(self.text_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        self._n_text += 1

        vec = self.embedder.get_embedding([text])[0].astype("float32")
        if os.path.exists(self.emb_path):
            matrix = np.load(self.emb_path, mmap_mode=None, allow_pickle=False)
            matrix = np.vstack([matrix, vec[None, :]])
        else:
            matrix = vec[None, :]
        np.save(self.emb_path, matrix)

        self._prune()

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        if not os.path.exists(self.emb_path) or self._n_text == 0:
            return []

        query_vec = self.embedder.get_embedding([query])[0].astype("float32")
        matrix = np.load(self.emb_path, mmap_mode="r")

        def _norm(arr: np.ndarray) -> np.ndarray:
            norms = np.linalg.norm(arr, axis=-1, keepdims=True)
            norms[norms == 0] = 1.0
            return arr / norms

        query_n = _norm(query_vec[None, :])[0]
        matrix_n = _norm(matrix)
        sims = (matrix_n @ query_n).astype("float32")
        k = max(1, min(int(top_k or 1), sims.shape[0]))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return self._read_lines_by_index(idx.tolist())

    def _count_lines(self, path: str) -> int:
        if not os.path.exists(path):
            return 0
        with open(path, "rb") as f:
            return sum(1 for _ in f)

    def _read_lines_by_index(self, indices: Sequence[int]) -> List[str]:
        if not indices:
            return []
        wanted = set(indices)
        found: Dict[int, str] = {}
        with open(self.text_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i in wanted:
                    try:
                        found[i] = str(json.loads(line)["text"])
                    except Exception:
                        found[i] = line.strip()
                if len(found) == len(wanted):
                    break
        ordered = [found.get(i, "") for i in indices]
        return [t for t in ordered if t]

    def _prune(self) -> None:
        if self._n_text <= self.max_entries:
            return
        keep = self.max_entries

        with open(self.text_path, "rb") as f:
            lines = f.readlines()[-keep:]
        with open(self.text_path, "wb") as f:
            f.writelines(lines)
        self._n_text = keep

        if os.path.exists(self.emb_path):
            matrix = np.load(self.emb_path, mmap_mode=None, allow_pickle=False)
            if len(matrix) > keep:
                np.save(self.emb_path, matrix[-keep:])


class MemoryRetriever:
    """Shared retrieval service for prompt injection and memory tool calls."""

    def __init__(self, short_term: ConversationMemory, long_term: Optional[DiskMemory] = None) -> None:
        self.short_term = short_term
        self.long_term = long_term

    def retrieve_sections(self, query: str, top_k: int = 3, scope: str = "both") -> Tuple[List[str], List[str]]:
        scope = (scope or "both").strip().lower()
        if scope not in {"short", "long", "both"}:
            scope = "both"
        k = max(1, int(top_k or 1))
        short_hits: List[str] = []
        long_hits: List[str] = []
        if scope in {"short", "both"}:
            short_hits = self.short_term.retrieve(query, top_k=k)
        if scope in {"long", "both"} and self.long_term is not None:
            long_hits = self.long_term.retrieve(query, top_k=k)
        return short_hits, long_hits

    def retrieve(self, query: str, top_k: int = 3, scope: str = "both") -> List[str]:
        short_hits, long_hits = self.retrieve_sections(query, top_k=top_k, scope=scope)
        seen = set()
        merged: List[str] = []
        for item in short_hits + long_hits:
            if item not in seen:
                merged.append(item)
                seen.add(item)
        return merged


def _format_memory_sections(short_hits: List[str], long_hits: List[str]) -> str:
    sections: List[str] = []
    if short_hits:
        sections.append("|Short-term memory|:\n" + "\n".join(f"- {x}" for x in short_hits))
    if long_hits:
        sections.append("|Long-term memory|:\n" + "\n".join(f"- {x}" for x in long_hits))
    if not sections:
        return "(no relevant memories)"
    return "\n\n".join(sections)


def _build_augmented_prompt(user_message: str, memory_text: str) -> str:
    return (
        "Use the memory context as soft background. It may be incomplete or outdated.\n"
        "Prioritize the current user message and ask follow-up questions when needed.\n\n"
        f"Memory context:\n{memory_text}\n\n"
        f"{USER_PROMPT_MARKER}{user_message.strip()}"
    )


class EmbeddingIngestionWorker:
    """Background ingestion worker for embedding writes."""

    def __init__(self, memory: ConversationMemory, long_term: Optional[DiskMemory] = None) -> None:
        self.memory = memory
        self.long_term = long_term
        self._queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        self._task: Optional[asyncio.Task[None]] = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    def enqueue(self, role: str, text: str) -> None:
        value = text.strip()
        if not value:
            return
        self._queue.put_nowait(f"{role}: {value}")

    async def _run(self) -> None:
        while True:
            item = await self._queue.get()
            try:
                if item is None:
                    return
                await asyncio.to_thread(self.memory.add, item)
                if self.long_term is not None:
                    await asyncio.to_thread(self.long_term.add, item)
            except Exception as exc:
                print(f"[memory] ingest warning: {exc}")
            finally:
                self._queue.task_done()

    async def flush(self) -> None:
        await self._queue.join()

    async def shutdown(self) -> None:
        if self._task is None:
            return
        await self._queue.put(None)
        await self._queue.join()
        await self._task
        self._task = None


class MemorySubscriber:
    """Consumes agent events and sends user/assistant text to embedding worker."""

    def __init__(self, ingestion_worker: EmbeddingIngestionWorker) -> None:
        self.ingestion_worker = ingestion_worker

    def on_event(self, event: Dict[str, Any]) -> None:
        try:
            if event.get("type") != "message_end":
                return
            message = event.get("message")
            if message is None:
                return
            role = str(getattr(message, "role", "")).strip().lower()
            if role not in {"user", "assistant"}:
                return
            text = "\n".join([x for x in _extract_texts(message) if x.strip()]).strip()
            if not text:
                return
            if role == "user":
                text = _extract_original_user_message(text)
            if text:
                self.ingestion_worker.enqueue(role, text)
        except Exception as exc:
            print(f"[memory] subscriber warning: {exc}")


class MemorySearchTool(AgentTool):
    name = "memory_search"
    description = "Semantic search over stored chat memory."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "top_k": {"type": "integer", "description": "Number of results"},
            "scope": {
                "type": "string",
                "description": "Memory scope: short, long, or both",
                "enum": ["short", "long", "both"],
            },
        },
        "required": ["query"],
    }
    label = "Memory Search"

    def __init__(self, retriever: MemoryRetriever) -> None:
        self.retriever = retriever

    async def execute(
        self,
        tool_call_id: str,
        params: Dict[str, Any],
        signal: Optional[Any] = None,
        on_update: Optional[Any] = None,
    ) -> AgentToolResult:
        del tool_call_id, signal, on_update
        query = str(params.get("query", "")).strip()
        if not query:
            return AgentToolResult(
                content=[TextContent(type="text", text="memory_search error: missing query")],
                details={"error": "missing query"},
            )
        try:
            top_k = int(params.get("top_k", 3) or 3)
        except Exception:
            top_k = 3
        top_k = max(1, top_k)
        scope = str(params.get("scope", "both")).strip().lower()
        if scope not in {"short", "long", "both"}:
            scope = "both"
        try:
            hits = self.retriever.retrieve(query=query, top_k=top_k, scope=scope)
        except Exception as exc:
            return AgentToolResult(
                content=[TextContent(type="text", text=f"memory_search error: {exc}")],
                details={"error": str(exc)},
            )
        if not hits:
            text = "No matching memories found."
        else:
            text = "Memory search results:\n" + "\n".join(f"{i + 1}. {h}" for i, h in enumerate(hits))
        return AgentToolResult(
            content=[TextContent(type="text", text=text)],
            details={"query": query, "top_k": top_k, "scope": scope, "count": len(hits)},
        )


class ToolsmakerTool(AgentTool):
    name = "toolsmaker"
    description = "Manage generated tools via create, approve, activate, reject, or list actions."
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "approve", "activate", "reject", "list"],
                "description": "Lifecycle action to run",
            },
            "intent": {
                "type": "object",
                "description": "Structured tool intent payload used by action=create",
            },
            "name": {
                "type": "string",
                "description": "Tool name used by approve/activate/reject",
            },
            "version": {
                "type": "integer",
                "description": "Tool version used by approve/activate/reject",
            },
            "reason": {
                "type": "string",
                "description": "Optional rejection reason for action=reject",
            },
            "max_output_chars": {
                "type": "integer",
                "description": "Optional output cap used by action=activate",
            },
        },
        "required": ["action"],
    }
    label = "Toolsmaker"

    def __init__(self, agent: Agent, allowed_capabilities: Sequence[str]) -> None:
        self.agent = agent
        self.allowed_capabilities = [str(x) for x in allowed_capabilities if x in TOOL_CAPABILITIES]

    @staticmethod
    def _error(text: str, details: Dict[str, Any]) -> AgentToolResult:
        return AgentToolResult(
            content=[TextContent(type="text", text=text)],
            details={"ok": False, **details},
        )

    @staticmethod
    def _ok(text: str, details: Dict[str, Any]) -> AgentToolResult:
        return AgentToolResult(
            content=[TextContent(type="text", text=text)],
            details={"ok": True, **details},
        )

    @staticmethod
    def _result_summary(status: str, name: str, version: int, review_path: str = "") -> str:
        lines = [f"tool={name}", f"version={version}", f"status={status}"]
        if review_path:
            lines.append(f"review_path={review_path}")
        return "\n".join(lines)

    @staticmethod
    def _parse_name_version(params: Dict[str, Any]) -> Tuple[str, int]:
        name = str(params.get("name", "")).strip()
        if not name:
            raise ValueError("missing required field: name")
        version_raw = params.get("version", None)
        if version_raw is None:
            raise ValueError("missing required field: version")
        try:
            version = int(version_raw)
        except Exception as exc:
            raise ValueError("version must be an integer") from exc
        if version <= 0:
            raise ValueError("version must be > 0")
        return name, version

    @staticmethod
    def _infer_required_capabilities(intent: Dict[str, Any]) -> List[str]:
        name_text = str(intent.get("name", "")).strip().lower()
        purpose_text = str(intent.get("purpose", "")).strip().lower()
        text = f"{name_text} {purpose_text}"
        required: List[str] = []
        if any(token in text for token in ["write", "writer", "save", "append"]):
            required.append("fs_write")
        if any(token in text for token in ["read", "reader", "load"]):
            required.append("fs_read")
        if any(token in text for token in ["http", "url", "fetch", "request", "download"]):
            required.append("http")
        if any(token in text for token in ["secret", "token", "password", "api key", "env"]):
            required.append("secrets")
        return sorted(set(required))

    def _validate_intent_contract(self, intent: Dict[str, Any]) -> Optional[AgentToolResult]:
        raw_caps = intent.get("capabilities", [])
        if raw_caps is None:
            raw_caps = []
        if not isinstance(raw_caps, list):
            return self._error(
                "toolsmaker create error: intent.capabilities must be an array.",
                {"action": "create", "error": "invalid_capabilities"},
            )

        requested = sorted({str(x).strip() for x in raw_caps if str(x).strip()})
        if not requested:
            inferred = self._infer_required_capabilities(intent)
            return self._error(
                "toolsmaker create blocked: intent.capabilities is empty; this would generate a no-op tool.",
                {
                    "action": "create",
                    "error": "missing_capabilities",
                    "requested": requested,
                    "inferred_required": inferred,
                    "hint": "Include capabilities such as fs_write/fs_read/http/secrets in the intent.",
                },
            )

        inferred_required = self._infer_required_capabilities(intent)
        missing = sorted(set(inferred_required) - set(requested))
        if missing:
            return self._error(
                "toolsmaker create blocked: intent likely needs capabilities that are missing.",
                {
                    "action": "create",
                    "error": "inferred_capabilities_missing",
                    "requested": requested,
                    "inferred_required": inferred_required,
                    "missing": missing,
                },
            )

        if "fs_write" in requested and not list(intent.get("allowed_paths") or []):
            return self._error(
                "toolsmaker create blocked: fs_write requires allowed_paths in intent.",
                {
                    "action": "create",
                    "error": "missing_allowed_paths",
                    "requested": requested,
                },
            )
        if "fs_read" in requested and not list(intent.get("allowed_paths") or []):
            return self._error(
                "toolsmaker create blocked: fs_read requires allowed_paths in intent.",
                {
                    "action": "create",
                    "error": "missing_allowed_paths",
                    "requested": requested,
                },
            )
        if "http" in requested and not list(intent.get("allowed_domains") or []):
            return self._error(
                "toolsmaker create blocked: http requires allowed_domains in intent.",
                {
                    "action": "create",
                    "error": "missing_allowed_domains",
                    "requested": requested,
                },
            )
        return None

    def _check_capability_guard(self, intent: Dict[str, Any]) -> Optional[AgentToolResult]:
        raw_caps = intent.get("capabilities", [])
        if raw_caps is None:
            raw_caps = []
        if not isinstance(raw_caps, list):
            return self._error(
                "toolsmaker create error: intent.capabilities must be an array.",
                {"action": "create", "error": "invalid capabilities"},
            )

        requested = sorted({str(x).strip() for x in raw_caps if str(x).strip()})
        allowed = sorted(set(self.allowed_capabilities))
        disallowed = sorted(set(requested) - set(allowed))
        if disallowed:
            return self._error(
                "toolsmaker create blocked: requested capabilities are not allowed by current runtime policy.",
                {
                    "action": "create",
                    "error": "capability_not_allowed",
                    "requested": requested,
                    "allowed": allowed,
                    "disallowed": disallowed,
                    "hint": "Set POP_AGENT_TOOLSMAKER_ALLOWED_CAPS to include the required capabilities.",
                },
            )
        return None

    def _handle_create(self, params: Dict[str, Any]) -> AgentToolResult:
        intent = params.get("intent", None)
        if not isinstance(intent, dict):
            return self._error(
                "toolsmaker create error: missing or invalid intent object.",
                {"action": "create", "error": "missing_intent"},
            )

        invalid = self._validate_intent_contract(intent)
        if invalid is not None:
            return invalid

        blocked = self._check_capability_guard(intent)
        if blocked is not None:
            return blocked

        result = self.agent.build_dynamic_tool_from_intent(intent)
        next_steps = "Next steps: approve this version, then activate it."
        text = "\n".join(
            [
                self._result_summary(
                    status=str(result.status),
                    name=result.spec.name,
                    version=result.spec.version,
                    review_path=result.review_path,
                ),
                next_steps,
            ]
        )
        return self._ok(
            text,
            {
                "action": "create",
                "status": result.status,
                "name": result.spec.name,
                "version": result.spec.version,
                "review_path": result.review_path,
                "spec_path": result.spec_path,
                "code_path": result.code_path,
                "validation": result.validation,
                "requested_capabilities": list(intent.get("capabilities") or []),
                "requested_allowed_paths": list(intent.get("allowed_paths") or []),
                "requested_allowed_domains": list(intent.get("allowed_domains") or []),
                "next_steps": ["approve", "activate"],
            },
        )

    def _handle_approve(self, params: Dict[str, Any]) -> AgentToolResult:
        name, version = self._parse_name_version(params)
        result = self.agent.approve_dynamic_tool(name=name, version=version)
        text = self._result_summary(status=str(result.status), name=result.spec.name, version=result.spec.version)
        return self._ok(
            text,
            {
                "action": "approve",
                "status": result.status,
                "name": result.spec.name,
                "version": result.spec.version,
                "review_path": result.review_path,
                "validation": result.validation,
            },
        )

    def _handle_activate(self, params: Dict[str, Any]) -> AgentToolResult:
        name, version = self._parse_name_version(params)
        try:
            max_output_chars = int(params.get("max_output_chars", 20_000) or 20_000)
        except Exception as exc:
            raise ValueError("max_output_chars must be an integer") from exc
        max_output_chars = max(1, max_output_chars)

        tool = self.agent.activate_tool_version(name=name, version=version, max_output_chars=max_output_chars)
        tools = self.agent.list_tools()
        text = "\n".join(
            [
                f"activated={tool.name}",
                f"version={version}",
                "available_tools=" + ", ".join(tools),
            ]
        )
        return self._ok(
            text,
            {
                "action": "activate",
                "status": "activated",
                "name": name,
                "version": version,
                "activated_tool": tool.name,
                "max_output_chars": max_output_chars,
                "tools": tools,
            },
        )

    def _handle_reject(self, params: Dict[str, Any]) -> AgentToolResult:
        name, version = self._parse_name_version(params)
        reason = str(params.get("reason", "rejected_by_reviewer")).strip() or "rejected_by_reviewer"
        result = self.agent.reject_dynamic_tool(name=name, version=version, reason=reason)
        text = self._result_summary(status=str(result.status), name=result.spec.name, version=result.spec.version)
        return self._ok(
            text,
            {
                "action": "reject",
                "status": result.status,
                "name": result.spec.name,
                "version": result.spec.version,
                "reason": reason,
            },
        )

    def _handle_list(self) -> AgentToolResult:
        tools = self.agent.list_tools()
        if tools:
            text = "tools:\n" + "\n".join(f"{i + 1}. {name}" for i, name in enumerate(tools))
        else:
            text = "tools: (none)"
        return self._ok(text, {"action": "list", "tools": tools, "count": len(tools)})

    async def execute(
        self,
        tool_call_id: str,
        params: Dict[str, Any],
        signal: Optional[Any] = None,
        on_update: Optional[Any] = None,
    ) -> AgentToolResult:
        del tool_call_id, signal, on_update
        action = str(params.get("action", "")).strip().lower()
        if action not in {"create", "approve", "activate", "reject", "list"}:
            return self._error(
                "toolsmaker error: action must be one of create|approve|activate|reject|list.",
                {"action": action or None, "error": "invalid_action"},
            )
        try:
            if action == "create":
                return self._handle_create(params)
            if action == "approve":
                return self._handle_approve(params)
            if action == "activate":
                return self._handle_activate(params)
            if action == "reject":
                return self._handle_reject(params)
            return self._handle_list()
        except Exception as exc:
            return self._error(
                f"toolsmaker {action} error: {exc}",
                {"action": action, "error": str(exc)},
            )


class ToolsmakerApprovalSubscriber:
    """Prompt the terminal user for tool approval after toolsmaker create calls."""

    def __init__(self, agent: Agent, auto_activate_default: bool = True) -> None:
        self.agent = agent
        self.auto_activate_default = auto_activate_default
        self._handled: set[Tuple[str, int]] = set()

    def _read_details(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if event.get("type") != "tool_execution_end":
            return None
        if str(event.get("toolName", "")).strip() != "toolsmaker":
            return None
        result = event.get("result")
        details = getattr(result, "details", None)
        if not isinstance(details, dict):
            return None
        if not bool(details.get("ok")):
            return None
        if str(details.get("action", "")).strip().lower() != "create":
            return None
        if str(details.get("status", "")).strip().lower() != "approval_required":
            return None
        return details

    def on_event(self, event: Dict[str, Any]) -> None:
        try:
            details = self._read_details(event)
            if details is None:
                return
            name = str(details.get("name", "")).strip()
            version = int(details.get("version", 0) or 0)
            if not name or version <= 0:
                return

            key = (name, version)
            if key in self._handled:
                return
            self._handled.add(key)

            review_path = str(details.get("review_path", "")).strip()
            print("\n[toolsmaker] Manual approval requested.")
            print(f"[toolsmaker] tool={name} version={version}")
            if review_path:
                print(f"[toolsmaker] review={review_path}")
            requested_capabilities = list(details.get("requested_capabilities") or [])
            if requested_capabilities:
                print(f"[toolsmaker] requested_capabilities={requested_capabilities}")
            else:
                print("[toolsmaker] requested_capabilities=(none)")

            decision = input("[toolsmaker] Approve this tool version? [y/N]: ").strip().lower()
            if decision in {"y", "yes"}:
                approved = self.agent.approve_dynamic_tool(name=name, version=version)
                print(f"[toolsmaker] approved status={approved.status}")

                if self.auto_activate_default:
                    activation_prompt = "[toolsmaker] Activate now? [Y/n]: "
                else:
                    activation_prompt = "[toolsmaker] Activate now? [y/N]: "
                activate_choice = input(activation_prompt).strip().lower()
                should_activate = activate_choice in {"y", "yes"} or (activate_choice == "" and self.auto_activate_default)
                if should_activate:
                    activated_tool = self.agent.activate_tool_version(name=name, version=version)
                    print(f"[toolsmaker] activated tool={activated_tool.name} version={version}")
                else:
                    print("[toolsmaker] activation skipped")
            else:
                reason = input("[toolsmaker] Reject reason (enter for default): ").strip() or "rejected_by_reviewer"
                rejected = self.agent.reject_dynamic_tool(name=name, version=version, reason=reason)
                print(f"[toolsmaker] rejected status={rejected.status} reason={reason}")
        except Exception as exc:
            print(f"[toolsmaker] manual approval warning: {exc}")


class BashExecApprovalPrompter:
    """Prompt the terminal user for medium/high risk bash_exec commands."""

    def __call__(self, request: Dict[str, Any]) -> bool:
        try:
            command = str(request.get("command", "")).strip()
            cwd = str(request.get("cwd", "")).strip()
            risk = str(request.get("risk", "")).strip() or "unknown"
            justification = str(request.get("justification", "")).strip()

            print("\n[bash_exec] Approval requested.")
            print(f"[bash_exec] risk={risk}")
            print(f"[bash_exec] cwd={cwd}")
            print(f"[bash_exec] command={command}")
            if justification:
                print(f"[bash_exec] justification={justification}")
            else:
                print("[bash_exec] justification=(none)")

            decision = input("[bash_exec] Allow this command? [y/N]: ").strip().lower()
            return decision in {"y", "yes"}
        except Exception as exc:
            print(f"[bash_exec] approval prompt warning: {exc}")
            return False


async def _read_input(prompt: str) -> str:
    return input(prompt)


async def main() -> None:
    agent = Agent({"stream_fn": stream})
    agent.set_model({"provider": "gemini", "id": "gemini-3-flash-preview", "api": None})
    agent.set_timeout(120)

    embedder = Embedder(use_api="openai")
    short_memory = ConversationMemory(embedder=embedder, max_entries=100)
    long_memory = DiskMemory(filepath=os.path.join("agent", "mem", "chat"), embedder=embedder, max_entries=1000)
    retriever = MemoryRetriever(short_term=short_memory, long_term=long_memory)

    ingestion_worker = EmbeddingIngestionWorker(memory=short_memory, long_term=long_memory)
    ingestion_worker.start()
    memory_subscriber = MemorySubscriber(ingestion_worker=ingestion_worker)

    memory_search_tool = MemorySearchTool(retriever=retriever)
    toolsmaker_caps = _parse_toolsmaker_allowed_capabilities(os.getenv("POP_AGENT_TOOLSMAKER_ALLOWED_CAPS"))
    toolsmaker_tool = ToolsmakerTool(agent=agent, allowed_capabilities=toolsmaker_caps)
    workspace_root = os.path.realpath(os.getcwd())
    bash_allowed_roots = _parse_path_list_env(
        "POP_AGENT_BASH_ALLOWED_ROOTS",
        default_paths=[workspace_root],
        base_dir=workspace_root,
    )
    bash_writable_roots = _parse_path_list_env(
        "POP_AGENT_BASH_WRITABLE_ROOTS",
        default_paths=[workspace_root],
        base_dir=workspace_root,
    )
    bash_timeout_s = _parse_float_env("POP_AGENT_BASH_TIMEOUT_S", 15.0)
    bash_max_output_chars = _parse_int_env("POP_AGENT_BASH_MAX_OUTPUT_CHARS", 20_000)
    bash_prompt_approval = _parse_bool_env("POP_AGENT_BASH_PROMPT_APPROVAL", True)
    bash_approval_fn = BashExecApprovalPrompter() if bash_prompt_approval else None
    bash_exec_tool = BashExecTool(
        BashExecConfig(
            project_root=workspace_root,
            allowed_roots=bash_allowed_roots,
            writable_roots=bash_writable_roots,
            read_commands=BASH_READ_COMMANDS,
            write_commands=BASH_WRITE_COMMANDS,
            git_read_subcommands=BASH_GIT_READ_SUBCOMMANDS,
            default_timeout_s=bash_timeout_s,
            max_timeout_s=60.0,
            default_max_output_chars=bash_max_output_chars,
            max_output_chars_limit=100_000,
        ),
        approval_fn=bash_approval_fn,
    )
    bash_read_csv = _sorted_csv(BASH_READ_COMMANDS)
    bash_write_csv = _sorted_csv(BASH_WRITE_COMMANDS)
    bash_git_csv = _sorted_csv(BASH_GIT_READ_SUBCOMMANDS)
    bash_exec_tool.description = (
        "Run one safe shell command without a shell. "
        f"Allowed read commands: {bash_read_csv}. "
        f"Allowed write commands: {bash_write_csv}. "
        f"For git, allowed subcommands: {bash_git_csv}. "
        "Write commands require approval."
    )
    agent.set_system_prompt(
        "You are a helpful assistant. "
        "Use tools when they improve accuracy or when the user asks for external actions. "
        "Prefer existing tools first, especially bash_exec for filesystem/shell inspection tasks. "
        "Use toolsmaker only when no existing tool can safely complete the request. "
        "When existing tools are insufficient, use the toolsmaker tool to create minimal-capability tools. "
        "Follow the lifecycle: create first, then approve, then activate. "
        "When calling toolsmaker create, always include intent.capabilities; "
        "fs_read/fs_write require allowed_paths and http requires allowed_domains. "
        f"Bash_exec allowlist (read): {bash_read_csv}. "
        f"Bash_exec allowlist (write): {bash_write_csv}. "
        f"Bash_exec git subcommands: {bash_git_csv}. "
        "Never call bash_exec with a command/subcommand outside those allowlists."
    )
    agent.set_tools([SlowTool(), FastTool(), WebSnapshotTool(), memory_search_tool, toolsmaker_tool, bash_exec_tool])

    log_level = os.getenv("POP_AGENT_LOG_LEVEL", "quiet")
    toolsmaker_manual_approval = _parse_bool_env("POP_AGENT_TOOLSMAKER_PROMPT_APPROVAL", True)
    toolsmaker_auto_activate = _parse_bool_env("POP_AGENT_TOOLSMAKER_AUTO_ACTIVATE", True)

    unsubscribe_log = agent.subscribe(make_event_logger(log_level))
    unsubscribe_memory = agent.subscribe(memory_subscriber.on_event)
    if toolsmaker_manual_approval:
        approval_subscriber = ToolsmakerApprovalSubscriber(
            agent=agent,
            auto_activate_default=toolsmaker_auto_activate,
        )
        unsubscribe_approval = agent.subscribe(approval_subscriber.on_event)
    else:
        unsubscribe_approval = lambda: None

    try:
        top_k = max(1, int(os.getenv("POP_AGENT_MEMORY_TOP_K", "3") or "3"))
    except Exception:
        top_k = 3

    print("POP Chatroom Agent (tools + embedding memory)")
    if toolsmaker_manual_approval:
        print(
            "[toolsmaker] manual approval prompts: on "
            f"(default auto-activate={'on' if toolsmaker_auto_activate else 'off'})"
        )
    else:
        print("[toolsmaker] manual approval prompts: off")
    if bash_prompt_approval:
        print("[bash_exec] approval prompts: on")
    else:
        print("[bash_exec] approval prompts: off (medium/high commands will be denied)")
    print("Type 'exit' or 'quit' to stop.\n")
    try:
        while True:
            try:
                user_message = (await _read_input("User: ")).strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting chat.")
                break

            if not user_message:
                continue
            if user_message.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break

            await ingestion_worker.flush()

            memory_text = "(no relevant memories)"
            try:
                short_hits, long_hits = retriever.retrieve_sections(user_message, top_k=top_k, scope="both")
                memory_text = _format_memory_sections(short_hits, long_hits)
            except Exception as exc:
                print(f"[memory] retrieval warning: {exc}")

            augmented_prompt = _build_augmented_prompt(user_message, memory_text)
            try:
                await agent.prompt(augmented_prompt)
            except Exception as exc:
                print(f"Assistant error: {exc}\n")
                continue

            reply = _extract_latest_assistant_text(agent)
            if not reply:
                reply = "(no assistant text returned)"
            print(f"Assistant: {reply}\n")
    finally:
        try:
            await ingestion_worker.shutdown()
        except Exception as exc:
            print(f"[memory] shutdown warning: {exc}")
        unsubscribe_memory()
        unsubscribe_log()
        unsubscribe_approval()


if __name__ == "__main__":
    asyncio.run(main())
