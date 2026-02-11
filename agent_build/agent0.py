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
from agent.tools import FastTool, SlowTool, WebSnapshotTool

# Logging helpers
LOG_LEVELS = {
    "quiet": 0,
    "messages": 1,
    "stream": 2,
    "debug": 3,
}

USER_PROMPT_MARKER = "Current user message:\n"


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


def make_event_logger(level: str = "quiet"):
    """Create an event logger function for agent events."""
    level_value = _resolve_log_level(level)

    def log(event: Dict[str, Any]) -> None:
        if level_value <= LOG_LEVELS["quiet"]:
            return
        etype = event.get("type")

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
        sections.append("Short-term memory:\n" + "\n".join(f"- {x}" for x in short_hits))
    if long_hits:
        sections.append("Long-term memory:\n" + "\n".join(f"- {x}" for x in long_hits))
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


async def _read_input(prompt: str) -> str:
    return input(prompt)


async def main() -> None:
    agent = Agent({"stream_fn": stream})
    agent.set_model({"provider": "openai", "id": "gpt-5-mini", "api": None})
    agent.set_timeout(120)
    agent.set_system_prompt(
        "You are a helpful assistant. "
        "Use tools when they improve accuracy or when the user asks for external actions."
    )

    embedder = Embedder(use_api="openai")
    short_memory = ConversationMemory(embedder=embedder, max_entries=100)
    long_memory = DiskMemory(filepath=os.path.join("agent", "mem", "chat"), embedder=embedder, max_entries=1000)
    retriever = MemoryRetriever(short_term=short_memory, long_term=long_memory)

    ingestion_worker = EmbeddingIngestionWorker(memory=short_memory, long_term=long_memory)
    ingestion_worker.start()
    memory_subscriber = MemorySubscriber(ingestion_worker=ingestion_worker)

    memory_search_tool = MemorySearchTool(retriever=retriever)
    agent.set_tools([SlowTool(), FastTool(), WebSnapshotTool(), memory_search_tool])

    log_level = os.getenv("POP_AGENT_LOG_LEVEL", "quiet")
    unsubscribe_log = agent.subscribe(make_event_logger(log_level))
    unsubscribe_memory = agent.subscribe(memory_subscriber.on_event)

    try:
        top_k = max(1, int(os.getenv("POP_AGENT_MEMORY_TOP_K", "3") or "3"))
    except Exception:
        top_k = 3

    print("POP Chatroom Agent (tools + embedding memory)")
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


if __name__ == "__main__":
    asyncio.run(main())
