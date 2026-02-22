import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from POP.embedder import Embedder

from .constants import USER_PROMPT_MARKER
from .message_utils import extract_original_user_message, extract_texts


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
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
        scores = [cosine_similarity(query_emb, entry.embedding) for entry in self._entries]
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


def format_memory_sections(short_hits: List[str], long_hits: List[str]) -> str:
    sections: List[str] = []
    if short_hits:
        sections.append("|Short-term memory|:\n" + "\n".join(f"- {x}" for x in short_hits))
    if long_hits:
        sections.append("|Long-term memory|:\n" + "\n".join(f"- {x}" for x in long_hits))
    if not sections:
        return "(no relevant memories)"
    return "\n\n".join(sections)


def build_augmented_prompt(user_message: str, memory_text: str) -> str:
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
            text = "\n".join([x for x in extract_texts(message) if x.strip()]).strip()
            if not text:
                return
            if role == "user":
                text = extract_original_user_message(text)
            if text:
                self.ingestion_worker.enqueue(role, text)
        except Exception as exc:
            print(f"[memory] subscriber warning: {exc}")
