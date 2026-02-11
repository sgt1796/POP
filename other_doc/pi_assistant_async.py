"""
pi_assistant_async.py
======================

This module implements a simple personal assistant powered by the
`PromptFunction` abstraction from the POP library.  In addition to
generating responses via a large‑language model, the assistant
maintains an **evolving memory**.  Relevant past exchanges are
embedded via the `Embedder` class and retrieved using cosine
similarity whenever the assistant receives a new user message.  This
longer‑term memory helps the assistant provide consistent and
contextually aware replies over the course of a conversation.

The assistant is designed to run asynchronously.  Calls to the
synchronous `PromptFunction.execute` method are offloaded to a
background thread via `asyncio.to_thread`.  To experiment with the
assistant interactively, run this module as a script.

Example::

    python pi_assistant_async.py

This will start a REPL where you can chat with the assistant.  Type
"exit" or press Ctrl‑C to terminate the session.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from POP.Embedder import Embedder
from POP import PromptFunction


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the cosine similarity between two vectors.

    Parameters
    ----------
    a, b : np.ndarray
        1D vectors of the same dimension.

    Returns
    -------
    float
        The cosine similarity between `a` and `b` in the range [‑1, 1].

    Notes
    -----
    If either vector has zero norm, the similarity is defined to be 0 to
    avoid division by zero.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


@dataclass
class MemoryEntry:
    """A single memory entry consisting of text and its embedding."""

    text: str
    embedding: np.ndarray


class ConversationMemory:
    """Stores and retrieves conversation snippets using vector similarity.

    The memory keeps a chronological list of all user and assistant
    utterances.  Each entry is embedded upon insertion.  At query
    time, the memory computes cosine similarity between the query
    embedding and every stored embedding, returning the top N most
    similar snippets.

    Parameters
    ----------
    embedder : Embedder
        An instance of the POP `Embedder` class used to compute
        sentence embeddings.
    max_entries : int, optional
        Maximum number of memory entries to retain.  When the limit is
        exceeded, the oldest entries are dropped.  Defaults to 100.
    """

    def __init__(self, embedder: Embedder, max_entries: int = 100) -> None:
        self.embedder = embedder
        self.max_entries = max_entries
        self._entries: List[MemoryEntry] = []

    def add(self, text: str) -> None:
        """Add a new memory entry.

        The text is embedded immediately.  If the memory size exceeds
        `max_entries`, the oldest item is removed.

        Parameters
        ----------
        text : str
            The conversation snippet (e.g., "user: hello" or
            "assistant: hi there!") to store.
        """
        # Compute the embedding.  `get_embedding` expects a list of
        # strings and returns a 2D array of shape (n, dim).
        embedding = self.embedder.get_embedding([text])[0]
        self._entries.append(MemoryEntry(text, embedding))
        # Trim memory if it exceeds the maximum length.
        if len(self._entries) > self.max_entries:
            self._entries.pop(0)

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve the most relevant memory snippets for a query.

        Parameters
        ----------
        query : str
            The current user input for which to find similar past
            conversation snippets.
        top_k : int, optional
            Number of top snippets to return.  Defaults to 3.

        Returns
        -------
        List[str]
            A list of memory texts ordered by descending similarity.  If
            the memory is empty, an empty list is returned.
        """
        if not self._entries:
            return []
        # Embed the query once for efficiency.
        query_emb = self.embedder.get_embedding([query])[0]
        # Compute similarity scores for each stored entry.
        scores = [
            _cosine_similarity(query_emb, entry.embedding) for entry in self._entries
        ]
        # Get indices of the ``top_k`` highest scores in descending order.  The
        # slice ``[-top_k:]`` uses a plain ASCII hyphen.  Negative
        # indexing with a Unicode hyphen (U+2011) would produce a
        # SyntaxError.
        if top_k <= 0:
            top_k = 1
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [self._entries[i].text for i in top_indices]


class DiskMemory:
    """
    Persistent text+vector memory split across two files:
      - <base>.text.jsonl        (one {"text": "..."} per line)
      - <base>.embeddings.npy    (float32 array, shape = [N, D])

    Lines and rows are aligned by index: line i ↔ row i.
    """

    def __init__(self, filepath: str, embedder: Embedder, max_entries: int = 1000) -> None:
        self.base = os.path.splitext(filepath)[0] if filepath.endswith(".jsonl") else filepath
        self.text_path = f"{self.base}.text.jsonl"
        self.emb_path  = f"{self.base}.embeddings.npy"

        self.embedder = embedder
        self.max_entries = max_entries

        os.makedirs(os.path.dirname(self.text_path) or ".", exist_ok=True)

        # Lightweight line count cache
        self._n_text = self._count_lines(self.text_path)

    # ---------- public API ----------
    def add(self, text: str) -> None:
        """Append text to JSONL and its vector to .npy; prune if needed."""
        # 1) text
        with open(self.text_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        self._n_text += 1

        # 2) vector
        vec = self.embedder.get_embedding([text])[0].astype("float32")
        if os.path.exists(self.emb_path):
            M = np.load(self.emb_path, mmap_mode=None, allow_pickle=False)
            M = np.vstack([M, vec[None, :]])
        else:
            M = vec[None, :]
        np.save(self.emb_path, M)

        # 3) prune both synchronously if over limit
        self._prune()

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """Embed query, cosine-search over .npy, return matching text lines."""
        if not os.path.exists(self.emb_path) or self._n_text == 0:
            return []

        Q = self.embedder.get_embedding([query])[0].astype("float32")
        M = np.load(self.emb_path, mmap_mode="r")  # memory-map for big files

        # Normalize (safe against zeros)
        def _norm(x): 
            n = np.linalg.norm(x, axis=-1, keepdims=True)
            n[n == 0] = 1.0
            return x / n

        Qn = _norm(Q[None, :])[0]
        Mn = _norm(M)

        sims = (Mn @ Qn).astype("float32")
        k = max(1, min(top_k, sims.shape[0]))
        idx = np.argpartition(-sims, k-1)[:k]
        idx = idx[np.argsort(-sims[idx])]

        # Fetch those text lines by index
        return self._read_lines_by_index(idx.tolist())

    # ---------- helpers ----------
    def _read_lines_by_index(self, indices: List[int]) -> List[str]:
        """Read specific lines from JSONL by index (0-based)."""
        out = []
        wanted = set(indices)
        max_i = max(indices) if indices else -1
        with open(self.text_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i in wanted:
                    try:
                        out.append(json.loads(line)["text"])
                    except Exception:
                        out.append(line.strip())
                if i >= max_i and len(out) == len(indices):
                    break
        # Ensure order matches `indices`
        order = {v: i for i, v in enumerate(indices)}
        return [t for _, t in sorted(zip(indices, out), key=lambda p: order[p[0]])]

    def _count_lines(self, path: str) -> int:
        if not os.path.exists(path): return 0
        with open(path, "rb") as f:
            return sum(1 for _ in f)

    def _prune(self) -> None:
        """Keep only newest `max_entries` lines/rows across both files."""
        if self._n_text <= self.max_entries:
            return
        keep = self.max_entries

        # 1) prune text: keep last `keep` lines
        with open(self.text_path, "rb") as f:
            lines = f.readlines()[-keep:]
        with open(self.text_path, "wb") as f:
            f.writelines(lines)
        self._n_text = keep

        # 2) prune vectors: keep last `keep` rows
        if os.path.exists(self.emb_path):
            M = np.load(self.emb_path, mmap_mode=None, allow_pickle=False)
            if len(M) > keep:
                np.save(self.emb_path, M[-keep:])

class PIAssistant:
    """A personal assistant that uses PromptFunction and vector memory.

    This class wraps a `PromptFunction` instance and couples it with
    `ConversationMemory` to produce contextually grounded replies.  The
    assistant retrieves relevant memories based on the current user
    message and feeds them to the language model as additional
    context.

    Parameters
    ----------
    system_prompt : str
        The system prompt used to initialise the underlying
        `PromptFunction`.  This should describe the assistant's
        persona and high‑level behaviour.
    embedder_model : str, optional
        Name of the model to use for computing embeddings via
        `Embedder`.  When ``None``, the embedder defaults to the
        Jina or OpenAI API depending on environment variables.  See
        `Embedder` documentation for details.
    embedder_api : str, optional
        Which API to use for embedding (`"openai"`, `"jina"` or
        `None` for local embeddings).  Defaults to ``None``.  If
        using API‑based embeddings, ensure the relevant API keys are
        configured in your environment.
    client : str, optional
        LLM client identifier for `PromptFunction`, e.g., ``"openai"``.
        Defaults to ``"openai"``.
    max_memory : int, optional
        Maximum number of memory entries to keep.  Defaults to 100.
    """

    def __init__(
        self,
        system_prompt: str,
        embedder_model: Optional[str] = None,
        embedder_api: Optional[str] = None,
        client: str = "openai",
        max_memory: int = 100,
        memory_file: Optional[str] = None,
        show_memory: bool = False,
        long_term_max_entries: int = 1000,
    ) -> None:
        """Initialise the PI Assistant with both short‑ and long‑term memory.

        Additional parameters allow optional persistence of memory to a file
        and printing of retrieved memories.

        Parameters
        ----------
        system_prompt : str
            The system prompt describing the assistant's persona.
        embedder_model : str, optional
            Local model name for the embedder.  If ``None`` and
            ``embedder_api`` is also ``None``, defaults to ``"openai"``.
        embedder_api : str, optional
            API name for remote embeddings (``"openai"`` or ``"jina"``).
        client : str, optional
            LLM client identifier used by `PromptFunction`.  Defaults to
            ``"openai"``.
        max_memory : int, optional
            Maximum number of entries to keep in the short‑term memory.
        memory_file : str, optional
            Path to the JSON file used for long‑term memory storage.  If
            provided, long‑term memory will persist across sessions.
        show_memory : bool, optional
            Whether to print the retrieved memory snippets when
            generating a reply.  Defaults to ``False``.
        long_term_max_entries : int, optional
            Maximum number of entries to keep in the long‑term memory file.
        """
        # If neither a local model nor an API is specified, default to OpenAI
        if embedder_model is None and embedder_api is None:
            embedder_api = "openai"
        self.embedder = Embedder(model_name=embedder_model, use_api=embedder_api)
        # Short‑term memory for current session
        self.memory = ConversationMemory(self.embedder, max_entries=max_memory)
        # Optional long‑term memory persisted to disk
        self.long_term_memory: Optional[DiskMemory] = None
        if memory_file:
            # Ensure directory exists
            os.makedirs(os.path.dirname(memory_file) or ".", exist_ok=True)
            self.long_term_memory = DiskMemory(
                filepath=memory_file,
                embedder=self.embedder,
                max_entries=long_term_max_entries,
            )
        # Control whether to display retrieved memory
        self.show_memory = show_memory
        # Construct base prompt with placeholders
        base_prompt = (
            "You are a helpful personal assistant.  The user may ask you any"
            " question or request assistance with tasks.  Below is a list of"
            " relevant memories from previous interactions, followed by the"
            " user's current message.  Provide a concise, helpful response.\n\n"
            "Relevant memories (if any):\n<<<memory>>>\n\n"
            "User's message:\n<<<message>>>\n\n"
            "Assistant's response:"
        )
        self.prompt_fn = PromptFunction(sys_prompt=system_prompt, prompt=base_prompt, client=client)

    async def generate_reply(self, user_message: str, top_k: int = 3) -> str:
        """Generate an assistant reply given a user message.

        In addition to retrieving relevant snippets from the current
        session, this method also consults the long‑term memory (if
        configured).  Both sets of memories are concatenated and fed
        into the prompt.  After generating the response, the user
        message and assistant reply are stored in both memory stores.

        Parameters
        ----------
        user_message : str
            The current message from the user.
        top_k : int, optional
            Number of memory entries to retrieve from each memory store.
        """
        # Retrieve from short‑term memory
        relevant_short = self.memory.retrieve(user_message, top_k=top_k)
        # Retrieve from long‑term memory if present
        relevant_long: List[str] = []
        if self.long_term_memory is not None:
            relevant_long = self.long_term_memory.retrieve(user_message, top_k=top_k)
        # Construct memory sections with labels
        memory_sections: List[str] = []
        if relevant_short:
            memory_sections.append(
                "Short‑term memory:\n" + "\n".join(relevant_short)
            )
        if relevant_long:
            memory_sections.append(
                "Long‑term memory:\n" + "\n".join(relevant_long)
            )
        memory_text = "\n\n".join(memory_sections) if memory_sections else "(no relevant memories)"
        # Optionally print memory for debugging or inspection
        if self.show_memory and memory_sections:
            print("-- Relevant memory snippets --")
            for section in memory_sections:
                print(section)
                print()
        # Offload the synchronous LLM call to a background thread.  This
        # prevents blocking the event loop during the API request.
        response = await asyncio.to_thread(
            self.prompt_fn.execute,
            memory=memory_text,
            message=user_message,
        )
        # Update short‑term memory
        self.memory.add(f"user: {user_message}")
        self.memory.add(f"assistant: {response}")
        # Update long‑term memory if configured
        if self.long_term_memory is not None:
            self.long_term_memory.add(f"user: {user_message}")
            self.long_term_memory.add(f"[{self.prompt_fn.client.__class__.__name__}] assistant: {response}")
        return response

    async def chat_loop(self) -> None:
        """Start an interactive chat session with the assistant.

        This method runs a simple read‑eval‑print loop (REPL).  It
        prompts the user for input, generates an assistant reply, and
        displays it.  The loop terminates when the user types
        "exit" or "quit".  Keyboard interrupts are caught and
        gracefully exit the session.
        """
        print("Personal Assistant (with memory)\nType 'exit' to quit.\n")
        while True:
            try:
                user_message = input("User: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting chat.")
                break
            if user_message.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break
            if not user_message:
                continue
            reply = await self.generate_reply(user_message)
            print(f"Assistant: {reply}\n")


async def main() -> None:
    """Entry point when running the module as a script.

    A default system prompt is supplied.  Modify the prompt to
    customise the assistant's persona or behaviour.
    """
    default_system_prompt = (
        "You are PI Assistant, a polite and knowledgeable AI created to"
        " help with research, programming, and general queries.  Answer"
        " clearly and succinctly."
    )
    # Initialise with a persistent memory file.  The file will be
    # created if it does not exist.  Set `show_memory=True` to
    # display retrieved snippets during conversation.
    assistant = PIAssistant(
        system_prompt=default_system_prompt,
        memory_file="memory_store.jsonl",
        show_memory=False, # Change to True to see retrieved memories
        client="openai", # Change to your preferred LLM client
    )
    await assistant.chat_loop()


if __name__ == "__main__":  # pragma: no cover
    # When run directly, start the asynchronous chat loop.
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExited.")