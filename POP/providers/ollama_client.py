"""Ollama-compatible LLM client.

This client sends requests to an Ollama instance via its HTTP API.  It
implements the :class:`LLMClient` interface by translating chat
messages into the prompt format expected by Ollama’s `/api/generate`
endpoint.  The default model and base URL can be customised on
instantiation.
"""

from typing import List, Dict, Any
from os import getenv
import requests

from .llm_client import LLMClient


class OllamaClient(LLMClient):
    """Client for Ollama's generate endpoint."""

    def __init__(
        self,
        model: str = "llama3:latest",
        base_url: str = "http://localhost:11434",
        default_options: Dict[str, Any] | None = None,
        timeout: int = 300,
    ) -> None:
        self.model = model
        self.model_name = model
        self.base_url = base_url
        # Reasonable defaults for extraction and summarisation tasks
        self.default_options = default_options or {
            "num_ctx": 8192,
            "temperature": 0.02,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.05,
            "mirostat": 0,
        }
        self.timeout = timeout

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> Any:
        # Build the system and prompt strings for Ollama
        system_parts: List[str] = []
        user_assistant_lines: List[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_parts.append(content)
            elif role == "assistant":
                user_assistant_lines.append(f"[Assistant]: {content}")
            else:
                user_assistant_lines.append(f"[User]: {content}")
        system = "\n".join(system_parts) if system_parts else None
        prompt = "\n".join(user_assistant_lines)

        # Merge caller options with defaults.  Accept both a nested
        # ``ollama_options`` dict and top-level parameters (e.g. max_tokens)
        caller_opts = kwargs.pop("ollama_options", {}) or {}
        options = {**self.default_options, **caller_opts}
        # Keep ``temperature`` in sync with options if provided
        if temperature is not None:
            options["temperature"] = temperature

        payload: Dict[str, Any] = {
            "model": model or self.model,
            "prompt": prompt,
            "stream": False,
            "num_predict": kwargs.get("max_tokens", 1024),
            "options": options,
        }
        # Add system instruction separately
        if system:
            payload["system"] = system
        # Handle JSON schema formatting
        fmt = kwargs.get("response_format")
        if fmt:
            fmt = self._normalize_schema(fmt)
            payload["format"] = fmt
        # Optional stop sequences and keep-alive
        if "stop" in kwargs and kwargs["stop"]:
            payload["stop"] = kwargs["stop"]
        if "keep_alive" in kwargs and kwargs["keep_alive"]:
            payload["keep_alive"] = kwargs["keep_alive"]

        try:
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout)
            response.raise_for_status()
            response_json = response.json()
            content = response_json.get("response", "")
            usage = self._extract_usage(response_json)
        except Exception as exc:
            raise RuntimeError(f"OllamaClient error: {exc}") from exc
        return self._wrap_response(content, usage=usage)

    def _extract_usage(self, response_json: Dict[str, Any]) -> Dict[str, Any]:
        prompt_tokens = response_json.get("prompt_eval_count")
        completion_tokens = response_json.get("eval_count")
        total_tokens = response_json.get("total_tokens")
        try:
            prompt_tokens = int(prompt_tokens) if prompt_tokens is not None else None
        except (TypeError, ValueError):
            prompt_tokens = None
        try:
            completion_tokens = int(completion_tokens) if completion_tokens is not None else None
        except (TypeError, ValueError):
            completion_tokens = None
        try:
            total_tokens = int(total_tokens) if total_tokens is not None else None
        except (TypeError, ValueError):
            total_tokens = None
        if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    def _normalize_schema(self, fmt: Any) -> Any:
        import json, os
        if fmt is None:
            return None
        if isinstance(fmt, str):
            # Interpret as a file path or JSON string
            if os.path.exists(fmt):
                return json.load(open(fmt, "r", encoding="utf-8"))
            return json.loads(fmt)
        if isinstance(fmt, dict) and "schema" in fmt and isinstance(fmt["schema"], dict):
            return fmt["schema"]  # unwrap OpenAI‑style wrapper
        if isinstance(fmt, dict):
            return fmt
        raise TypeError("response_format must be a JSON schema dict, a JSON string, or a file path")

    def _wrap_response(self, content: str, usage: Dict[str, Any] | None = None) -> Any:
        class Message:
            def __init__(self, content: str) -> None:
                self.content = content
                self.tool_calls = None
        class Choice:
            def __init__(self, message: Message) -> None:
                self.message = message
        class Response:
            def __init__(self, content: str, usage: Dict[str, Any] | None) -> None:
                self.choices = [Choice(Message(content))]
                self.usage = usage
        return Response(content, usage)


__all__ = ["OllamaClient"]
