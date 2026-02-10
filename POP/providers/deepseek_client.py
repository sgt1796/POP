"""Deepseek API client implementation.

This module provides a client for the Deepseek API using OpenAI's
Python client under the hood.  It mirrors the original POP
implementation but exists as its own provider module.
"""

from typing import List, Dict, Any
from os import getenv

from .llm_client import LLMClient


try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


class DeepseekClient(LLMClient):
    """Client for Deepseek chat completions."""

    def __init__(self, model = None) -> None:
        if OpenAI is None:
            raise ImportError(
                "openai package is not installed. Install it to use DeepseekClient."
            )
        # Use a custom base URL for Deepseek
        self.client = OpenAI(api_key=getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
        self.model_name = model

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> Any:
        request_payload: Dict[str, Any] = {
            "model": model,
            "messages": [],
            "temperature": temperature,
        }

        # Images are not currently supported by Deepseek
        images = kwargs.pop("images", None)
        if images:
            raise NotImplementedError("DeepseekClient does not support images yet.")

        # Build messages list
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            payload_msg: Dict[str, Any] = {"role": role, "content": content}
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                payload_msg["tool_calls"] = tool_calls
            tool_call_id = msg.get("tool_call_id") or msg.get("toolCallId")
            if tool_call_id:
                payload_msg["tool_call_id"] = tool_call_id
            request_payload["messages"].append(payload_msg)

        # Tools support (OpenAI‑style function calls)
        tools = kwargs.get("tools")
        if tools:
            request_payload["tools"] = tools
            request_payload["tool_choice"] = kwargs.get("tool_choice", "auto")

        try:
            response = self.client.chat.completions.create(**request_payload)
        except Exception as exc:
            raise RuntimeError(f"Deepseek chat_completion error: {exc}") from exc
        return response


__all__ = ["DeepseekClient"]
