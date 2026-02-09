"""Gemini API client implementation.

This module provides a :class:`LLMClient` implementation for Google
Gemini models using the OpenAI SDK compatibility endpoint. It sends
OpenAI-style chat completion requests to the Gemini API by configuring
``base_url`` on the OpenAI client.

Set ``GEMINI_API_KEY`` in your environment before using this client.
"""

from typing import List, Dict, Any, Optional
from os import getenv
from pydantic import BaseModel

from .llm_client import LLMClient


try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class GeminiClient(LLMClient):
    """Client for Google's Gemini models via OpenAI compatibility."""

    def __init__(self, model: str = "gemini-2.5-flash", base_url: str = GEMINI_OPENAI_BASE_URL) -> None:
        if OpenAI is None:
            raise ImportError("openai package is not installed. Install it to use GeminiClient.")
        self.client = OpenAI(api_key=getenv("GEMINI_API_KEY"), base_url=base_url)
        self.model_name = model

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> Any:
        # Build the request payload according to OpenAI's API.
        model_name = model or self.model_name
        request_payload: Dict[str, Any] = {
            "model": model_name,
            "messages": [],
            "temperature": temperature,
        }

        # Optional image attachments
        images: Optional[List[str]] = kwargs.pop("images", None)

        # Construct messages list. If images are present they are
        # attached to user messages (OpenAI multi-content messages).
        # Other roles are passed verbatim.
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if images and role == "user":
                multi: List[Any] = [{"type": "text", "text": content}]
                for img in images:
                    if isinstance(img, str) and img.startswith("http"):
                        multi.append({"type": "image_url", "image_url": {"url": img}})
                    else:
                        multi.append(
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
                        )
                content = multi
            payload_msg: Dict[str, Any] = {"role": role, "content": content}
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                payload_msg["tool_calls"] = tool_calls
            tool_call_id = msg.get("tool_call_id") or msg.get("toolCallId")
            if tool_call_id:
                payload_msg["tool_call_id"] = tool_call_id
            request_payload["messages"].append(payload_msg)

        # Response format: can be a pydantic model or raw JSON schema
        fmt: Any = kwargs.pop("response_format", None)
        if fmt:
            if isinstance(fmt, BaseModel):
                request_payload["response_format"] = fmt
            else:
                # wrap raw schema into OpenAI's wrapper
                request_payload["response_format"] = {"type": "json_schema", "json_schema": fmt}

        # Tools support: list of function descriptors, convert to gemini format
        tools = kwargs.pop("tools", None)
        for tool in tools:
            tool.pop("type", None)  # remove 'type' if present to avoid conflicts

        request_payload["tools"] = [{"type": "function", "function": tool} for tool in tools]
        request_payload["tool_choice"] = kwargs.pop("tool_choice", "auto")
        # Avoid collisions for core fields
        kwargs.pop("model", None)
        kwargs.pop("messages", None)
        kwargs.pop("temperature", None)
        
        # remove the event signal callback from kwargs to avoid sending it to Gemini API
        signal = kwargs.pop("signal", None)
        if kwargs:
            request_payload.update(kwargs)
        
        #print(f"GeminiClient sending request: {request_payload}")  # Debug log of the request payload
        # Send the request. If something goes wrong, raise a runtime error.
        try:
            response = self.client.chat.completions.create(**request_payload)
        except Exception as exc:
            raise RuntimeError(f"Gemini chat_completion error: {exc}") from exc
        return response


__all__ = ["GeminiClient"]
