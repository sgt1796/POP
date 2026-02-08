"""OpenAI client implementation.

This module provides a concrete :class:`LLMClient` for interacting with
OpenAI's chat completion endpoint.  It mirrors the behaviour of the
original POP ``OpenAIClient`` but lives in its own file as part of a
modular provider architecture.  If the ``openai`` package is not
installed in your environment this client will raise an
ImportError on instantiation.
"""

from typing import List, Dict, Any, Optional
from os import getenv
from pydantic import BaseModel

from .llm_client import LLMClient


try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


class OpenAIClient(LLMClient):
    """Client for the OpenAI Chat Completions API."""

    def __init__(self, model: str = "gpt-5-nano") -> None:
        if OpenAI is None:
            raise ImportError(
                "openai package is not installed. Install it to use OpenAIClient."
            )
        # The API key is pulled from the environment.  You must set
        # OPENAI_API_KEY before using this client.
        self.client = OpenAI(api_key=getenv("OPENAI_API_KEY"))
        self.model_name = model

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 1.0,
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

        # Construct messages list.  If images are present they are
        # attached to the last user message (OpenAI multi‑content
        # messages).  Other roles are passed verbatim.
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if images and role == "user":
                multi: List[Any] = [{"type": "text", "text": content}]
                for img in images:
                    if isinstance(img, str) and img.startswith("http"):
                        multi.append({"type": "image_url", "image_url": {"url": img}})
                    else:
                        multi.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})
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
        fmt: Any = kwargs.get("response_format")
        if fmt:
            if isinstance(fmt, BaseModel):
                request_payload["response_format"] = fmt
            else:
                # wrap raw schema into OpenAI's wrapper
                request_payload["response_format"] = {"type": "json_schema", "json_schema": fmt}

        # Tools support: list of function descriptors
        tools = kwargs.get("tools")
        if tools:
            request_payload["tools"] = tools
            request_payload["tool_choice"] = kwargs.get("tool_choice", "auto")

        # Temporary patch for models not supporting a system role (as in the
        # original POP).  Convert the first system message to a user
        # message when using the "o1-mini" model.
        if model_name == "o1-mini" and request_payload["messages"] and request_payload["messages"][0]["role"] == "system":
            request_payload["messages"][0]["role"] = "user"

        # Send the request.  If something goes wrong, raise a runtime error.
        try:
            response = self.client.chat.completions.create(**request_payload)
        except Exception as exc:
            raise RuntimeError(f"OpenAI chat_completion error: {exc}") from exc
        return response


__all__ = ["OpenAIClient"]
