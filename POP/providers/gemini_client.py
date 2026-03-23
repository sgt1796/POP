"""Gemini API client implementation.

This module provides a :class:`LLMClient` implementation for Google
Gemini models using the OpenAI SDK compatibility endpoint. It sends
OpenAI-style chat completion requests to the Gemini API by configuring
``base_url`` on the OpenAI client.

Set ``GEMINI_API_KEY`` in your environment before using this client.
"""

from copy import deepcopy
from os import getenv
from typing import Any, Dict, List, Optional
import warnings

from pydantic import BaseModel

from .llm_client import LLMClient


try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def _warn(message: str) -> None:
    warnings.warn(message, UserWarning, stacklevel=3)


class GeminiClient(LLMClient):
    """Client for Google's Gemini models via OpenAI compatibility."""

    provider_name = "gemini"

    def __init__(self, model: str = "gemini-2.5-flash", base_url: str = GEMINI_OPENAI_BASE_URL) -> None:
        if OpenAI is None:
            raise ImportError("openai package is not installed. Install it to use GeminiClient.")
        self.client = OpenAI(api_key=getenv("GEMINI_API_KEY"), base_url=base_url)
        self.model_name = model
        self.last_request_meta: Optional[Dict[str, Any]] = None

    @staticmethod
    def _normalize_tools(tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Normalize Gemini tool declarations to the OpenAI-compatible shape.

        Gemini callers in POP may send either standard OpenAI function tools or
        the agent loop's legacy flat shape, for example:
        ``{"type": "custom", "name": "...", "parameters": {...}}``.
        """
        normalized: List[Dict[str, Any]] = []
        for tool in tools or []:
            if not isinstance(tool, dict):
                continue

            function_decl: Optional[Dict[str, Any]] = None
            nested_function = tool.get("function")

            if isinstance(nested_function, dict):
                function_decl = deepcopy(nested_function)
            elif tool.get("name"):
                # Gemini compatibility must continue accepting POP's flat
                # tool shape, including legacy ``type: "custom"`` tools.
                function_decl = deepcopy(tool)
                function_decl.pop("type", None)
                function_decl.pop("function", None)

            if not function_decl or not function_decl.get("name"):
                continue

            normalized.append({"type": "function", "function": function_decl})
        return normalized

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
        raw_tools = kwargs.pop("tools", None)
        tools = self._normalize_tools(raw_tools)
        tool_choice = kwargs.pop("tool_choice", None)
        if raw_tools and not tools:
            raise RuntimeError(
                "Gemini tool configuration error: POP could not normalize any supplied tools. "
                "Supported shapes are OpenAI function tools or flat tool dicts with `name` "
                "and `parameters` such as POP's legacy `type: \"custom\"` schema."
            )
        if raw_tools and len(tools) < len(raw_tools):
            dropped_count = len(raw_tools) - len(tools)
            _warn(
                f"POP ignored {dropped_count} Gemini tool declaration(s) it could not normalize."
            )
        if tools:
            request_payload["tools"] = tools
            request_payload["tool_choice"] = "auto" if tool_choice is None else tool_choice
        elif tool_choice is not None:
            _warn("POP ignored `tool_choice` because no Gemini tools were supplied.")
        # Avoid collisions for core fields
        kwargs.pop("model", None)
        kwargs.pop("messages", None)
        kwargs.pop("temperature", None)
        
        # remove the event signal callback from kwargs to avoid sending it to Gemini API
        signal = kwargs.pop("signal", None)
        if kwargs:
            request_payload.update(kwargs)
        self.last_request_meta = {
            "messages": request_payload["messages"],
            "tools": request_payload.get("tools"),
            "response_format": request_payload.get("response_format"),
        }
        
        #print(f"GeminiClient sending request: {request_payload}")  # Debug log of the request payload
        # Send the request. If something goes wrong, raise a runtime error.
        try:
            response = self.client.chat.completions.create(**request_payload)
        except Exception as exc:
            raise RuntimeError(f"Gemini chat_completion error: {exc}") from exc
        return response


__all__ = ["GeminiClient"]
