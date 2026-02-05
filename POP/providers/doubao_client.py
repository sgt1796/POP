"""Doubao (Volcengine Ark) API client implementation.

This client wraps the Doubao chat completion endpoint.  It requires
the ``openai`` package because Doubao is compatible with the OpenAI
client library when configured with a custom base URL.  If the
package is missing, instantiating :class:`DoubaoClient` will raise an
error.
"""

from typing import List, Dict, Any
from os import getenv

from .llm_client import LLMClient


try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


class DoubaoClient(LLMClient):
    """Client for Doubao (Volcengine Ark) LLMs."""

    def __init__(self, model = None) -> None:
        if OpenAI is None:
            raise ImportError(
                "openai package is not installed. Install it to use DoubaoClient."
            )
        # Use the custom base URL for Doubao.  Requires DOUBAO_API_KEY.
        self.client = OpenAI(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=getenv("DOUBAO_API_KEY"),
        )
        self.model_name = model

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> Any:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [],
            "temperature": temperature,
        }

        # Normalize images input into a list
        images = kwargs.pop("images", None)
        if images and not isinstance(images, list):
            images = [images]

        # Copy through common generation options supported by Doubao
        passthrough = [
            "top_p",
            "max_tokens",
            "stop",
            "frequency_penalty",
            "presence_penalty",
            "logprobs",
            "top_logprobs",
            "logit_bias",
            "service_tier",
            "thinking",
            "stream",
            "stream_options",
        ]
        for key in passthrough:
            if key in kwargs and kwargs[key] is not None:
                payload[key] = kwargs[key]

        # Build messages, attaching images to user messages
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
            payload["messages"].append({"role": role, "content": content})

        # Function tools
        tools = kwargs.get("tools")
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = kwargs.get("tool_choice", "auto")

        try:
            response = self.client.chat.completions.create(**payload)
        except Exception as exc:
            raise RuntimeError(f"Doubao chat_completion error: {exc}") from exc
        return response


__all__ = ["DoubaoClient"]
