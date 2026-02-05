"""Streaming support for LLM responses.

This module defines a simple generator function that yields events
during a call to an LLM client.  Each event is represented by a
dictionary with an ``event`` key and optionally additional data.  The
sequence of events loosely follows the pattern used by the TypeScript
``pi‑ai`` project: ``start`` → ``text_start`` → ``text_delta`` →
``text_end`` → ``done``.

The generator hides the blocking nature of LLM calls behind an
iterator interface.  For providers that support true streaming
responses, you can extend this function to yield tokens as they
arrive.
"""

from typing import Generator, Dict, Any, Optional

from .context import Context
from .api_registry import get_client, get_default_model


def stream(
    provider: str,
    context: Context,
    model: Optional[str] = None,
    **kwargs: Any,
) -> Generator[Dict[str, Any], None, None]:
    """Yield events representing a streamed LLM response.

    Parameters
    ----------
    provider : str
        Identifier of the provider to use (e.g. ``"openai"``).
    context : Context
        Conversation context containing messages and optional system prompt.
    model : str, optional
        Model name; if not provided the provider's default is used.
    **kwargs : Any
        Additional options passed directly to the provider's
        ``chat_completion`` method (e.g. ``temperature``, ``tools``).
    Yields
    ------
    dict
        Streaming events with at least an ``event`` key.  Events are:
        ``start`` (begin request), ``text_start`` (before first token),
        ``text_delta`` (with ``value`` containing the full text for
        synchronous providers), ``text_end`` (after last token), and
        ``done`` (request completed).
    """
    client = get_client(provider)
    if client is None:
        raise ValueError(f"Unknown provider: {provider}")
    # Determine model: fallback to default models
    if model is None:
        model = get_default_model(provider)
    yield {"event": "start"}
    # Convert context into messages
    messages = context.to_messages()
    yield {"event": "text_start"}
    # Call the provider synchronously.  If providers support streaming,
    # you could instead iterate over a streaming response here.
    response = client.chat_completion(messages=messages, model=model, **kwargs)
    # Extract the content; OpenAI‑style responses put the text in
    # response.choices[0].message.content
    text = ""
    try:
        text = response.choices[0].message.content
    except Exception:
        # If the response is a simple string or dict, fall back
        text = str(response)
    # For synchronous providers we send the entire text in one delta
    yield {"event": "text_delta", "value": text}
    yield {"event": "text_end"}
    yield {"event": "done"}


__all__ = ["stream"]
