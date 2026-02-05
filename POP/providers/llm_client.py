"""Abstract interface for LLM clients.

This module defines the :class:`LLMClient` abstract base class used by
provider implementations.  Each provider must implement a
``chat_completion`` method that accepts a list of message dictionaries
([{role: str, content: str}]), a ``model`` name, a ``temperature``, and
any additional keyword arguments.  The method should return an object
with a ``choices`` attribute similar to OpenAI's response schema where
``choices[0].message.content`` contains the generated text.  If the
provider supports tool calling it should also populate
``choices[0].message.tool_calls`` accordingly.

Original POP placed all provider implementations in a single file.
This module holds only the abstract base class; see the
``pop/providers`` subpackage for concrete clients.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> Any:
        """Generate a chat completion.

        Parameters
        ----------
        messages : list of dict
            Messages in the conversation.  Each dict must have a
            ``role`` (e.g. ``"system"``, ``"user"``, ``"assistant"``) and
            ``content`` (the text of the message).  Implementations may
            support additional keys (for example a list of
            ``images``) but should document this behaviour.
        model : str
            Identifier for the model variant (e.g. ``"gpt-3.5-turbo"`` or
            provider‑specific names).
        temperature : float, optional
            Controls the randomness of the output.  The default of 0.0
            yields deterministic output for many providers.
        **kwargs : Any
            Provider‑specific options such as ``tools``, ``response_format``
            or image attachments.
        Returns
        -------
        Any
            A provider‑specific response object.  Consumers should not
            rely on the exact type; instead they should access
            ``response.choices[0].message.content`` for the text and
            ``response.choices[0].message.tool_calls`` if present.
        """
        raise NotImplementedError