"""Conversation context objects.

In order to build and track a conversation with an LLM, this module
defines simple data classes for system messages and user/assistant
messages.  A :class:`Context` holds a list of :class:`MessageBlock`
objects and optionally a system prompt and a list of tool
descriptions.  It can be converted into the message list expected by
``LLMClient.chat_completion``.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class MessageBlock:
    """Represents a single message from a user or assistant."""
    role: str
    content: str


@dataclass
class Context:
    """Collects conversation messages and metadata."""
    system: Optional[str] = None
    messages: List[MessageBlock] = field(default_factory=list)
    tools: Optional[List[Dict[str, Any]]] = None

    def append(self, role: str, content: str) -> None:
        """Append a message to the context."""
        self.messages.append(MessageBlock(role=role, content=content))

    def to_messages(self) -> List[Dict[str, Any]]:
        """Convert the context into a list of message dictionaries.

        The resulting list begins with a system message if one is set,
        followed by all appended messages in order.
        """
        output: List[Dict[str, Any]] = []
        if self.system:
            output.append({"role": "system", "content": self.system})
        for mb in self.messages:
            output.append({"role": mb.role, "content": mb.content})
        return output


__all__ = ["MessageBlock", "Context"]