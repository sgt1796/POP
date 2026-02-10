"""Type definitions for Pop Agent.

This module contains dataclasses and type aliases used throughout the
agent implementation.  They mirror the structures defined in the
TypeScript version of the `pi‑agent` package but are simplified to
work idiomatically in Python.

The emphasis here is on clarity rather than strict type checking.
All classes are normal Python dataclasses with optional fields.  If
you need additional attributes for your own application you can
subclass these dataclasses or use ``typing.TypedDict`` for more
precise typing.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Union


###############################################################################
# Message content types
###############################################################################


@dataclass
class TextContent:
    """Represents a chunk of plain text returned by an LLM.

    The ``text_signature`` field is optional and may be supplied by
    certain providers to allow streaming clients to verify the
    integrity of partial responses.
    """

    type: str = "text"
    text: str = ""
    text_signature: Optional[str] = None


@dataclass
class ImageContent:
    """Represents an image returned or sent to the LLM.

    Currently only the MIME type and raw bytes are stored.  You can
    extend this class to include additional metadata (e.g. width and
    height) if required by your application.
    """

    type: str = "image"
    data: bytes = b""
    mime_type: str = "image/png"


@dataclass
class ThinkingContent:
    """Represents internal reasoning content returned by an LLM.

    ``thinking`` is an opaque string that represents the model's chain
    of thought.  The ``thinking_signature`` field mirrors the
    behaviour of providers that sign partial reasoning streams.
    """

    type: str = "thinking"
    thinking: str = ""
    thinking_signature: Optional[str] = None


@dataclass
class ToolCallContent:
    """Represents a tool call requested by the assistant.

    A tool call instructs the agent to invoke a named tool with
    structured arguments.  During streaming the ``partial_json`` field
    may be used to accumulate a JSON payload; once complete it is
    parsed into the ``arguments`` dictionary.  ``extra_content`` stores
    provider-specific metadata (e.g., Gemini thought signatures for
    OpenAI compatibility).
    """

    type: str = "toolCall"
    id: str = ""
    name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    partial_json: Optional[str] = None
    extra_content: Optional[Dict[str, Any]] = None


# Union of all content item types
ContentItem = Union[TextContent, ImageContent, ThinkingContent, ToolCallContent]


###############################################################################
# Agent message definitions
###############################################################################


@dataclass
class AgentMessage:
    """A message exchanged between the user, assistant or a tool.

    The ``role`` field identifies the origin of the message.  For
    assistant messages additional metadata such as the model ID and API
    provider can be attached.  Tool results include the name and
    identifier of the originating tool call.  Arbitrary extra
    attributes are supported via ``extras``; this allows custom
    applications to attach their own fields without needing to
    subclass the dataclass.
    """

    role: str
    content: List[ContentItem]
    timestamp: float
    # Assistant specific fields
    api: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    stop_reason: Optional[str] = None
    error_message: Optional[str] = None
    # Tool result specific fields
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    is_error: Optional[bool] = None
    # Catch‑all for any other attributes
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation of the message.

        This helper method flattens the dataclass fields into a plain
        dictionary which can be serialised to JSON or passed to
        ``POP.stream_simple``.  All dataclass fields are
        included except for those that are ``None``.  The content
        items are themselves converted to dictionaries using their
        ``__dict__`` attribute.
        """

        data: Dict[str, Any] = {
            "role": self.role,
            "content": [dataclasses.asdict(item) for item in self.content],
            "timestamp": self.timestamp,
        }
        # Optional fields
        if self.api is not None:
            data["api"] = self.api
        if self.provider is not None:
            data["provider"] = self.provider
        if self.model is not None:
            data["model"] = self.model
        if self.usage is not None:
            data["usage"] = self.usage
        if self.stop_reason is not None:
            data["stopReason"] = self.stop_reason
        if self.error_message is not None:
            data["errorMessage"] = self.error_message
        if self.tool_call_id is not None:
            data["toolCallId"] = self.tool_call_id
        if self.tool_name is not None:
            data["toolName"] = self.tool_name
        if self.details is not None:
            data["details"] = self.details
        if self.is_error is not None:
            data["isError"] = self.is_error
        # Merge extras last so that user defined keys override defaults
        data.update(self.extras)
        return data


###############################################################################
# Agent tools
###############################################################################


class AgentTool(Protocol):
    """Protocol for tools that the agent can execute.

    Tools extend this protocol by implementing an async ``execute``
    method.  The base fields ``name``, ``description``, ``parameters``
    and ``label`` describe the tool for the benefit of the LLM.  The
    ``parameters`` attribute should follow the JSON Schema format
    expected by `POP` so that the LLM knows how to call the
    tool.  The return value of :meth:`execute` must be a
    :class:`AgentToolResult` instance describing the output.
    """

    # Name of the tool as referenced by the assistant
    name: str
    # Human readable description of what the tool does
    description: str
    # JSON Schema describing accepted parameters
    parameters: Dict[str, Any]
    # Display name shown in UIs
    label: str

    async def execute(
        self,
        tool_call_id: str,
        params: Dict[str, Any],
        signal: Optional[Any] = None,
        on_update: Optional[Callable[["AgentToolResult"], None]] = None,
    ) -> "AgentToolResult":
        """Execute the tool and return the result.

        Parameters
        ----------
        tool_call_id : str
            The identifier assigned by the assistant to this tool call.
        params : dict
            Structured arguments parsed from the assistant request.
        signal : object, optional
            Optional cancellation signal; if provided the tool should
            stop execution when the signal is set.
        on_update : callable, optional
            Callback to stream partial results back to the agent.  The
            callback receives a partial :class:`AgentToolResult` and
            should return nothing.

        Returns
        -------
        AgentToolResult
            The final tool result.
        """

        ...  # pragma: no cover


@dataclass
class AgentToolResult:
    """Represents the outcome of a tool execution.

    Tools return a list of content items and an arbitrary details
    object.  The details are intended for UI consumption; they are
    available on the corresponding :class:`AgentMessage` under the
    ``details`` field.
    """

    content: List[ContentItem]
    details: Dict[str, Any]


###############################################################################
# Agent context and events
###############################################################################


@dataclass
class AgentContext:
    """Encapsulates the current state passed to the LLM transport layer."""

    system_prompt: str
    messages: List[AgentMessage]
    tools: Optional[List[AgentTool]] = None


# Thinking level definitions.  These are taken from the original
# JavaScript implementation and may be ignored by providers that do
# not support explicit reasoning control.
from typing import Literal

# Thinking level definitions.  These are taken from the original
# JavaScript implementation and may be ignored by providers that do
# not support explicit reasoning control.  The correct syntax for
# specifying literal types uses square brackets rather than call
# syntax.  See https://docs.python.org/3/library/typing.html#typing.Literal
ThinkingLevel = Union[
    Literal["off"],
    Literal["minimal"],
    Literal["low"],
    Literal["medium"],
    Literal["high"],
    Literal["xhigh"],
]


# Agent state used internally by the Agent class
@dataclass
class AgentState:
    system_prompt: str
    model: Dict[str, Any]
    thinking_level: str
    tools: List[AgentTool]
    messages: List[AgentMessage]
    is_streaming: bool
    stream_message: Optional[AgentMessage]
    pending_tool_calls: set[str]
    error: Optional[str] = None


# Agent event definitions.  Events are simple dictionaries with a
# ``type`` field and additional attributes depending on the event.  For
# convenience we define a type alias here; see :mod:`agent_loop` for
# actual event generation.
@dataclass
class AgentEvent:
    type: str
    # Additional fields depend on the event type; we allow arbitrary
    # extra attributes for flexibility.
    extras: Dict[str, Any] = field(default_factory=dict)
