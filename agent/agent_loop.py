"""Core event‑driven agent loop.

This module contains the low level functions that orchestrate
conversation turns, call the underlying language model, invoke
tools and emit events.  It is a fairly direct translation of the
TypeScript implementation found in the original `pi‑agent`
repository but adapted to idiomatic Python and asyncio.

The principal entry points are :func:`agent_loop` and
:func:`agent_loop_continue`.  Both return an
:class:`pop_agent.event_stream.EventStream` instance which
produces dictionaries describing the lifecycle of an agent run.
Consumers can iterate over the stream asynchronously to update
their UI or internal state as events arrive.

The loop functions accept a configuration object which contains
callbacks for converting agent messages into the LLM format,
optionally transforming the context window, resolving API keys
dynamically, and retrieving queued steering or follow‑up messages.

Note that no concrete LLM provider is hard coded into this
module.  Instead a `stream_fn` function must be supplied (or
implicitly imported from `POP`) which accepts a model
definition, an LLM context and a dictionary of options, and
returns an asynchronous stream of events from the provider.  See
the :mod:`pop_agent.proxy` module for an example of such a
function which proxies requests through an external server.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple, Union
from dotenv import load_dotenv

from .event_stream import EventStream
from .agent_types import (
    AgentContext,
    AgentMessage,
    AgentTool,
    AgentToolResult,
    AgentEvent,
    TextContent,
    ImageContent,
    ThinkingContent,
    ToolCallContent,
)

# Attempt to import POP for the default LLM transport.  If
# unavailable the user must supply their own `stream_fn`.
try:
    import POP  # type: ignore
except ImportError:
    POP = None


###############################################################################
# Configuration objects
###############################################################################

@dataclass
class AgentLoopConfig:
    """Configuration for the agent loop.

    The fields mirror those in the TypeScript implementation but are
    adapted to Python.  See the documentation of the original
    project for semantics.

    Parameters
    ----------
    model : dict
        Definition of the LLM model to call.  At a minimum this
        should include ``api``, ``provider`` and ``id`` keys.  The
        model object is passed directly to the LLM transport
        function.
    convert_to_llm : callable
        A coroutine or regular function that accepts a list of
        :class:`AgentMessage` instances and returns a list of
        dictionaries compatible with the underlying LLM API.  Only
        messages that the LLM understands should be returned; others
        must be filtered out.
    transform_context : callable, optional
        An optional coroutine that receives the current list of
        messages before each LLM call.  It can prune, reorder or
        otherwise modify the context window.  If omitted the
        messages are passed through unchanged.
    get_api_key : callable, optional
        Optional callback invoked before each LLM call to obtain a
        short lived API key.  This is useful when tokens expire
        during long running tool execution.  The callback receives
        the provider string from the model and returns a string or
        ``None``.
    get_steering_messages : callable, optional
        Called after each tool execution and before each LLM call to
        check if the user has queued steering messages.  Returns a
        list of :class:`AgentMessage` instances to prepend to the
        context.  If the list is non empty remaining tool calls in
        the current turn are skipped.
    get_follow_up_messages : callable, optional
        Called when the agent would otherwise finish (no more tool
        calls and no steering messages).  Returns messages to be
        appended to the context before starting another turn.  If
        empty the loop terminates.
    reasoning : str, optional
        Reasoning level to request from the LLM.  Only relevant
        models honour this flag.  Use ``None`` to disable explicit
        reasoning control.
    session_id : str, optional
        Session identifier forwarded to the LLM provider.  Allows
        providers to perform caching of previous turns.  Set to
        ``None`` to let the provider decide.
    thinking_budgets : dict, optional
        Custom token budgets for different reasoning levels.  Passed
        through unchanged to the LLM transport function.
    max_retry_delay_ms : int, optional
        Maximum delay in milliseconds to honour when the server
        requests a backoff.  If a larger value is requested the
        request is aborted and an error event is emitted.
    request_timeout_s : float, optional
        Overall timeout in seconds for an LLM request.  Set to
        ``None`` or a non-positive value to disable the timeout.
    api_key : str, optional
        Static API key forwarded to the LLM provider.  If provided
        this overrides any key returned from :attr:`get_api_key`.
    other_options : dict
        Additional options forwarded to the LLM transport.  Typical
        entries are ``temperature``, ``max_tokens`` and ``reasoning``.
    """

    model: Dict[str, Any]
    convert_to_llm: Callable[[List[AgentMessage]], Awaitable[List[dict]]]
    transform_context: Optional[Callable[[List[AgentMessage], Optional[asyncio.Event]], Awaitable[List[AgentMessage]]]] = None
    get_api_key: Optional[Callable[[str], Awaitable[Optional[str]]]] = None
    get_steering_messages: Optional[Callable[[], Awaitable[List[AgentMessage]]]] = None
    get_follow_up_messages: Optional[Callable[[], Awaitable[List[AgentMessage]]]] = None
    reasoning: Optional[str] = None
    session_id: Optional[str] = None
    thinking_budgets: Optional[Dict[str, Any]] = None
    max_retry_delay_ms: Optional[int] = None
    request_timeout_s: Optional[float] = 120.0
    api_key: Optional[str] = None
    other_options: Dict[str, Any] = field(default_factory=dict)

# Try loading environment variables from a .env file, if present.
load_dotenv()

# Type alias for the LLM transport function
StreamFn = Callable[[Dict[str, Any], Dict[str, Any], Dict[str, Any]], Awaitable[Any]]

DEFAULT_STREAM_TIMEOUT_S = 120.0


def _coerce_timeout(value: Optional[float]) -> Optional[float]:
    """Return a positive timeout in seconds or None if disabled/invalid."""
    if value is None:
        return None
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        return None
    if timeout <= 0:
        return None
    return timeout


def _build_error_message(exc: Exception, model: Dict[str, Any], aborted: bool = False) -> AgentMessage:
    """Create an assistant error message from an exception."""
    return AgentMessage(
        role="assistant",
        content=[TextContent(type="text", text="")],
        timestamp=time.time(),
        api=model.get("api") if isinstance(model, dict) else None,
        provider=model.get("provider") if isinstance(model, dict) else None,
        model=model.get("id") if isinstance(model, dict) else None,
        usage={},
        stop_reason="aborted" if aborted else "error",
        error_message=str(exc),
    )


###############################################################################
# Helper functions
###############################################################################

def _validate_tool_arguments(tool: AgentTool, tool_call: ToolCallContent) -> Dict[str, Any]:
    """Validate the arguments supplied by a tool call.

    The default implementation simply returns the arguments as provided by
    the assistant without performing any schema validation.  If you
    require strict adherence to a JSON Schema you may implement your
    own validation routine here.
    """
    # In the TypeScript version this uses validateToolArguments from
    # `pi‑ai` to ensure required parameters are present.  Here we
    # assume the assistant sends well formed input or that the tool
    # itself will validate.
    return dict(tool_call.arguments or {})


def _dict_to_agent_message(msg: Dict[str, Any]) -> AgentMessage:
    """Convert a dictionary representing an assistant message into an AgentMessage.

    The underlying LLM transport returns message objects as plain
    dictionaries.  This helper constructs a new :class:`AgentMessage`
    instance, converting content items into the appropriate dataclass
    types.  Unknown fields are stored in the ``extras`` attribute to
    preserve any provider specific metadata.
    """
    role = msg.get("role") or "assistant"
    timestamp = msg.get("timestamp", time.time())
    content: List[Union[TextContent, ImageContent, ThinkingContent, ToolCallContent]] = []
    for item in msg.get("content", []):
        # Determine content type and instantiate the corresponding dataclass
        t = item.get("type")
        if t == "text":
            content.append(TextContent(type="text", text=item.get("text", ""), text_signature=item.get("textSignature")))
        elif t == "image":
            content.append(ImageContent(type="image", data=item.get("data", b""), mime_type=item.get("mimeType", "image/png")))
        elif t == "thinking":
            content.append(ThinkingContent(type="thinking", thinking=item.get("thinking", ""), thinking_signature=item.get("thinkingSignature")))
        elif t == "toolCall":
            # During streaming a partialJson field may be present; we ignore it here
            content.append(
                ToolCallContent(
                    type="toolCall",
                    id=item.get("id", ""),
                    name=item.get("name", ""),
                    arguments=item.get("arguments", {}),
                    partial_json=item.get("partialJson"),
                )
            )
        else:
            # Unknown content type; treat as plain text
            content.append(TextContent(type=t or "text", text=str(item)))

    # Extract known fields
    api = msg.get("api")
    provider = msg.get("provider")
    model_id = msg.get("model")
    usage = msg.get("usage")
    stop_reason = msg.get("stopReason") or msg.get("stop_reason")
    error_message = msg.get("errorMessage") or msg.get("error_message")

    # Additional metadata not captured above is stored in extras
    extras: Dict[str, Any] = {}
    for k, v in msg.items():
        if k not in {
            "role",
            "content",
            "timestamp",
            "api",
            "provider",
            "model",
            "usage",
            "stopReason",
            "stop_reason",
            "errorMessage",
            "error_message",
        }:
            extras[k] = v

    return AgentMessage(
        role=role,
        content=content,
        timestamp=timestamp,
        api=api,
        provider=provider,
        model=model_id,
        usage=usage,
        stop_reason=stop_reason,
        error_message=error_message,
        extras=extras,
    )


def _copy_message(msg: AgentMessage) -> AgentMessage:
    """Create a shallow copy of an AgentMessage for event emission.

    When pushing messages onto the event stream we do not want
    downstream consumers to mutate the message stored in the agent
    context.  This helper constructs a new instance with the same
    field values.  ``content`` is not deep copied to keep the
    overhead low; if you mutate nested structures you should
    duplicate them yourself.
    """
    return AgentMessage(
        role=msg.role,
        content=list(msg.content),
        timestamp=msg.timestamp,
        api=msg.api,
        provider=msg.provider,
        model=msg.model,
        usage=msg.usage,
        stop_reason=msg.stop_reason,
        error_message=msg.error_message,
        tool_call_id=msg.tool_call_id,
        tool_name=msg.tool_name,
        details=msg.details,
        is_error=msg.is_error,
        extras=dict(msg.extras),
    )


###############################################################################
# Main loop functions
###############################################################################

def agent_loop(
    prompts: Sequence[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    signal: Optional[asyncio.Event] = None,
    stream_fn: Optional[StreamFn] = None,
) -> EventStream[AgentEvent, List[AgentMessage]]:
    """Start an agent loop with a new prompt message.

    The provided prompt messages are appended to the existing context
    before sending a request to the LLM.  The returned event stream
    yields lifecycle events for the entire run.  When the `agent_end`
    event is encountered the stream closes and the result is the list
    of all new messages produced during the run.

    This function schedules the loop on the current asyncio event
    loop and returns immediately.

    Parameters
    ----------
    prompts : sequence of AgentMessage
        One or more user supplied messages to start the turn.  Each
        prompt will generate `message_start` and `message_end` events.
    context : AgentContext
        Current conversation state containing system prompt, message
        history and tool definitions.  The context is not mutated
        until the loop completes.
    config : AgentLoopConfig
        Configuration object controlling how the loop interacts with
        the LLM and external environment.
    signal : asyncio.Event, optional
        Cancellation signal.  If set during execution the loop will
        attempt to abort pending LLM calls and tool executions.
    stream_fn : callable, optional
        Custom LLM transport.  If omitted the global default from
        ``POP.stream.stream`` is used (if available).
    """
    stream: EventStream[AgentEvent, List[AgentMessage]] = EventStream(
        done_predicate=lambda e: e.get("type") == "agent_end",
        result_selector=lambda e: e.get("messages", []),
    )

    async def _run() -> None:
        new_messages: List[AgentMessage] = []
        # Copy the context so that modifications do not affect the caller
        current_context = AgentContext(
            system_prompt=context.system_prompt,
            messages=list(context.messages),
            tools=context.tools,
        )

        # Append prompts to the context and emit events
        stream.push({"type": "agent_start"})
        stream.push({"type": "turn_start"})
        for prompt in prompts:
            stream.push({"type": "message_start", "message": _copy_message(prompt)})
            stream.push({"type": "message_end", "message": _copy_message(prompt)})
            current_context.messages.append(prompt)
            new_messages.append(prompt)

        try:
            await _run_loop(
                current_context,
                new_messages,
                config,
                signal,
                stream,
                stream_fn,
            )
        except Exception as exc:
            error_msg = _build_error_message(exc, config.model, aborted=bool(signal and signal.is_set()))
            new_messages.append(error_msg)
            stream.push({"type": "message_start", "message": _copy_message(error_msg)})
            stream.push({"type": "message_end", "message": _copy_message(error_msg)})
            stream.push({"type": "turn_end", "message": _copy_message(error_msg), "toolResults": []})
            stream.push({"type": "agent_end", "messages": [m for m in new_messages]})
            stream.end(new_messages)

    # Schedule the loop on the running event loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop; raise an error with guidance
        raise RuntimeError(
            "agent_loop must be called from within an asyncio event loop. "
            "Use asyncio.run() or create an event loop manually."
        )
    loop.create_task(_run())
    return stream


def agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    signal: Optional[asyncio.Event] = None,
    stream_fn: Optional[StreamFn] = None,
) -> EventStream[AgentEvent, List[AgentMessage]]:
    """Continue an agent loop from the current context without adding new messages.

    This is used to retry a previous turn when the last message in the
    context was not generated by the assistant.  If the last message
    is from the assistant this function raises a ``ValueError`` since
    continuing in that state would violate the expected message order.
    """
    if not context.messages:
        raise ValueError("Cannot continue: no messages in context")
    last = context.messages[-1]
    if last.role == "assistant":
        raise ValueError("Cannot continue from message role: assistant")

    stream: EventStream[AgentEvent, List[AgentMessage]] = EventStream(
        done_predicate=lambda e: e.get("type") == "agent_end",
        result_selector=lambda e: e.get("messages", []),
    )

    async def _run() -> None:
        new_messages: List[AgentMessage] = []
        current_context = AgentContext(
            system_prompt=context.system_prompt,
            messages=list(context.messages),
            tools=context.tools,
        )
        stream.push({"type": "agent_start"})
        stream.push({"type": "turn_start"})
        try:
            await _run_loop(
                current_context,
                new_messages,
                config,
                signal,
                stream,
                stream_fn,
            )
        except Exception as exc:
            error_msg = _build_error_message(exc, config.model, aborted=bool(signal and signal.is_set()))
            new_messages.append(error_msg)
            stream.push({"type": "message_start", "message": _copy_message(error_msg)})
            stream.push({"type": "message_end", "message": _copy_message(error_msg)})
            stream.push({"type": "turn_end", "message": _copy_message(error_msg), "toolResults": []})
            stream.push({"type": "agent_end", "messages": [m for m in new_messages]})
            stream.end(new_messages)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        raise RuntimeError(
            "agent_loop_continue must be called from within an asyncio event loop. "
            "Use asyncio.run() or create an event loop manually."
        )
    loop.create_task(_run())
    return stream


async def _run_loop(
    current_context: AgentContext,
    new_messages: List[AgentMessage],
    config: AgentLoopConfig,
    signal: Optional[asyncio.Event],
    stream: EventStream[AgentEvent, List[AgentMessage]],
    stream_fn: Optional[StreamFn],
) -> None:
    """Internal shared loop logic used by both agent_loop and agent_loop_continue.

    Parameters
    ----------
    current_context : AgentContext
        Copy of the current conversation context.  This object will be
        mutated as new messages and tool results are appended.
    new_messages : list of AgentMessage
        List used to collect all messages produced during this run.
    config : AgentLoopConfig
        Configuration controlling LLM calls and message transformations.
    signal : asyncio.Event, optional
        Cancellation signal propagated into the LLM transport and
        tools.  When set the loop attempts to abort pending calls.
    stream : EventStream
        Event queue to which lifecycle events are pushed.
    stream_fn : callable, optional
        LLM transport.  If omitted and POP is available this
        defaults to ``POP.stream.stream``.  Otherwise a
        ``RuntimeError`` is raised.
    """
    first_turn = True
    pending_messages: List[AgentMessage] = []
    # Preload any steering messages queued before the run starts
    if config.get_steering_messages:
        try:
            pending_messages = await config.get_steering_messages()
        except Exception:
            pending_messages = []

    # Determine the transport function
    if stream_fn is None:
        if POP is None:
            raise RuntimeError(
                "No stream_fn supplied and POP is not installed. "
                "Please provide a custom LLM transport function."
            )
        stream_fn = POP.stream.stream  # type: ignore

    # Outer loop: continue while follow‑up messages are returned
    while True:
        has_more_tool_calls = True
        steering_after_tools: Optional[List[AgentMessage]] = None

        # Inner loop: handle tool calls and steering messages within a turn
        while has_more_tool_calls or pending_messages:
            if not first_turn:
                stream.push({"type": "turn_start"})
            else:
                first_turn = False

            # Inject pending steering or follow‑up messages before LLM call
            if pending_messages:
                for message in pending_messages:
                    stream.push({"type": "message_start", "message": _copy_message(message)})
                    stream.push({"type": "message_end", "message": _copy_message(message)})
                    current_context.messages.append(message)
                    new_messages.append(message)
                pending_messages = []

            # Request assistant response
            message = await _stream_assistant_response(
                current_context,
                config,
                signal,
                stream,
                stream_fn,
            )
            # Track new assistant message
            new_messages.append(message)

            # Exit early on error or abort
            if message.stop_reason in {"error", "aborted"}:
                stream.push({"type": "turn_end", "message": _copy_message(message), "toolResults": []})
                stream.push({"type": "agent_end", "messages": [m for m in new_messages]})
                stream.end(new_messages)
                return

            # Determine if there are tool calls in the assistant message
            tool_calls = [item for item in message.content if isinstance(item, ToolCallContent)]
            has_more_tool_calls = len(tool_calls) > 0

            tool_results: List[AgentMessage] = []
            if has_more_tool_calls:
                execution = await _execute_tool_calls(
                    current_context.tools,
                    tool_calls,
                    signal,
                    stream,
                    config.get_steering_messages,
                )
                tool_results.extend(execution[0])
                steering_after_tools = execution[1]
                for result in tool_results:
                    current_context.messages.append(result)
                    new_messages.append(result)

            stream.push({"type": "turn_end", "message": _copy_message(message), "toolResults": tool_results})

            # Handle steering messages returned after tool execution
            if steering_after_tools and len(steering_after_tools) > 0:
                pending_messages = steering_after_tools
                steering_after_tools = None
            else:
                if config.get_steering_messages:
                    try:
                        pending_messages = await config.get_steering_messages()
                    except Exception:
                        pending_messages = []

        # No more tool calls; check for follow‑up messages before exiting
        follow_ups: List[AgentMessage] = []
        if config.get_follow_up_messages:
            try:
                follow_ups = await config.get_follow_up_messages()
            except Exception:
                follow_ups = []
        if follow_ups:
            pending_messages = follow_ups
            continue
        break

    # Normal termination
    stream.push({"type": "agent_end", "messages": [m for m in new_messages]})
    stream.end(new_messages)


async def _stream_assistant_response(
    context: AgentContext,
    config: AgentLoopConfig,
    signal: Optional[asyncio.Event],
    stream: EventStream[AgentEvent, List[AgentMessage]],
    stream_fn: StreamFn,
) -> AgentMessage:
    """Call the LLM transport and stream an assistant response.

    This function applies any context transformation, converts the
    messages to the provider specific format and invokes the LLM.  As
    events are received from the LLM they are pushed onto the event
    stream.  When the final response arrives the full assistant
    message is returned.
    """
    # Transform context if requested
    messages = context.messages
    if config.transform_context:
        try:
            messages = await config.transform_context(messages, signal)
        except Exception:
            # Ignore errors from context transform and use original
            messages = context.messages
            Warning("Context transformation failed; using original messages")

    # Convert AgentMessage objects to LLM compatible dicts
    llm_messages = await config.convert_to_llm(messages)

    # Build context dictionary expected by the LLM transport
    llm_context: Dict[str, Any] = {
        "system_prompt": context.system_prompt,
        "messages": llm_messages,
    }
    # Attach tool definitions if present.  Tools are expected to
    # implement name, description and parameters attributes.
    if context.tools:
        provider = (config.model.get("provider") or config.model.get("api") or "").lower()
        # OpenAI-compatible providers expect OpenAI tool schema.
        openai_like = {"openai", "deepseek", "doubao"}
        if provider in openai_like:
            llm_context["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": getattr(tool, "description", ""),
                        "parameters": getattr(tool, "parameters", {}),
                    },
                }
                for tool in context.tools
            ]
        else:
            llm_context["tools"] = [
                {
                    "type": "custom",
                    "name": tool.name,
                    "description": getattr(tool, "description", ""),
                    "parameters": getattr(tool, "parameters", {}),
                }
                for tool in context.tools
            ]

    # Assemble options for the LLM transport
    options: Dict[str, Any] = dict(config.other_options)
    if config.reasoning is not None:
        options["reasoning"] = config.reasoning
    if config.session_id is not None:
        options["session_id"] = config.session_id
    if config.thinking_budgets is not None:
        options["thinking_budgets"] = config.thinking_budgets
    if config.max_retry_delay_ms is not None:
        options["max_retry_delay_ms"] = config.max_retry_delay_ms
    # Determine API key: dynamic override takes precedence over static key
    api_key: Optional[str] = None
    if config.get_api_key:
        try:
            api_key = await config.get_api_key(config.model.get("provider", ""))
        except Exception:
            api_key = None
    if api_key is None:
        api_key = config.api_key
    if api_key:
        options["api_key"] = api_key
    # Propagate cancellation signal if present
    if signal:
        options["signal"] = signal

    partial_message: Optional[AgentMessage] = None
    added_partial = False

    def _finalize_error(exc: Exception) -> AgentMessage:
        error_msg = _build_error_message(exc, config.model, aborted=bool(signal and signal.is_set()))
        if added_partial and context.messages:
            context.messages[-1] = error_msg
        else:
            context.messages.append(error_msg)
        if not added_partial:
            stream.push({"type": "message_start", "message": _copy_message(error_msg)})
        stream.push({"type": "message_end", "message": _copy_message(error_msg)})
        return error_msg

    async def _consume_stream() -> AgentMessage:
        nonlocal partial_message, added_partial
        # Invoke the LLM transport.  The transport may be an async function
        # returning either a coroutine or the stream object directly.  We
        # support both patterns to remain compatible with various backends.
        resp_or_coro = stream_fn(config.model, llm_context, options)
        if asyncio.iscoroutine(resp_or_coro):
            response = await resp_or_coro
        else:
            response = resp_or_coro

        # Iterate over streaming events
        async for event in response:
            etype = event.get("type")
            if etype == "start":
                # Partial assistant message begins
                partial = event.get("partial")
                if partial is None:
                    continue
                partial_message = _dict_to_agent_message(partial)
                context.messages.append(partial_message)
                added_partial = True
                stream.push({"type": "message_start", "message": _copy_message(partial_message)})
            elif etype in {
                "text_start",
                "text_delta",
                "text_end",
                "thinking_start",
                "thinking_delta",
                "thinking_end",
                "toolcall_start",
                "toolcall_delta",
                "toolcall_end",
            }:
                # Update the partial message with the provided partial
                partial = event.get("partial")
                if partial is not None:
                    partial_message = _dict_to_agent_message(partial)
                    # Replace the last assistant message in the context
                    if context.messages:
                        context.messages[-1] = partial_message
                    stream.push(
                        {
                            "type": "message_update",
                            "assistantMessageEvent": event,
                            "message": _copy_message(partial_message),
                        }
                    )
            elif etype in {"done", "error"}:
                # Final assistant message
                # Try to obtain the full message from response.result()
                full_msg: Optional[Dict[str, Any]] = None
                if hasattr(response, "result"):
                    try:
                        full_msg = await response.result()
                    except Exception:
                        full_msg = None
                # Fall back to event provided message or error
                if full_msg is None:
                    full_msg = event.get("message") or event.get("error") or None
                if full_msg is None and partial_message is not None:
                    full_msg = partial_message.to_dict()  # type: ignore
                if full_msg is None:
                    # Should not happen; create a minimal assistant message
                    full_msg = {
                        "role": "assistant",
                        "content": [],
                        "timestamp": time.time(),
                        "stopReason": etype,
                    }
                final_message = _dict_to_agent_message(full_msg)
                # Replace or append final message in the context
                if added_partial and context.messages:
                    context.messages[-1] = final_message
                else:
                    context.messages.append(final_message)
                # Emit start event if we never emitted one
                if not added_partial:
                    stream.push({"type": "message_start", "message": _copy_message(final_message)})
                # Emit end event
                stream.push({"type": "message_end", "message": _copy_message(final_message)})
                return final_message
        # If the loop exits without receiving a done/error event fetch result
        final: Optional[Dict[str, Any]] = None
        if hasattr(response, "result"):
            try:
                final = await response.result()
            except Exception:
                final = None
        if final is None and partial_message is not None:
            final = partial_message.to_dict()  # type: ignore
        if final is None:
            final = {
                "role": "assistant",
                "content": [],
                "timestamp": time.time(),
                "stopReason": "error",
            }
        final_message = _dict_to_agent_message(final)
        if added_partial and context.messages:
            context.messages[-1] = final_message
        else:
            context.messages.append(final_message)
        # Emit final events
        if not added_partial:
            stream.push({"type": "message_start", "message": _copy_message(final_message)})
        stream.push({"type": "message_end", "message": _copy_message(final_message)})
        return final_message

    timeout_s = _coerce_timeout(config.request_timeout_s)
    try:
        if timeout_s is not None:
            return await asyncio.wait_for(_consume_stream(), timeout_s)
        return await _consume_stream()
    except asyncio.TimeoutError:
        if signal:
            signal.set()
        timeout_msg = f"LLM request timed out after {timeout_s:.0f}s" if timeout_s else "LLM request timed out"
        return _finalize_error(RuntimeError(timeout_msg))
    except Exception as exc:
        return _finalize_error(exc)


async def _execute_tool_calls(
    tools: Optional[Sequence[AgentTool]],
    tool_calls: List[ToolCallContent],
    signal: Optional[asyncio.Event],
    stream: EventStream[AgentEvent, List[AgentMessage]],
    get_steering_messages: Optional[Callable[[], Awaitable[List[AgentMessage]]]],
) -> Tuple[List[AgentMessage], Optional[List[AgentMessage]]]:
    """Execute a list of tool calls and return their results.

    Parameters
    ----------
    tools : sequence of AgentTool or None
        The available tools.  Each tool must define a ``name``
        attribute and an async ``execute`` method.  If ``None`` no
        tool calls are executed.
    tool_calls : list of ToolCallContent
        The tool calls extracted from the assistant message.
    signal : asyncio.Event, optional
        Cancellation signal.  Passed to tool executors; tools should
        periodically check and abort if set.
    stream : EventStream
        Event queue used to emit tool lifecycle events.
    get_steering_messages : callable, optional
        Callback to retrieve steering messages.  If non empty the
        remaining tool calls are skipped and those messages returned.

    Returns
    -------
    results : list of AgentMessage
        Messages representing tool results.  Each result is a
        ``toolResult`` message ready to append to the context.
    steering_messages : list of AgentMessage or None
        Steering messages returned from the callback if the user
        interrupted the run.  ``None`` means no steering
        interruption occurred.
    """
    results: List[AgentMessage] = []
    steering_messages: Optional[List[AgentMessage]] = None

    for index, tool_call in enumerate(tool_calls):
        tool = None
        if tools:
            for t in tools:
                if getattr(t, "name", None) == tool_call.name:
                    tool = t
                    break

        # Emit start event
        stream.push({
            "type": "tool_execution_start",
            "toolCallId": tool_call.id,
            "toolName": tool_call.name,
            "args": tool_call.arguments,
        })
        # Execute the tool
        is_error = False
        result: AgentToolResult
        try:
            if tool is None:
                raise RuntimeError(f"Tool {tool_call.name} not found")
            validated_args = _validate_tool_arguments(tool, tool_call)
            # Define update callback to stream partial results
            async def on_update(partial: AgentToolResult) -> None:
                stream.push({
                    "type": "tool_execution_update",
                    "toolCallId": tool_call.id,
                    "toolName": tool_call.name,
                    "args": tool_call.arguments,
                    "partialResult": partial,
                })
            # Some tool implementations may not expect an 'async' callback; wrap accordingly
            exec_result = tool.execute  # type: ignore
            # Call asynchronously and pass on_update
            result = await exec_result(tool_call.id, validated_args, signal, on_update)  # type: ignore
        except Exception as exc:
            # Convert error into a tool result
            is_error = True
            result = AgentToolResult(
                content=[TextContent(type="text", text=str(exc))],
                details={},
            )

        # Emit end event
        stream.push({
            "type": "tool_execution_end",
            "toolCallId": tool_call.id,
            "toolName": tool_call.name,
            "result": result,
            "isError": is_error,
        })
        # Build tool result message
        tool_result_msg = AgentMessage(
            role="toolResult",
            content=list(result.content),
            timestamp=time.time(),
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            details=result.details,
            is_error=is_error,
            extras={},
        )
        results.append(tool_result_msg)
        # Emit message events
        stream.push({"type": "message_start", "message": _copy_message(tool_result_msg)})
        stream.push({"type": "message_end", "message": _copy_message(tool_result_msg)})

        # Check for steering messages and skip remaining tools if needed
        if get_steering_messages:
            try:
                steering = await get_steering_messages()
            except Exception:
                steering = []
            if steering:
                steering_messages = steering
                # Skip remaining tool calls
                for skipped in tool_calls[index + 1 :]:
                    skipped_msg = _skip_tool_call(skipped, stream)
                    results.append(skipped_msg)
                break

    return results, steering_messages


def _skip_tool_call(tool_call: ToolCallContent, stream: EventStream[AgentEvent, List[AgentMessage]]) -> AgentMessage:
    """Create a tool result message for a skipped tool call.

    When the user interrupts a run via steering messages remaining
    tool calls are skipped.  This helper emits the appropriate
    lifecycle events and returns the tool result message to insert
    into the context.
    """
    result = AgentToolResult(
        content=[TextContent(type="text", text="Skipped due to queued user message.")],
        details={},
    )
    # Emit start and end events for skipped execution
    stream.push({
        "type": "tool_execution_start",
        "toolCallId": tool_call.id,
        "toolName": tool_call.name,
        "args": tool_call.arguments,
    })
    stream.push({
        "type": "tool_execution_end",
        "toolCallId": tool_call.id,
        "toolName": tool_call.name,
        "result": result,
        "isError": True,
    })
    # Build tool result message
    tool_result_msg = AgentMessage(
        role="toolResult",
        content=list(result.content),
        timestamp=time.time(),
        tool_call_id=tool_call.id,
        tool_name=tool_call.name,
        details=result.details,
        is_error=True,
        extras={},
    )
    stream.push({"type": "message_start", "message": _copy_message(tool_result_msg)})
    stream.push({"type": "message_end", "message": _copy_message(tool_result_msg)})
    return tool_result_msg
