# Pop Agent (Python)

**Pop Agent** is a Python implementation of the agentic loop used in the
[`pi‑agent`](https://github.com/badlogic/pi-mono) TypeScript project.  It
provides a lightweight event driven framework for building long running
LLM agents with tool calling, steering messages and follow‑up
interactions.  The goal of this package is to mirror the behaviour of
the original TypeScript implementation while remaining easy to read and
modify.  It deliberately avoids pulling in large frameworks so that it
can be used as a simple building block for your own applications.

The package is designed to work together with the
[`pop‑python`](https://pypi.org/project/pop-python/) library.  That
library exposes a unified Large Language Model (LLM) API similar to
[`pi‑ai`] from the JavaScript ecosystem.  By keeping the agent layer
separate from the LLM backend you can swap out providers without
changing your business logic.

## Features

- **Agent state management** – encapsulates system prompt, model
  selection, message history, tool definitions and error state.
- **Event streaming API** – agents emit a sequence of events during
  execution such as `agent_start`, `message_start`, `tool_execution_end`,
  etc.  Consumers (e.g. TUI or web UIs) can update their UI in
  response to these events.
- **Steering and follow‑up queues** – at any time you can queue
  additional user messages to steer the agent mid‑conversation or wait
  until the current turn completes.  Follow‑ups allow you to insert
  messages only after the agent has finished processing all tool
  calls.
- **Pluggable LLM transport** – agents rely on a user supplied
  ``stream_fn`` to invoke the underlying language model.  If you
  are using `pop‑python` you can simply pass
  ``POP.stream_simple``.  For proxy deployments you can use
  :func:`pop_agent.proxy.stream_proxy` together with
  :class:`pop_agent.proxy.ProxyStreamOptions` to route requests
  through an intermediate server without exposing API keys to
  clients.
- **Tool calling** – define your own tools by subclassing
  `AgentTool` and implementing the `execute` coroutine.  Tools can
  stream partial results back to the agent via the `on_update`
  callback.
- **Secure shell execution (`bash_exec`)** – the chatroom runtime
  (`agent_build/agent0.py`) registers a static one-shot shell tool
  with command allowlists, path scoping, risk-based approvals, timeout
  and output limits.

This project only provides the agent logic.  It does **not** ship
any concrete LLM providers or authentication logic.  See the
[`pop-python`](https://pypi.org/project/pop-python/) documentation for
information on configuring models and API keys.

## Getting Started

First install the package from a local checkout or via pip once
published:

```bash
pip install pop-agent
```

Below is a minimal example that sends a greeting to an agent, waits
for the response and prints out all emitted events.  The example
uses a dummy `stream_fn` that simply echoes back the last user
message instead of calling a real LLM.

```python
import asyncio
from pop_agent import Agent, AgentMessage, AgentTool


class EchoTool(AgentTool):
    """Simple tool that echoes its arguments back as text."""
    name = "echo"
    description = "Echo the provided text."
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    label = "Echo"

    async def execute(self, tool_call_id: str, params: dict, signal=None, on_update=None):
        text = params.get("text", "")
        result = {"content": [{"type": "text", "text": text}], "details": {}}
        return result


async def dummy_stream_fn(model, context, options):
    """Very simple stream function that returns one assistant message."""
    # The last message from the user will be echoed back.
    user_messages = [m for m in context["messages"] if m["role"] == "user"]
    reply = user_messages[-1]["content"][0]["text"] if user_messages else "Hello"
    # Build assistant message
    assistant_msg = {
        "role": "assistant",
        "content": [{"type": "text", "text": reply}],
        "timestamp": options.get("timestamp", 0),
        "stopReason": "stop",
        "api": model.get("api"),
        "provider": model.get("provider"),
        "model": model.get("id"),
        "usage": {},
    }
    # Yield one done event; a real implementation would yield partial
    # updates followed by a final event.
    class SimpleStream:
        def __init__(self, message):
            self._message = message
        async def __aiter__(self):
            yield {"type": "done", "reason": "stop", "message": self._message}
        async def result(self):
            return self._message

    return SimpleStream(assistant_msg)


async def main():
    agent = Agent({"initial_state": {"system_prompt": "You are helpful."}, "stream_fn": dummy_stream_fn})
    # Subscribe to events
    def on_event(event):
        print(f"event: {event['type']}")
    agent.subscribe(on_event)
    # Send a prompt
    await agent.prompt("Hello, world!")
    # Wait until the agent finishes
    await agent.wait_for_idle()
    # Inspect the conversation
    for msg in agent.state["messages"]:
        print(msg)

asyncio.run(main())
```

In a real application you would replace `dummy_stream_fn` with
`POP.stream_simple` and supply your model via
`POP.get_model(...)`.  You can also define tools that call
external services or your own business logic.

## Project Structure

- **`agent.py`** – High level `Agent` class that maintains the state
  of the conversation and exposes convenience methods like
  `prompt`, `continue` and `abort`.
- **`agent_loop.py`** – Core loop functions (`agent_loop` and
  `agent_loop_continue`) which orchestrate turns, calls the LLM
  transport and tools, and emit events.
- **`event_stream.py`** – Lightweight implementation of an
  asynchronous event queue.  Instances are returned from the loop
  functions and yield `AgentEvent` dictionaries.
- **`proxy.py`** – Optional support for routing requests through an
  external proxy server that wraps the underlying LLM API.  Useful
  for server side authentication and caching.
- **`agent_types.py`** – Dataclasses and type definitions used throughout
  the package.  These provide structure to messages, tools and
  events but remain flexible so that you can extend them for your
  specific needs.

Refer to the inline documentation within each module for details on
the available functions, classes and options.

## Secure `bash_exec` tool (chatroom runtime)

The chatroom entrypoint (`agent_build/agent0.py`) registers a built-in
`bash_exec` tool with a strict safety model:

- One-shot execution only (`asyncio.create_subprocess_exec`, invoked without shell).
- No shell control operators or command chaining (`|`, `;`, `&&`, `||`,
  backticks, `$(`, redirection, newlines).
- Command allowlist only:
  - Read: `pwd`, `ls`, `cat`, `head`, `tail`, `wc`, `find`, `rg`, `git`
  - Write: `mkdir`, `touch`, `cp`, `mv`, `rm`
- Conservative flag restrictions:
  - `find`: blocks execution/file-writing primitives (for example `-exec`, `-delete`)
  - `rg`: blocks `--pre` and `--pre-glob`
  - `git`: blocks global path/config escape options (for example `-C`, `--git-dir`)
  - `cp`/`mv`: blocks `-t`/`--target-directory`
- Path policy:
  - `cwd` must be inside `allowed_roots`
  - Read paths must resolve inside `allowed_roots`
  - Write targets must resolve inside `writable_roots`
  - Writes targeting a writable root directory itself are blocked
- Risk policy:
  - `low`: read commands (auto-run)
  - `medium`: `mkdir`, `touch`, `cp`, `mv` (approval required)
  - `high`: `rm` (approval required)
- Final output only (no streaming chunks), with timeout and max output cap.

Environment variables used by `agent_build/agent0.py`:

- `POP_AGENT_BASH_ALLOWED_ROOTS` (comma-separated, default current workspace root)
- `POP_AGENT_BASH_WRITABLE_ROOTS` (comma-separated, default current workspace root)
- `POP_AGENT_BASH_TIMEOUT_S` (default `15`)
- `POP_AGENT_BASH_MAX_OUTPUT_CHARS` (default `20000`)
- `POP_AGENT_BASH_PROMPT_APPROVAL` (default `true`; if `false`, approval-required commands are denied)

Example read call:

```json
{"cmd":"ls agent/tools"}
```

Example write call requiring approval:

```json
{"cmd":"touch notes.txt","justification":"Create an empty notes file for this task"}
```
