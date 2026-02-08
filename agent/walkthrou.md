Below is a documentation-style walkthrough of **`agent.py`**: what it’s for, how it’s structured, the lifecycle/state machine, and how each method interacts with `agent_loop`. 

---

# `agent.py` — `Agent`: a stateful conversation orchestrator

## Purpose

`agent.py` exposes a high-level `Agent` class that:

* Owns **conversation state** (system prompt, model, messages, tools, streaming flags).
* Provides a convenient API to:

  * send prompts (`prompt`)
  * resume a paused turn (`continue_run`)
  * enqueue **steering** messages (interrupt / change direction mid-loop)
  * enqueue **follow-up** messages (append after tool processing completes)
* Bridges your app/UI with the low-level **event-driven loop** (`agent_loop`, `agent_loop_continue`)
* Supports **event subscribers** to observe fine-grained events (token streaming, tool execution, turn end, etc.)

Think: `Agent` is *the façade and state container*; `agent_loop.py` is *the engine*.

---

# Imports and dependencies

### Internal dependencies

* `agent_loop.AgentLoopConfig`, `agent_loop`, `agent_loop_continue`
* `EventStream` (used as the iterable async event stream returned by the loop)
* Types from `agent_types`:

  * `AgentContext`, `AgentMessage`, `AgentState`, `AgentTool`, `AgentToolResult`
  * `ThinkingLevel`, `TextContent`, `ImageContent`

### Optional dependency: `POP`

`agent.py` tries to import `POP` to obtain defaults:

* A default model via `POP.get_model(...)`
* A default streaming transport via `POP.stream_simple`

If POP is not available, it falls back to “unknown” model metadata and sets `stream_fn` to `None` (error delayed until used). 

---

# Helper: `_default_convert_to_llm(messages)`

## What it does

This is the default “adapter” from internal `AgentMessage` to whatever the LLM transport expects (as dicts).

Key behaviors:

1. **Filters roles**: only forwards messages where `m.role in {"user", "assistant", "toolResult"}`.

   * Any other roles (custom notifications) are discarded.
2. **Primary conversion**: calls `m.to_dict()`.
3. **Fallback conversion**: if `to_dict()` fails, it builds a best-effort dict with:

   * `"role"`, `"content"` (via `vars(c)` for each content block), `"timestamp"`

It returns an awaitable by wrapping the result in an async `_return()` coroutine. 

## Why it matters

`AgentLoopConfig.convert_to_llm` is one of the key extension points: if your provider needs a different schema (e.g., tool specs, image formats, structured reasoning channels), you override this function when constructing `Agent(opts={...})`.

---

# Class: `Agent`

## Responsibilities

The `Agent` class is responsible for:

1. **State** (`self._state: AgentState`)
2. **Listeners** (`self._listeners: set[Callable[[AgentEvent], None]]`)
3. **Queues**

   * `_steering_queue: List[AgentMessage]`
   * `_follow_up_queue: List[AgentMessage]`
   * plus modes controlling dequeue semantics
4. **Loop configuration**

   * message conversion
   * context transform
   * model selection
   * reasoning control (`thinking_level`)
   * session id / api key hooks / retry budget hooks
   * stream function (`stream_fn`)
5. **Concurrency primitives**

   * `_idle_event`: signals “agent is idle”
   * `_abort_event`: signals “abort current turn”

---

# Construction: `__init__(opts=None)`

## 1) Establish defaults

### Default model

If `POP` is available, it attempts:

```python
POP.get_model("google", "gemini-2.5-flash-lite-preview-06-17")
```

If that fails, or POP is absent, it uses:

```python
{"provider": "unknown", "id": "unknown", "api": None}
```

### Default state

Initial `AgentState` includes:

* `system_prompt=""`
* `model=default_model`
* `thinking_level="off"`
* `tools=[]`
* `messages=[]`
* streaming flags:

  * `is_streaming=False`
  * `stream_message=None`
* tool bookkeeping:

  * `pending_tool_calls=set()`
* `error=None`

Then it merges in `opts["initial_state"]` fields if provided *and* they exist on the dataclass. 

## 2) Hooks / extension points

* `_convert_to_llm`: from opts or `_default_convert_to_llm`
* `_transform_context`: optional function to prune/augment context before each LLM call

## 3) Steering / follow-up config

* `_steering_mode`: `"one-at-a-time"` (default) or `"all"`
* `_follow_up_mode`: `"one-at-a-time"` (default) or `"all"`

## 4) Transport & session configuration

* `stream_fn`:

  * from opts if provided
  * else `POP.stream_simple` if POP is available
  * else `None` (delayed error)
* `_session_id`: optional, forwarded to provider
* `get_api_key`: optional async hook to fetch key per call
* `_thinking_budgets`: optional
* `_max_retry_delay_ms`: optional

## 5) Concurrency primitives

* `_idle_event` is created and **set** (agent starts idle)
* `_abort_event` starts as `None` until a run begins 

---

# Properties

## `state`

Returns the current `AgentState`. Docstring warns: treat it read-only; mutate via setters.

## `session_id` (getter/setter)

Simple stored field used in the loop config.

## `thinking_budgets` and `max_retry_delay_ms` (getter/setter)

Stored fields forwarded into `AgentLoopConfig`. 

---

# Event subscription

## `subscribe(fn) -> unsubscribe_fn`

* Adds `fn` to `_listeners`
* Returns an `unsubscribe` closure that removes it

Important detail: listeners are invoked synchronously (in `_emit`) and exceptions are swallowed so a bad listener can’t break the agent. 

---

# State mutators

These are straightforward setters against `self._state` plus mode validation:

* `set_system_prompt(prompt)`
* `set_model(model_dict)`
* `set_thinking_level(level)`
* `set_steering_mode(mode)` validates mode ∈ {"all","one-at-a-time"}
* `set_follow_up_mode(mode)` validates mode ∈ {"all","one-at-a-time"}
* `set_tools(tools)`
* `replace_messages(messages)`
* `append_message(message)`

They do **not** trigger a run; they only mutate configuration/state. 

---

# Queue management

## `steer(message)`

Appends to `_steering_queue`. These messages are meant to be pulled *mid-turn* by the low-level loop via the callback `get_steering_messages`.

## `follow_up(message)`

Appends to `_follow_up_queue`. These are pulled when the loop is otherwise about to finish.

## Clear helpers

* `clear_steering_queue`
* `clear_follow_up_queue`
* `clear_all_queues`

Also:

* `clear_messages` wipes conversation history but does not touch tools or system prompt. 

---

# Control operations

## `abort()`

If a run is active (`_abort_event` not None), sets the abort event. The low-level loop should be checking this and terminating.

## `wait_for_idle()`

Waits on `_idle_event`. This is the canonical way for “block until the current run finishes.”

## `reset()`

Resets:

* messages
* streaming flags
* `pending_tool_calls`
* error
* steering/follow-up queues

It does *not* reset system prompt, model, tools, session id, etc. 

---

# High-level run APIs

## `prompt(input, images=None)`

### Input forms accepted

* `str` → wrapped into a `user` `AgentMessage`

  * content begins with `TextContent(type="text", text=input)`
  * any `images` are appended to content
* `AgentMessage` → used directly
* `Sequence[AgentMessage]` → used directly

### Guardrail

If `self._state.is_streaming` is True, it raises:

> Agent is already processing a prompt. Use steer() or follow_up()...

This ensures a single run at a time; steering/follow-up are the supported concurrent interaction mechanism. 

### Call path

Once messages are normalized, calls:

```python
await self._run_loop(msgs)
```

---

## `continue_run()`

Intended when the last message in history is not an assistant message and you want the model to continue.

Guards:

* cannot be currently streaming
* must have messages
* last message cannot be `role == "assistant"`

Then calls:

```python
await self._run_loop(None)
```

So `_run_loop(None)` means “continue mode” (use `agent_loop_continue`). 

## `continue_()`

Alias for backwards compatibility (since `continue` is a keyword).

---

# Core engine glue: `_run_loop(messages: Optional[Sequence[AgentMessage]])`

This is the most important method in `agent.py`. It:

1. Prepares concurrency primitives and resets “streaming state”
2. Builds `AgentContext` and `AgentLoopConfig`
3. Chooses `agent_loop` vs `agent_loop_continue`
4. Consumes events and updates internal `AgentState`
5. Emits events to subscribers
6. Converts exceptions into an assistant error message
7. Guarantees cleanup in `finally`

## 1) Begin a run

It does:

* `_idle_event.clear()` (agent is no longer idle)
* `_abort_event = asyncio.Event()` (new abort signal for this run)
* `_state.is_streaming = True`
* `_state.stream_message = None`
* `_state.error = None`

## 2) Build context

`AgentContext` contains:

* system prompt
* a *copy* of message history
* a *copy* of tool list

This is important: the loop works from snapshots, while the agent updates state based on events. 

## 3) Build loop config (`AgentLoopConfig`)

Key fields mapped:

* `model=self._state.model`
* `convert_to_llm=self._convert_to_llm`
* `transform_context=self._transform_context`
* `get_api_key=self.get_api_key`
* `get_steering_messages=self._get_steering_messages`
* `get_follow_up_messages=self._get_follow_up_messages`
* `reasoning = None if thinking_level == "off" else thinking_level`
* `session_id=self._session_id`
* `thinking_budgets=self._thinking_budgets`
* `max_retry_delay_ms=self._max_retry_delay_ms`
* `api_key=None` (interesting: it keeps a separate static key slot but they leave it empty)
* `other_options={}`

## 4) Choose which loop function to call

* If `messages` is provided (non-empty) → **new turn**:

  ```python
  stream = agent_loop(messages, context, loop_config, self._abort_event, self.stream_fn)
  ```

* Else → **continue turn**:

  ```python
  stream = agent_loop_continue(context, loop_config, self._abort_event, self.stream_fn)
  ```

So `agent.py` does not implement continuation logic itself; it delegates to the loop. 

## 5) Consume events and update state

This block is the internal state machine. For each event:

### Message streaming

* `message_start` → `state.stream_message = event["message"]`
* `message_update` → keep updating `state.stream_message`
* `message_end`:

  * clear `stream_message`
  * append the final `AgentMessage` to `state.messages` (if it is an `AgentMessage`)

This is what lets a UI show “currently streaming message” separately from finalized history.

### Tool execution bookkeeping

* `tool_execution_start`:

  * extracts `toolCallId` and adds it to `pending_tool_calls`
* `tool_execution_end`:

  * removes `toolCallId` from `pending_tool_calls`

This is primarily a UI/telemetry hook so you can show “tools in flight”.

### Turn completion / error propagation

* `turn_end`:

  * if event includes an assistant message with `error_message`, copy it to `state.error`

### Agent completion

* `agent_end`:

  * `state.is_streaming = False`
  * `state.stream_message = None`

### Event subscribers

Regardless of type, `self._emit(event)` dispatches the event to all listeners.

## 6) Exception handling

Any exception in the above run is turned into an assistant `AgentMessage` with:

* `stop_reason = "aborted"` if abort event set else `"error"`
* `error_message = str(exc)`
* content is a blank text content `TextContent(type="text", text="")`
* provider/api/model fields copied from `self._state.model` when possible

Then:

* it appends this error message to `state.messages`
* sets `state.error`
* emits a synthetic `{"type": "agent_end", "messages": [error_msg]}`

This is a “never throw outward; always record error in transcript” design. 

## 7) Cleanup (`finally`)

No matter what:

* `is_streaming=False`
* `stream_message=None`
* `pending_tool_calls=set()` (clears everything)
* `_abort_event=None` (discard abort signal)
* `_idle_event.set()` (agent returns idle)

This makes `wait_for_idle()` reliable even on failure.

---

# Event dispatch: `_emit(event)`

Iterates listeners and calls each, swallowing exceptions.

Implementation note: it iterates over `list(self._listeners)` to avoid issues if listeners modify subscriptions during callbacks. 

---

# Queue dequeue callbacks: `_get_steering_messages`, `_get_follow_up_messages`

These are async functions passed into `AgentLoopConfig` so the low-level loop can pull messages at decision points.

## Steering dequeue logic

* If queue empty → `[]`
* If mode `"one-at-a-time"` → pop index 0 and return `[msg]`
* Else `"all"` → return all queued and clear queue

## Follow-up dequeue logic

Same structure, different queue.

These callbacks are the only interface the loop needs; it doesn’t access Agent internals directly. 

---

# Key invariants and “gotchas” (practical reading)

1. **Single run at a time**
   `prompt()` and `continue_run()` hard-fail if `is_streaming` is true. You must use `steer()` / `follow_up()` for mid-run interaction.

2. **`stream_fn` may be `None`**
   If POP isn’t installed and you didn’t pass `stream_fn`, you’ll fail at first run. This is intentional (“delay error until first invocation”). 

3. **State snapshots vs live state**
   `_run_loop` builds `AgentContext` from **copies** of messages/tools, then updates `self._state` from events. If you mutate `self._state.messages` externally while streaming, you’re racing the loop’s event updates.

4. **Error message content is empty**
   On exceptions, it creates an assistant message with empty text and uses `error_message` field for the error. Your UI should display `error_message` if present.

5. **Pending tool calls tracking depends on event shape**
   It assumes events contain `toolCallId`. If your transport emits a different key, your UI/tool tracking won’t work unless you normalize events upstream in the loop/transport.

---

# How you typically use this in an app

* Subscribe to events:

  * update token stream display from `message_update`
  * append finalized assistant messages on `message_end`
  * show tool execution spinners on `tool_execution_start/end`
* Trigger `prompt(...)`
* If the user interrupts:

  * call `agent.steer(user_msg)` while it’s streaming
* If the user adds “oh also…” after the agent finishes tool calls:

  * call `agent.follow_up(user_msg)`

---

If you want, I can also produce:

* a small “event dictionary schema” cheat sheet inferred from what `Agent` consumes (`message_*`, `tool_execution_*`, `turn_end`, `agent_end`)
* a sequence diagram for `prompt → _run_loop → agent_loop → events → state updates`
