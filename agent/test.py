import asyncio, time, os
from .agent import Agent
from .agent_types import AgentMessage, TextContent, AgentToolResult, AgentTool
from POP.stream import stream

class SlowTool(AgentTool):
    name = "slow"
    description = "Sleep a bit"
    parameters = {"type": "object", "properties": {"seconds": {"type": "number"}}}
    label = "Slow"

    async def execute(self, tool_call_id, params, signal=None, on_update=None):
        t0 = time.time()
        seconds = float(params.get("seconds", 1.0))
        steps = max(1, int(seconds * 10))
        for _ in range(steps):
            if signal and signal.is_set():
                break
            await asyncio.sleep(0.1)
        return AgentToolResult(
            content=[TextContent(type="text", text=f"slow done {seconds}s")],
            details={"time_elapsed": time.time() - t0},
        )

class FastTool(AgentTool):
    name = "fast"
    description = "Return quickly"
    parameters = {"type": "object", "properties": {}}
    label = "Fast"

    async def execute(self, tool_call_id, params, signal=None, on_update=None):
        return AgentToolResult(
            content=[TextContent(type="text", text="fast done")],
            details={},
        )

    
# Logging helpers
LOG_LEVELS = {
    "quiet": 0,
    "messages": 1,
    "stream": 2,
    "debug": 3,
}


def _resolve_log_level(value: str) -> int:
    if not value:
        return LOG_LEVELS["messages"]
    key = str(value).strip().lower()
    if key.isdigit():
        return int(key)
    return LOG_LEVELS.get(key, LOG_LEVELS["messages"])


def _extract_texts(message):
    texts = []
    if not message:
        return texts
    content = getattr(message, "content", None)
    if not content:
        return texts
    for item in content:
        if isinstance(item, TextContent):
            texts.append(item.text)
        elif isinstance(item, dict) and item.get("type") == "text":
            texts.append(item.get("text", ""))
    return texts


def _format_message_line(message):
    role = getattr(message, "role", "unknown")
    return f"- {role}: {_extract_texts(message)}"


def make_event_logger(level: str = "messages"):
    '''
    Create an event logger function for agent events.
    
    Parameters
    ----------
    level : str
        The logging level. One of "quiet", "messages", "stream", "debug".
        Defaults to "messages".
    '''
    level_value = _resolve_log_level(level)

    def log(event):
        if level_value <= LOG_LEVELS["quiet"]:
            return
        etype = event.get("type")

        if etype == "message_end":
            message = event.get("message")
            if message:
                print(_format_message_line(message))
            return

        if level_value >= LOG_LEVELS["stream"] and etype == "message_update":
            assistant_event = event.get("assistantMessageEvent") or {}
            if assistant_event.get("type") == "text_delta":
                delta = assistant_event.get("delta")
                if delta:
                    print(f"- assistant: {[delta]}")
            return
        # Level 4: debug - log all events
        if level_value >= LOG_LEVELS["debug"]:
            print(f"Event: {event}")

    return log


def print_state(agent):
    print("Agent State:")
    for k, v in agent._state.__dict__.items():
        if k == "messages":
            print(f"{k}:")
            for m in v:
                print(f"  - {m.role}: {[c.text for c in m.content if isinstance(c, TextContent)]}")
            continue
    print(f"{k}: {v}")


async def main():
    agent = Agent({"stream_fn": stream})
    agent.set_model({"provider": "openai", "id": None, "api": None})
    agent.set_tools([SlowTool(), FastTool()])

    log_level = "messages"
    unsubscribe_log = agent.subscribe(make_event_logger(log_level))
    #unsubscribe_log()

    agent.follow_up(AgentMessage(
        role="user",
        content=[TextContent(type="text", text="follow up: summarize")],
        timestamp=time.time(),
    ))

    task = asyncio.create_task(agent.prompt("Call tool slow with seconds=1.2, then call tool fast"))
    agent.set_timeout(120)  # Set a timeout of 10 seconds for the agent's operations

    ##await asyncio.sleep(5)
    agent.steer(AgentMessage(
        role="user",
        content=[TextContent(type="text", text="steer: actually, call slow 4 times, and in between of each call add a fast call, but keep the total time unchanged as 1.2s. then fast")],
        timestamp=time.time(),
    ))
    await task
    print_state(agent)

    print("Final message roles:")
    print([m.role for m in agent.state.messages])



if __name__ == "__main__":
    asyncio.run(main())
