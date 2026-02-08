
import asyncio, time
from agent import Agent
from agent_types import AgentMessage, TextContent, AgentToolResult, AgentTool
from POP.stream import stream

class SlowTool(AgentTool):
    name = "slow"
    description = "Sleep a bit"
    parameters = {"type": "object", "properties": {"seconds": {"type": "number"}}}
    label = "Slow"

    async def execute(self, tool_call_id, params, signal=None, on_update=None):
        seconds = float(params.get("seconds", 1.0))
        steps = max(1, int(seconds / 0.1))
        for _ in range(steps):
            if signal and signal.is_set():
                break
            await asyncio.sleep(0.1)
        return AgentToolResult(
            content=[TextContent(type="text", text=f"slow done {seconds}s")],
            details={},
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

class FakeStream:
    def __init__(self, message):
        self._message = message
        self._started = False

    async def __aiter__(self):
        if self._started:
            return
        self._started = True
        yield {"type": "done", "message": self._message}

    async def result(self):
        return self._message
    
def print_state(agent):
    print("Agent State:")
    for k, v in agent._state.__dict__.items():
        if k == "messages":
            print(f"{k}:")
            for m in v:
                print(f"  - {m.role}: {[c.text for c in m.content if isinstance(c, TextContent)]}")
            continue
    print(f"{k}: {v}")


def make_stream(script):
    state = {"i": 0}
    async def stream_fn(model, context, options):
        msg = script[state["i"]]
        state["i"] += 1
        return FakeStream(msg)
    return stream_fn

def assistant_toolcall_message():
    return {
        "role": "assistant",
        "content": [
            {"type": "toolCall", "id": "call1", "name": "slow", "arguments": {"seconds": 1.2}},
            {"type": "toolCall", "id": "call2", "name": "fast", "arguments": {}},
        ],
        "timestamp": time.time(),
        "stopReason": "stop",
    }

def assistant_text_message(text):
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "timestamp": time.time(),
        "stopReason": "stop",
    }

async def main():
    script = [
        assistant_toolcall_message(),
        assistant_text_message("Saw your steering message. New plan."),
        assistant_text_message("Follow up handled."),
    ]

    agent = Agent({"stream_fn": stream})
    agent.set_model({"provider": "openai", "id": None, "api": None})
    agent.set_tools([SlowTool(), FastTool()])

    def log(event):
        cp = event.copy() # don't mutate the original event
        t = cp.pop("type")
        print(f"[ {t} ] {cp}")


    unsubscribe_log = agent.subscribe(log)
    #unsubscribe_log()

    agent.follow_up(AgentMessage(
        role="user",
        content=[TextContent(type="text", text="follow up: summarize")],
        timestamp=time.time(),
    ))

    task = asyncio.create_task(agent.prompt("Call tool slow with seconds=1.2, then call tool fast"))
    print(agent._state)
    ##await asyncio.sleep(5)
    agent.steer(AgentMessage(
        role="user",
        content=[TextContent(type="text", text="steer: change direction")],
        timestamp=time.time(),
    ))
    await task
    print_state(agent)

    print("Final message roles:")
    print([m.role for m in agent.state.messages])



if __name__ == "__main__":
    asyncio.run(main())
