import asyncio, time
from .agent import Agent
from .agent_types import AgentMessage, TextContent, AgentToolResult, AgentTool
from pop.stream import stream

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

    def log(event):
        cp = event.copy() # don't mutate the original event
        t = cp.pop("type")
        # print(agent.state.messages)
        print(f"Event: {t}, Message: {cp.get('message').content if cp.get('message') else None}, Details: {cp.get('details')}")
        


    unsubscribe_log = agent.subscribe(log)
    #unsubscribe_log()

    agent.follow_up(AgentMessage(
        role="user",
        content=[TextContent(type="text", text="follow up: summarize")],
        timestamp=time.time(),
    ))

    task = asyncio.create_task(agent.prompt("Call tool slow with seconds=1.2, then call tool fast"))

    ##await asyncio.sleep(5)
    agent.steer(AgentMessage(
        role="user",
        content=[TextContent(type="text", text="steer: actually, call slow 4 times, but keep the total time unchanged. then fast")],
        timestamp=time.time(),
    ))
    await task
    print_state(agent)

    print("Final message roles:")
    print([m.role for m in agent.state.messages])



if __name__ == "__main__":
    asyncio.run(main())
