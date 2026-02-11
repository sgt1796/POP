"""Pop Agent Python package.

This package implements a lightweight, event driven agent loop for
Large Language Models (LLMs) inspired by the JavaScript `pi‑agent`
project.  See the documentation in :mod:`pop_agent.README` for an
overview and usage examples.

The primary entry points are:

* :class:`pop_agent.agent.Agent` – high level API for managing
  conversations and tools.
* :func:`pop_agent.agent_loop.agent_loop` – runs a single prompt
  through the agent loop.
* :func:`pop_agent.agent_loop.agent_loop_continue` – resumes an
  existing conversation when the last message was not from the
  assistant.
* :func:`pop_agent.proxy.stream_proxy` – optional proxy transport
  used to route LLM calls through an intermediate server.

"""

from .agent import Agent
from .agent_loop import agent_loop, agent_loop_continue
from .proxy import stream_proxy
from .agent_types import (
    AgentMessage,
    AgentState,
    AgentTool,
    AgentToolResult,
    AgentContext,
    AgentEvent,
    ThinkingLevel,
    ToolCapability,
    ToolPolicy,
    ToolSpec,
    ToolBuildRequest,
    ToolBuildResult,
)
from .dynamic_tools.registry import DynamicToolRegistry

__all__ = [
    "Agent",
    "agent_loop",
    "agent_loop_continue",
    "stream_proxy",
    "AgentMessage",
    "AgentState",
    "AgentTool",
    "AgentToolResult",
    "AgentContext",
    "AgentEvent",
    "ThinkingLevel",
    "ToolCapability",
    "ToolPolicy",
    "ToolSpec",
    "ToolBuildRequest",
    "ToolBuildResult",
    "DynamicToolRegistry",
]
