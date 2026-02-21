from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, Tuple

from agent import Agent
from agent.agent_types import AgentToolResult
from agent_build.agent0 import ToolsmakerTool


async def _dummy_stream_fn(model: Dict[str, Any], context: Dict[str, Any], options: Dict[str, Any]):
    del model, context, options

    message = {
        "role": "assistant",
        "content": [{"type": "text", "text": "ok"}],
        "timestamp": 0.0,
        "stopReason": "stop",
        "usage": {},
    }

    class _DummyStream:
        async def __aiter__(self):
            yield {"type": "done", "message": message}

        async def result(self):
            return message

    return _DummyStream()


def _make_agent_and_tool(tmp_path: Path) -> Tuple[Agent, ToolsmakerTool]:
    agent = Agent(
        {
            "stream_fn": _dummy_stream_fn,
            "toolsmaker_dir": str(tmp_path / "toolsmaker"),
            "toolsmaker_audit_path": str(tmp_path / "toolsmaker" / "audit.jsonl"),
            "project_root": str(tmp_path),
        }
    )
    tool = ToolsmakerTool(agent=agent, allowed_capabilities=["fs_read", "http"])
    return agent, tool


def _text(result: AgentToolResult) -> str:
    parts = [getattr(item, "text", "") for item in result.content if getattr(item, "type", "") == "text"]
    return "\n".join(parts)


def _safe_intent(name: str) -> Dict[str, Any]:
    return {
        "name": name,
        "purpose": "Read text files in an allowed folder",
        "inputs": {},
        "outputs": ["summary"],
        "capabilities": ["fs_read"],
        "allowed_paths": ["workspace"],
        "risk": "medium",
    }


def test_toolsmaker_create_returns_approval_required(tmp_path: Path):
    _, tool = _make_agent_and_tool(tmp_path)

    result = asyncio.run(
        tool.execute(
            "tc1",
            {
                "action": "create",
                "intent": _safe_intent("doc_reader"),
            },
        )
    )

    assert result.details["ok"] is True
    assert result.details["status"] == "approval_required"
    assert result.details["name"] == "doc_reader"
    assert result.details["version"] == 1
    assert result.details["review_path"]
    assert "status=approval_required" in _text(result)


def test_toolsmaker_blocks_disallowed_capabilities(tmp_path: Path):
    _, tool = _make_agent_and_tool(tmp_path)
    disallowed_intent = {
        "name": "writer_tool",
        "purpose": "Write files",
        "inputs": {},
        "outputs": ["result"],
        "capabilities": ["fs_write"],
        "allowed_paths": ["workspace"],
        "risk": "medium",
    }

    result = asyncio.run(tool.execute("tc1", {"action": "create", "intent": disallowed_intent}))

    assert result.details["ok"] is False
    assert result.details["error"] == "capability_not_allowed"
    assert result.details["disallowed"] == ["fs_write"]
    assert result.details["allowed"] == ["fs_read", "http"]
    assert "blocked" in _text(result).lower()


def test_toolsmaker_blocks_empty_capabilities_for_writer_intent(tmp_path: Path):
    _, tool = _make_agent_and_tool(tmp_path)
    no_caps_intent = {
        "name": "text_file_writer",
        "purpose": "Write text content to a file in workspace",
        "inputs": {},
        "outputs": ["result"],
        "capabilities": [],
        "allowed_paths": ["workspace"],
        "risk": "medium",
    }

    result = asyncio.run(tool.execute("tc1", {"action": "create", "intent": no_caps_intent}))

    assert result.details["ok"] is False
    assert result.details["error"] == "missing_capabilities"
    assert "no-op tool" in _text(result).lower()


def test_toolsmaker_blocks_placeholder_tool_name(tmp_path: Path):
    _, tool = _make_agent_and_tool(tmp_path)
    result = asyncio.run(
        tool.execute(
            "tc1",
            {
                "action": "create",
                "intent": _safe_intent("generated_tool"),
            },
        )
    )

    assert result.details["ok"] is False
    assert "meaningful" in str(result.details.get("error", "")).lower()


def test_toolsmaker_approve_activate_and_list(tmp_path: Path):
    agent, tool = _make_agent_and_tool(tmp_path)
    create = asyncio.run(tool.execute("tc1", {"action": "create", "intent": _safe_intent("flow_tool")}))
    name = str(create.details["name"])
    version = int(create.details["version"])

    approve = asyncio.run(tool.execute("tc2", {"action": "approve", "name": name, "version": version}))
    assert approve.details["ok"] is True
    assert approve.details["status"] == "approved"

    activate = asyncio.run(
        tool.execute(
            "tc3",
            {"action": "activate", "name": name, "version": version, "max_output_chars": 4096},
        )
    )
    assert activate.details["ok"] is True
    assert activate.details["status"] == "activated"

    listed = asyncio.run(tool.execute("tc4", {"action": "list"}))
    assert listed.details["ok"] is True
    assert name in listed.details["tools"]
    assert name in set(agent.list_tools())


def test_toolsmaker_reject_path(tmp_path: Path):
    _, tool = _make_agent_and_tool(tmp_path)
    create = asyncio.run(tool.execute("tc1", {"action": "create", "intent": _safe_intent("rejectable_tool")}))
    name = str(create.details["name"])
    version = int(create.details["version"])

    rejected = asyncio.run(
        tool.execute(
            "tc2",
            {
                "action": "reject",
                "name": name,
                "version": version,
                "reason": "not_needed",
            },
        )
    )

    assert rejected.details["ok"] is True
    assert rejected.details["status"] == "rejected"
    assert rejected.details["reason"] == "not_needed"
