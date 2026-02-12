from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict

import pytest

from agent import Agent
from agent.agent_loop import _execute_tool_calls
from agent.agent_types import ToolBuildRequest, ToolCallContent
from agent.toolsmaker.policy import ToolPolicyViolation
from agent.toolsmaker.registry import ToolsmakerRegistry
from agent.tools import FastTool, SlowTool, WebSnapshotTool


def _request(
    name: str,
    purpose: str,
    *,
    capabilities: list[str] | None = None,
    allowed_paths: list[str] | None = None,
    allowed_domains: list[str] | None = None,
    timeout_s: float = 30.0,
) -> ToolBuildRequest:
    return ToolBuildRequest(
        name=name,
        purpose=purpose,
        inputs={},
        outputs=["summary"],
        capabilities=capabilities or [],
        risk="medium",
        allowed_paths=allowed_paths or [],
        allowed_domains=allowed_domains or [],
        required_secrets=[],
        timeout_s=timeout_s,
    )


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def test_generated_tool_activation_requires_approval(tmp_path: Path):
    registry = ToolsmakerRegistry(
        base_dir=str(tmp_path / "toolsmaker"),
        project_root=str(tmp_path),
        audit_path=str(tmp_path / "toolsmaker" / "audit.jsonl"),
    )
    request = _request(
        "file_helper",
        "Read and write local files",
        capabilities=["fs_write"],
        allowed_paths=["workspace"],
    )
    result = registry.build_tool(request)
    assert result.status == "approval_required"

    with pytest.raises(ValueError):
        registry.activate_tool_version(result.spec.name, result.spec.version)

    approved = registry.approve_tool(result.spec.name, result.spec.version)
    assert approved.status == "approved"
    activated = registry.activate_tool_version(result.spec.name, result.spec.version)
    assert activated.name == result.spec.name


def test_forbidden_import_is_rejected_before_approval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    registry = ToolsmakerRegistry(
        base_dir=str(tmp_path / "toolsmaker"),
        project_root=str(tmp_path),
        audit_path=str(tmp_path / "toolsmaker" / "audit.jsonl"),
    )
    monkeypatch.setattr(
        registry._builder,  # type: ignore[attr-defined]
        "render_code",
        lambda spec, request: (
            "import subprocess\n"
            "from agent.toolsmaker.policy import GeneratedToolBase\n"
            "class GeneratedTool(GeneratedToolBase):\n"
            "    pass\n"
        ),
    )
    request = _request("bad_tool", "Attempt forbidden import")
    result = registry.build_tool(request)
    assert result.status == "rejected"
    assert result.validation["code"]["ok"] is False

    with pytest.raises(ValueError):
        registry.approve_tool(result.spec.name, result.spec.version)


def test_policy_blocks_path_outside_allowlist(tmp_path: Path):
    registry = ToolsmakerRegistry(
        base_dir=str(tmp_path / "toolsmaker"),
        project_root=str(tmp_path),
        audit_path=str(tmp_path / "toolsmaker" / "audit.jsonl"),
    )
    request = _request(
        "writer_tool",
        "Write files in an allowed folder",
        capabilities=["fs_write"],
        allowed_paths=["allowed"],
    )
    build = registry.build_tool(request)
    registry.approve_tool(build.spec.name, build.spec.version)
    tool = registry.activate_tool_version(build.spec.name, build.spec.version)

    asyncio.run(
        tool.execute(
            "t1",
            {"write_path": "allowed/ok.txt", "write_content": "hello"},
        )
    )
    assert (tmp_path / "allowed" / "ok.txt").exists()

    with pytest.raises(ToolPolicyViolation):
        asyncio.run(
            tool.execute(
                "t2",
                {"write_path": "outside/nope.txt", "write_content": "deny"},
            )
        )


def test_policy_blocks_non_whitelisted_domain(tmp_path: Path):
    registry = ToolsmakerRegistry(
        base_dir=str(tmp_path / "toolsmaker"),
        project_root=str(tmp_path),
        audit_path=str(tmp_path / "toolsmaker" / "audit.jsonl"),
    )
    request = _request(
        "http_tool",
        "Fetch approved domains",
        capabilities=["http"],
        allowed_domains=["allowed.example.com"],
    )
    build = registry.build_tool(request)
    registry.approve_tool(build.spec.name, build.spec.version)
    tool = registry.activate_tool_version(build.spec.name, build.spec.version)

    with pytest.raises(ToolPolicyViolation):
        asyncio.run(tool.execute("t1", {"url": "https://blocked.example.com/data"}))


def test_fs_write_supports_file_path_and_content_aliases(tmp_path: Path):
    registry = ToolsmakerRegistry(
        base_dir=str(tmp_path / "toolsmaker"),
        project_root=str(tmp_path),
        audit_path=str(tmp_path / "toolsmaker" / "audit.jsonl"),
    )
    request = _request(
        "alias_writer",
        "Write files in an allowed folder",
        capabilities=["fs_write"],
        allowed_paths=["allowed"],
    )
    build = registry.build_tool(request)
    registry.approve_tool(build.spec.name, build.spec.version)
    tool = registry.activate_tool_version(build.spec.name, build.spec.version)

    asyncio.run(
        tool.execute(
            "t1",
            {"file_path": "allowed/alias.txt", "content": "hello alias"},
        )
    )

    target = tmp_path / "allowed" / "alias.txt"
    assert target.exists()
    assert target.read_text(encoding="utf-8") == "hello alias"


def test_tool_timeout_is_enforced(tmp_path: Path):
    registry = ToolsmakerRegistry(
        base_dir=str(tmp_path / "toolsmaker"),
        project_root=str(tmp_path),
        audit_path=str(tmp_path / "toolsmaker" / "audit.jsonl"),
    )
    request = _request("slow_generated", "Delay tool", timeout_s=0.05)
    build = registry.build_tool(request)
    registry.approve_tool(build.spec.name, build.spec.version)
    tool = registry.activate_tool_version(build.spec.name, build.spec.version)

    with pytest.raises(RuntimeError, match="timed out"):
        asyncio.run(tool.execute("t1", {"delay_s": 0.2}))


def test_registry_hot_activation_and_listing(tmp_path: Path):
    registry = ToolsmakerRegistry(
        base_dir=str(tmp_path / "toolsmaker"),
        project_root=str(tmp_path),
        audit_path=str(tmp_path / "toolsmaker" / "audit.jsonl"),
    )
    request_v1 = _request("hot_tool", "v1 writer", capabilities=["fs_write"], allowed_paths=["allowed"])
    build_v1 = registry.build_tool(request_v1)
    registry.approve_tool(build_v1.spec.name, build_v1.spec.version)
    registry.activate_tool_version(build_v1.spec.name, build_v1.spec.version)

    request_v2 = _request("hot_tool", "v2 writer", capabilities=["fs_write"], allowed_paths=["allowed"])
    build_v2 = registry.build_tool(request_v2)
    registry.approve_tool(build_v2.spec.name, build_v2.spec.version)
    registry.activate_tool_version(build_v2.spec.name, build_v2.spec.version)

    assert "hot_tool" in registry.list_tools()
    record_v1 = registry.get_record("hot_tool", 1)
    record_v2 = registry.get_record("hot_tool", 2)
    assert record_v1 is not None and record_v1.status == "approved"
    assert record_v2 is not None and record_v2.status == "activated"


def test_static_tools_remain_compatible(tmp_path: Path):
    agent = Agent(
        {
            "stream_fn": _dummy_stream_fn,
            "toolsmaker_dir": str(tmp_path / "toolsmaker"),
            "toolsmaker_audit_path": str(tmp_path / "toolsmaker" / "audit.jsonl"),
            "project_root": str(tmp_path),
        }
    )
    agent.set_tools([SlowTool(), FastTool(), WebSnapshotTool()])
    names = set(agent.list_tools())
    assert {"slow", "fast", "websnapshot"}.issubset(names)

    removed = agent.remove_tool("fast")
    assert removed is True
    assert "fast" not in set(agent.list_tools())


def test_audit_events_and_policy_blocked_event(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    audit_path = tmp_path / "agent" / "toolsmaker" / "audit.jsonl"
    registry = ToolsmakerRegistry(
        base_dir=str(tmp_path / "agent" / "toolsmaker"),
        project_root=str(tmp_path),
        audit_path=str(audit_path),
    )
    request = _request(
        "audit_tool",
        "Write files in allowed folder",
        capabilities=["fs_write"],
        allowed_paths=["allowed"],
    )
    build = registry.build_tool(request)
    registry.approve_tool(build.spec.name, build.spec.version)
    tool = registry.activate_tool_version(build.spec.name, build.spec.version)

    class CaptureStream:
        def __init__(self) -> None:
            self.events: list[Dict[str, Any]] = []

        def push(self, event: Dict[str, Any]) -> None:
            self.events.append(event)

    stream = CaptureStream()
    tool_call = ToolCallContent(
        type="toolCall",
        id="tc1",
        name=tool.name,
        arguments={"write_path": "outside/nope.txt", "write_content": "blocked"},
    )
    asyncio.run(_execute_tool_calls([tool], [tool_call], signal=None, stream=stream, get_steering_messages=None))

    event_types = [event["type"] for event in stream.events]
    assert "tool_policy_blocked" in event_types

    audit_events = _read_jsonl(audit_path)
    audit_types = [item["type"] for item in audit_events]
    assert audit_types[:5] == [
        "tool_build_requested",
        "tool_build_generated",
        "tool_build_validated",
        "tool_approval_required",
        "tool_activated",
    ]
    assert "tool_policy_blocked" in audit_types
