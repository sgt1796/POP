from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from agent.tools import BashExecConfig, BashExecTool


class _FakeProcess:
    def __init__(
        self,
        *,
        stdout: bytes = b"",
        stderr: bytes = b"",
        returncode: int = 0,
        hang_first_communicate_s: float = 0.0,
    ) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self._hang_first_communicate_s = hang_first_communicate_s
        self._communicate_calls = 0
        self.killed = False

    async def communicate(self):
        self._communicate_calls += 1
        if self._hang_first_communicate_s > 0 and self._communicate_calls == 1 and not self.killed:
            await asyncio.sleep(self._hang_first_communicate_s)
        return self._stdout, self._stderr

    def kill(self) -> None:
        self.killed = True
        self.returncode = self.returncode if self.returncode is not None else 1


def _make_tool(
    tmp_path: Path,
    *,
    approval_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
    default_timeout_s: float = 15.0,
    default_max_output_chars: int = 20_000,
    allowed_roots: Optional[list[str]] = None,
    writable_roots: Optional[list[str]] = None,
) -> BashExecTool:
    root = str(tmp_path)
    config = BashExecConfig(
        project_root=root,
        allowed_roots=allowed_roots or [root],
        writable_roots=writable_roots or [root],
        read_commands={"pwd", "ls", "cat", "head", "tail", "wc", "find", "rg", "git"},
        write_commands={"mkdir", "touch", "cp", "mv", "rm"},
        git_read_subcommands={"status", "diff", "log", "show", "branch"},
        default_timeout_s=default_timeout_s,
        max_timeout_s=60.0,
        default_max_output_chars=default_max_output_chars,
        max_output_chars_limit=100_000,
    )
    return BashExecTool(config=config, approval_fn=approval_fn)


def _run(tool: BashExecTool, params: Dict[str, Any]):
    return asyncio.run(tool.execute("tc1", params))


def test_read_command_runs_without_approval(tmp_path: Path, monkeypatch):
    def _approval(_: Dict[str, Any]) -> bool:
        raise AssertionError("read commands should not request approval")

    async def _fake_create_subprocess_exec(*argv, **kwargs):
        del argv, kwargs
        return _FakeProcess(stdout=b"/tmp/test\n", returncode=0)

    monkeypatch.setattr("agent.tools.bash_exec_tool.asyncio.create_subprocess_exec", _fake_create_subprocess_exec)
    tool = _make_tool(tmp_path, approval_fn=_approval)
    result = _run(tool, {"cmd": "pwd"})

    assert result.details["ok"] is True
    assert result.details["blocked"] is False
    assert result.details["risk"] == "low"
    assert result.details["approved"] is True
    assert result.details["exit_code"] == 0


def test_blocks_control_operators(tmp_path: Path):
    tool = _make_tool(tmp_path)
    result = _run(tool, {"cmd": "ls; pwd"})

    assert result.details["ok"] is False
    assert result.details["blocked"] is True
    assert result.details["block_reason"] == "blocked_shell_operator"


def test_blocks_unknown_command(tmp_path: Path):
    tool = _make_tool(tmp_path)
    result = _run(tool, {"cmd": "echo hello"})

    assert result.details["ok"] is False
    assert result.details["blocked"] is True
    assert result.details["block_reason"] == "command_not_allowed"


def test_blocks_path_outside_allowed_roots(tmp_path: Path):
    tool = _make_tool(tmp_path)
    result = _run(tool, {"cmd": "cat /etc/hosts"})

    assert result.details["ok"] is False
    assert result.details["blocked"] is True
    assert result.details["block_reason"] == "path_outside_allowed_roots"


def test_write_command_requires_approval_and_denies_on_reject(tmp_path: Path):
    tool = _make_tool(tmp_path, approval_fn=lambda _: False)
    target = tmp_path / "denied.txt"
    result = _run(tool, {"cmd": "touch denied.txt"})

    assert result.details["ok"] is False
    assert result.details["blocked"] is True
    assert result.details["risk"] == "medium"
    assert result.details["approved"] is False
    assert result.details["block_reason"] == "approval_required_or_denied"
    assert not target.exists()


def test_write_command_executes_when_approved(tmp_path: Path, monkeypatch):
    calls = []

    def _approval(request: Dict[str, Any]) -> bool:
        calls.append(request)
        return True

    async def _fake_create_subprocess_exec(*argv, **kwargs):
        cwd = Path(str(kwargs.get("cwd", "")))
        if len(argv) >= 2 and str(argv[0]) == "touch":
            (cwd / str(argv[1])).touch()
        return _FakeProcess(returncode=0)

    monkeypatch.setattr("agent.tools.bash_exec_tool.asyncio.create_subprocess_exec", _fake_create_subprocess_exec)
    tool = _make_tool(tmp_path, approval_fn=_approval)
    target = tmp_path / "approved.txt"
    result = _run(tool, {"cmd": "touch approved.txt", "justification": "needed for test"})

    assert result.details["blocked"] is False
    assert result.details["ok"] is True
    assert result.details["risk"] == "medium"
    assert result.details["approved"] is True
    assert target.exists()
    assert len(calls) == 1
    assert calls[0]["justification"] == "needed for test"


def test_rm_is_high_risk_and_approval_required(tmp_path: Path):
    tool = _make_tool(tmp_path, approval_fn=None)
    target = tmp_path / "delete_me.txt"
    target.write_text("x", encoding="utf-8")
    result = _run(tool, {"cmd": "rm delete_me.txt"})

    assert result.details["ok"] is False
    assert result.details["blocked"] is True
    assert result.details["risk"] == "high"
    assert result.details["approved"] is False
    assert result.details["block_reason"] == "approval_required_or_denied"
    assert target.exists()


def test_timeout_enforced(tmp_path: Path, monkeypatch):
    proc = _FakeProcess(stdout=b"", stderr=b"", returncode=124, hang_first_communicate_s=5.0)

    async def _fake_create_subprocess_exec(*argv, **kwargs):
        del argv, kwargs
        return proc

    monkeypatch.setattr("agent.tools.bash_exec_tool.asyncio.create_subprocess_exec", _fake_create_subprocess_exec)
    tool = _make_tool(tmp_path)
    target = tmp_path / "loop.txt"
    target.write_text("", encoding="utf-8")
    result = _run(tool, {"cmd": "tail -f loop.txt", "timeout_s": 1})

    assert result.details["ok"] is False
    assert result.details["blocked"] is False
    assert result.details["timed_out"] is True
    assert proc.killed is True
    assert result.details["exit_code"] is not None


def test_output_truncation_flagged(tmp_path: Path, monkeypatch):
    async def _fake_create_subprocess_exec(*argv, **kwargs):
        del argv, kwargs
        return _FakeProcess(stdout=("a" * 5000).encode("utf-8"), returncode=0)

    monkeypatch.setattr("agent.tools.bash_exec_tool.asyncio.create_subprocess_exec", _fake_create_subprocess_exec)
    tool = _make_tool(tmp_path)
    target = tmp_path / "big.txt"
    target.write_text("a" * 5000, encoding="utf-8")
    result = _run(tool, {"cmd": "cat big.txt", "max_output_chars": 256})

    assert result.details["ok"] is True
    assert result.details["blocked"] is False
    assert result.details["truncated"] is True
    assert len(result.details["stdout"]) + len(result.details["stderr"]) <= 256


def test_result_content_includes_stdout_and_stderr_by_default(tmp_path: Path, monkeypatch):
    async def _fake_create_subprocess_exec(*argv, **kwargs):
        del argv, kwargs
        return _FakeProcess(stdout=b"line1\nline2\n", stderr=b"warn\n", returncode=0)

    monkeypatch.setattr("agent.tools.bash_exec_tool.asyncio.create_subprocess_exec", _fake_create_subprocess_exec)
    tool = _make_tool(tmp_path)
    result = _run(tool, {"cmd": "pwd"})
    text = result.content[0].text

    assert "stdout:" in text
    assert "line1" in text
    assert "stderr:" in text
    assert "warn" in text
    assert result.details["result_detail"] == "detailed"


def test_result_detail_summary_returns_summary_only(tmp_path: Path, monkeypatch):
    async def _fake_create_subprocess_exec(*argv, **kwargs):
        del argv, kwargs
        return _FakeProcess(stdout=b"abc\n", stderr=b"", returncode=0)

    monkeypatch.setattr("agent.tools.bash_exec_tool.asyncio.create_subprocess_exec", _fake_create_subprocess_exec)
    tool = _make_tool(tmp_path)
    result = _run(tool, {"cmd": "pwd", "result_detail": "summary"})
    text = result.content[0].text

    assert text.strip() == "bash_exec exit_code=0"
    assert "stdout:" not in text
    assert result.details["result_detail"] == "summary"


def test_per_stream_output_caps_are_respected(tmp_path: Path, monkeypatch):
    async def _fake_create_subprocess_exec(*argv, **kwargs):
        del argv, kwargs
        return _FakeProcess(stdout=b"abcdefghij", stderr=b"0123456789", returncode=0)

    monkeypatch.setattr("agent.tools.bash_exec_tool.asyncio.create_subprocess_exec", _fake_create_subprocess_exec)
    tool = _make_tool(tmp_path)
    result = _run(
        tool,
        {
            "cmd": "pwd",
            "stdout_max_chars": 4,
            "stderr_max_chars": 3,
            "max_output_chars": 256,
        },
    )

    assert result.details["stdout"] == "abcd"
    assert result.details["stderr"] == "012"
    assert result.details["stdout_truncated"] is True
    assert result.details["stderr_truncated"] is True
    assert result.details["combined_truncated"] is False


def test_git_allows_only_read_subcommands(tmp_path: Path):
    tool = _make_tool(tmp_path)

    blocked = _run(tool, {"cmd": "git init"})
    assert blocked.details["ok"] is False
    assert blocked.details["blocked"] is True
    assert blocked.details["block_reason"] == "git_subcommand_not_allowed"

    allowed = _run(tool, {"cmd": "git status"})
    assert allowed.details["blocked"] is False
    assert allowed.details["risk"] == "low"


def test_ls_uses_safe_fallback_when_command_missing(tmp_path: Path, monkeypatch):
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "subdir").mkdir()

    async def _raise_not_found(*argv, **kwargs):
        del argv, kwargs
        raise FileNotFoundError("ls not found")

    monkeypatch.setattr("agent.tools.bash_exec_tool.asyncio.create_subprocess_exec", _raise_not_found)
    tool = _make_tool(tmp_path)
    result = _run(tool, {"cmd": "ls -F"})

    assert result.details["blocked"] is False
    assert result.details["ok"] is True
    assert "a.txt" in result.details["stdout"]
    assert "subdir/" in result.details["stdout"]
