from types import SimpleNamespace

from agent_build.agent1.approvals import ToolsmakerAutoContinueSubscriber


class _FakeActivatedTool:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeAgent:
    def __init__(self, *, fail_approve: bool = False, fail_activate: bool = False) -> None:
        self.fail_approve = fail_approve
        self.fail_activate = fail_activate
        self.approve_calls = []
        self.activate_calls = []

    def approve_dynamic_tool(self, name: str, version: int):
        self.approve_calls.append((name, version))
        if self.fail_approve:
            raise RuntimeError("approve failed")
        return SimpleNamespace(status="approved")

    def activate_tool_version(self, name: str, version: int):
        self.activate_calls.append((name, version))
        if self.fail_activate:
            raise RuntimeError("activate failed")
        return _FakeActivatedTool(name=name)


def _event(
    *,
    etype: str = "tool_execution_end",
    tool_name: str = "toolsmaker",
    action: str = "create",
    status: str = "approval_required",
    ok: bool = True,
    name: str = "gmail_fetcher",
    version: int = 1,
    result_as_dict: bool = False,
):
    details = {
        "ok": ok,
        "action": action,
        "status": status,
        "name": name,
        "version": version,
    }
    result = {"details": details} if result_as_dict else SimpleNamespace(details=details)
    return {"type": etype, "toolName": tool_name, "result": result}


def test_auto_continue_approves_and_activates_once():
    agent = _FakeAgent()
    subscriber = ToolsmakerAutoContinueSubscriber(agent=agent)  # type: ignore[arg-type]
    event = _event(name="pdf_merger", version=3)

    subscriber.on_event(event)
    subscriber.on_event(event)

    assert agent.approve_calls == [("pdf_merger", 3)]
    assert agent.activate_calls == [("pdf_merger", 3)]


def test_auto_continue_ignores_non_qualifying_events():
    agent = _FakeAgent()
    subscriber = ToolsmakerAutoContinueSubscriber(agent=agent)  # type: ignore[arg-type]

    subscriber.on_event(_event(etype="message_end"))
    subscriber.on_event(_event(tool_name="bash_exec"))
    subscriber.on_event(_event(action="approve"))
    subscriber.on_event(_event(status="approved"))
    subscriber.on_event(_event(ok=False))

    assert agent.approve_calls == []
    assert agent.activate_calls == []


def test_auto_continue_handles_result_dict_details():
    agent = _FakeAgent()
    subscriber = ToolsmakerAutoContinueSubscriber(agent=agent)  # type: ignore[arg-type]

    subscriber.on_event(_event(name="doc_tool", version=2, result_as_dict=True))

    assert agent.approve_calls == [("doc_tool", 2)]
    assert agent.activate_calls == [("doc_tool", 2)]


def test_auto_continue_handles_agent_exceptions_without_crashing():
    agent = _FakeAgent(fail_approve=True, fail_activate=True)
    subscriber = ToolsmakerAutoContinueSubscriber(agent=agent)  # type: ignore[arg-type]

    subscriber.on_event(_event(name="unstable_tool", version=5))

    assert agent.approve_calls == [("unstable_tool", 5)]
    assert agent.activate_calls == [("unstable_tool", 5)]
