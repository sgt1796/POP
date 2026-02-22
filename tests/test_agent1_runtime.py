from types import SimpleNamespace

from agent_build.agent1.runtime import build_runtime_tools


def _dummy_tool(name: str):
    return SimpleNamespace(name=name)


def test_build_runtime_tools_excludes_demo_tools_by_default():
    tools = build_runtime_tools(
        memory_search_tool=_dummy_tool("memory_search"),  # type: ignore[arg-type]
        toolsmaker_tool=_dummy_tool("toolsmaker"),  # type: ignore[arg-type]
        bash_exec_tool=_dummy_tool("bash_exec"),  # type: ignore[arg-type]
        gmail_fetch_tool=_dummy_tool("gmail_fetch"),  # type: ignore[arg-type]
        pdf_merge_tool=_dummy_tool("pdf_merge"),  # type: ignore[arg-type]
        include_demo_tools=False,
    )

    names = [tool.name for tool in tools]
    assert names == ["websnapshot", "memory_search", "toolsmaker", "bash_exec", "gmail_fetch", "pdf_merge"]


def test_build_runtime_tools_includes_demo_tools_when_enabled():
    tools = build_runtime_tools(
        memory_search_tool=_dummy_tool("memory_search"),  # type: ignore[arg-type]
        toolsmaker_tool=_dummy_tool("toolsmaker"),  # type: ignore[arg-type]
        bash_exec_tool=_dummy_tool("bash_exec"),  # type: ignore[arg-type]
        gmail_fetch_tool=_dummy_tool("gmail_fetch"),  # type: ignore[arg-type]
        pdf_merge_tool=_dummy_tool("pdf_merge"),  # type: ignore[arg-type]
        include_demo_tools=True,
    )

    names = [tool.name for tool in tools]
    assert names == ["websnapshot", "memory_search", "toolsmaker", "bash_exec", "gmail_fetch", "pdf_merge", "slow", "fast"]
