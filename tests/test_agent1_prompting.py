from agent_build.agent1.prompting import build_system_prompt, resolve_execution_profile


def _build_prompt(**overrides):
    params = {
        "bash_read_csv": "cat, ls",
        "bash_write_csv": "touch",
        "bash_git_csv": "status",
        "bash_prompt_approval": True,
        "toolsmaker_manual_approval": True,
        "toolsmaker_auto_continue": True,
        "execution_profile": "balanced",
        "workspace_root": "/tmp/workspace",
    }
    params.update(overrides)
    return build_system_prompt(**params)


def test_prompt_includes_execution_first_sections_and_allowlists():
    prompt = _build_prompt()

    assert "Mission:" in prompt
    assert "Execute user requests end-to-end whenever feasible" in prompt
    assert "Tool Policy:" in prompt
    assert "Allowed bash_exec read commands: cat, ls." in prompt
    assert "Allowed bash_exec write commands: touch." in prompt
    assert "Allowed bash_exec git subcommands: status." in prompt
    assert "Never call bash_exec with commands or subcommands outside allowlists." in prompt
    assert "Failure Recovery:" in prompt
    assert "Completion Criteria:" in prompt


def test_prompt_mode_dependent_lifecycle_lines():
    manual_prompt = _build_prompt(toolsmaker_manual_approval=True)
    assert "Manual toolsmaker approvals are enabled" in manual_prompt

    auto_continue_prompt = _build_prompt(toolsmaker_manual_approval=False, toolsmaker_auto_continue=True)
    assert "runtime auto-continues create results by approving and activating tool versions" in auto_continue_prompt

    llm_lifecycle_prompt = _build_prompt(toolsmaker_manual_approval=False, toolsmaker_auto_continue=False)
    assert "you must explicitly call toolsmaker approve and activate after create" in llm_lifecycle_prompt


def test_prompt_fallbacks_to_balanced_profile_and_denied_write_hint():
    prompt = _build_prompt(execution_profile="not-a-real-profile", bash_prompt_approval=False)

    assert resolve_execution_profile("not-a-real-profile") == "balanced"
    assert "Execution profile: balanced." in prompt
    assert "medium/high-risk write commands are denied" in prompt
