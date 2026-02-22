from typing import List


VALID_EXECUTION_PROFILES = {"balanced", "aggressive", "conservative"}


def resolve_execution_profile(value: str) -> str:
    profile = str(value or "").strip().lower()
    if profile in VALID_EXECUTION_PROFILES:
        return profile
    return "balanced"


def _profile_guidance(profile: str) -> str:
    if profile == "aggressive":
        return (
            "Execution profile: aggressive. Bias heavily toward immediate tool use and autonomous retries "
            "before asking the user."
        )
    if profile == "conservative":
        return (
            "Execution profile: conservative. Prefer safe, explicit confirmations for uncertain or high-impact "
            "actions."
        )
    return (
        "Execution profile: balanced. Be execution-first with tools while preserving guardrails and concise "
        "user confirmations only when unavoidable."
    )


def build_system_prompt(
    *,
    bash_read_csv: str,
    bash_write_csv: str,
    bash_git_csv: str,
    bash_prompt_approval: bool,
    toolsmaker_manual_approval: bool,
    toolsmaker_auto_continue: bool,
    execution_profile: str = "balanced",
    workspace_root: str = "",
) -> str:
    profile = resolve_execution_profile(execution_profile)
    lines: List[str] = []
    lines.append("Mission:")
    lines.append("Execute user requests end-to-end whenever feasible and produce concrete outcomes.")
    lines.append(_profile_guidance(profile))
    lines.append("")
    lines.append("Tool Policy:")
    lines.append("Prefer existing tools first before explaining limitations.")
    lines.append("Use bash_exec for allowed shell/filesystem inspection or edits within policy.")
    lines.append("Never call bash_exec with commands or subcommands outside allowlists.")
    lines.append(f"Allowed bash_exec read commands: {bash_read_csv}.")
    lines.append(f"Allowed bash_exec write commands: {bash_write_csv}.")
    lines.append(f"Allowed bash_exec git subcommands: {bash_git_csv}.")
    if bash_prompt_approval:
        lines.append("Write commands in bash_exec require user approval.")
    else:
        lines.append("bash_exec approvals are disabled; medium/high-risk write commands are denied.")
    if workspace_root:
        lines.append(f"Default workspace root: {workspace_root}.")
    lines.append("")
    lines.append("Missing Capability Flow:")
    lines.append("If existing tools are insufficient, use toolsmaker to create the minimal-capability tool.")
    lines.append("When calling toolsmaker create, include intent.capabilities.")
    lines.append("fs_read/fs_write require allowed_paths; http requires allowed_domains.")
    lines.append("")
    lines.append("Tool Lifecycle:")
    lines.append("Follow create -> approve -> activate before using generated tools.")
    if toolsmaker_manual_approval:
        lines.append("Manual toolsmaker approvals are enabled; ask only for the approval gate when required.")
    elif toolsmaker_auto_continue:
        lines.append(
            "Manual toolsmaker approvals are disabled; runtime auto-continues create results by approving and "
            "activating tool versions."
        )
    else:
        lines.append(
            "Manual toolsmaker approvals are disabled and runtime auto-continue is off; you must explicitly call "
            "toolsmaker approve and activate after create."
        )
    lines.append("")
    lines.append("Failure Recovery:")
    lines.append("If a tool call fails or is blocked, inspect error details, fix arguments, and retry.")
    lines.append("If progress is impossible due to a hard policy gate, ask one focused question for the missing input.")
    lines.append("")
    lines.append("Completion Criteria:")
    lines.append("When done, provide a brief result summary and include artifact paths, tool outputs, or next actions.")
    return "\n".join(lines)
