# Toolsmaker

`toolsmaker` is the agent subsystem that creates, validates, approves, stores, and activates generated tools at runtime.

It is designed for a human-gated workflow:
1. Agent proposes/builds a tool.
2. System validates it.
3. Human approves or rejects it.
4. Approved version is activated and callable by the agent.

## What It Solves

- Lets the agent add new tools without redeploying code.
- Keeps generated tools constrained by policy (filesystem, HTTP, secrets).
- Persists tool specs, generated code, and audit trail on disk.

## Main Components

- `registry.py`
  - `ToolsmakerRegistry` is the orchestrator for build/approve/activate flows.
  - Manages static tools and dynamic generated tools in one runtime registry.
- `builder.py`
  - Builds `ToolBuildRequest` from structured intent.
  - Converts request to canonical `ToolSpec`.
  - Renders deterministic Python tool code from template.
- `validator.py`
  - Validates tool spec fields and capabilities.
  - AST checks generated code for forbidden imports/calls and interface shape.
- `approvals.py`
  - Enforces status transitions:
    - `draft -> validated -> approval_required -> approved|rejected -> activated`
  - Writes review artifact JSON.
- `loader.py`
  - Dynamically loads generated modules and returns a tool instance.
- `policy.py`
  - Runtime policy enforcement:
    - Filesystem path allowlists.
    - HTTP domain allowlists.
    - Declared secret access only.
  - Timeout and max output size guard wrapper (`PolicyGuardedTool`).

## Agent Integration

`Agent` wires this subsystem in `agent/agent.py`.

Important `Agent` options:
- `toolsmaker_dir` (default: `agent/toolsmaker`)
- `toolsmaker_audit_path` (default: `agent/toolsmaker/audit.jsonl`)

Important `Agent` methods:
- Tool lifecycle:
  - `set_tools(...)`, `add_tool(...)`, `remove_tool(...)`, `list_tools(...)`
  - `activate_tool_version(name, version, max_output_chars=...)`
- Toolsmaker flow:
  - `create_tool_build_request_from_intent(intent)`
  - `build_dynamic_tool(request)`
  - `build_dynamic_tool_from_intent(intent)`
  - `approve_dynamic_tool(name, version)`
  - `reject_dynamic_tool(name, version, reason=...)`

Example:

```python
from agent import Agent

agent = Agent({
    "stream_fn": my_stream_fn,
    "toolsmaker_dir": "agent/toolsmaker",
    "toolsmaker_audit_path": "agent/toolsmaker/audit.jsonl",
})

intent = {
    "name": "file_writer",
    "purpose": "Write generated reports",
    "inputs": {},
    "outputs": ["result"],
    "capabilities": ["fs_write"],
    "allowed_paths": ["reports"],
    "risk": "medium",
}

result = agent.build_dynamic_tool_from_intent(intent)
if result.status == "approval_required":
    # Review result.review_path and generated code.
    agent.approve_dynamic_tool(result.spec.name, result.spec.version)
    agent.activate_tool_version(result.spec.name, result.spec.version)
```

## On-Disk Layout

Under `toolsmaker_dir` (default `agent/toolsmaker`):

- `specs/`
  - Persisted build records with `ToolSpec`, status, validation, request metadata.
- `generated/`
  - Generated Python tool source files (`<name>_v<version>.py`).
- `reviews/`
  - Human review artifacts (summary, diff, validation report, code path).
- `active_versions.json`
  - Map of tool name to currently active version.
- `audit.jsonl`
  - Append-only audit/events log.

## Safety Model

### Static checks

`validator.py` rejects generated code if it contains:
- Forbidden imports (`subprocess`, `socket`, `ctypes`, `importlib`, `pathlib`).
- Forbidden calls (`eval`, `exec`, `__import__`).
- Missing `GeneratedTool` class or missing `execute(...)`.

### Runtime policy

`policy.py` enforces at execution time:
- Capability must be declared in spec (`fs_read`, `fs_write`, `http`, `secrets`).
- Files must stay inside allowed paths.
- URLs must match allowed domains.
- Secrets must be declared in spec and present in env.
- Tool run timeout and output truncation are enforced.

Policy violations raise `ToolPolicyViolation` and produce `tool_policy_blocked` events.

## Events and Audit

Core events written to audit log:
- `tool_build_requested`
- `tool_build_generated`
- `tool_build_validated`
- `tool_approval_required`
- `tool_activated`
- `tool_build_rejected`
- `tool_policy_blocked`

## Notes for Maintainers

- `ToolsmakerRegistry` is the only class that should mutate toolsmaker state.
- `Agent` snapshots registry tools per turn so active tools remain stable during a run.
- Existing method names like `build_dynamic_tool(...)` are retained in `Agent` API even though module name is now `toolsmaker`.
