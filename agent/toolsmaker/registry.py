from __future__ import annotations

import dataclasses
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from agent.agent_types import AgentTool, ToolBuildRequest, ToolBuildResult, ToolPolicy, ToolSpec
from agent.toolsmaker.approvals import ApprovalStateMachine, write_review_artifact
from agent.toolsmaker.builder import ToolBuilder, sanitize_tool_name
from agent.toolsmaker.loader import ToolLoader
from agent.toolsmaker.policy import PolicyGuardedTool, ToolPolicyEnforcer
from agent.toolsmaker.validator import validate_generated_code, validate_tool_spec


DEFAULT_TOOLSMAKER_DIR = os.path.join("agent", "toolsmaker")
DEFAULT_SPECS_DIR = os.path.join(DEFAULT_TOOLSMAKER_DIR, "specs")
DEFAULT_GENERATED_DIR = os.path.join(DEFAULT_TOOLSMAKER_DIR, "generated")
DEFAULT_REVIEWS_DIR = os.path.join(DEFAULT_TOOLSMAKER_DIR, "reviews")
DEFAULT_AUDIT_PATH = os.path.join(DEFAULT_TOOLSMAKER_DIR, "audit.jsonl")
DEFAULT_ACTIVE_STATE = os.path.join(DEFAULT_TOOLSMAKER_DIR, "active_versions.json")
_DEFAULT_AUDIT_PATH_OVERRIDE: Optional[str] = None

AUDITABLE_EVENTS = {
    "tool_build_requested",
    "tool_build_generated",
    "tool_build_validated",
    "tool_approval_required",
    "tool_activated",
    "tool_build_rejected",
    "tool_policy_blocked",
}


def set_default_audit_path(audit_path: str) -> None:
    global _DEFAULT_AUDIT_PATH_OVERRIDE
    _DEFAULT_AUDIT_PATH_OVERRIDE = audit_path


def append_audit_event(event: Dict[str, Any], audit_path: Optional[str] = None) -> None:
    target_path = audit_path or _DEFAULT_AUDIT_PATH_OVERRIDE or DEFAULT_AUDIT_PATH
    os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)
    payload = dict(event)
    payload.setdefault("timestamp", time.time())
    with open(target_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


@dataclass
class DynamicToolRecord:
    spec: ToolSpec
    status: str
    code_path: str
    spec_path: str
    review_path: str
    validation: Dict[str, Any]
    request: ToolBuildRequest
    active_tool: Optional[AgentTool] = None


class ToolsmakerRegistry:
    """Registry for static and generated tools with approval workflow."""

    def __init__(
        self,
        base_dir: str = DEFAULT_TOOLSMAKER_DIR,
        project_root: Optional[str] = None,
        event_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
        audit_path: str = DEFAULT_AUDIT_PATH,
    ) -> None:
        self.base_dir = base_dir
        self.specs_dir = os.path.join(base_dir, "specs")
        self.generated_dir = os.path.join(base_dir, "generated")
        self.reviews_dir = os.path.join(base_dir, "reviews")
        self.active_state_path = os.path.join(base_dir, "active_versions.json")
        self.audit_path = audit_path
        set_default_audit_path(self.audit_path)
        os.makedirs(self.specs_dir, exist_ok=True)
        os.makedirs(self.generated_dir, exist_ok=True)
        os.makedirs(self.reviews_dir, exist_ok=True)

        self._event_sink = event_sink
        self._state_machine = ApprovalStateMachine()
        self._builder = ToolBuilder(self.generated_dir)
        self._loader = ToolLoader()
        self._policy_enforcer = ToolPolicyEnforcer(project_root=project_root)
        self._static_tools: Dict[str, AgentTool] = {}
        self._dynamic: Dict[str, Dict[int, DynamicToolRecord]] = {}
        self._active_versions: Dict[str, int] = {}
        self._load_records()
        self._load_active_versions()
        self._restore_active_tools()

    def set_event_sink(self, sink: Optional[Callable[[Dict[str, Any]], None]]) -> None:
        self._event_sink = sink

    def _emit(self, event_type: str, **payload: Any) -> None:
        event = {"type": event_type, **payload}
        if self._event_sink:
            try:
                self._event_sink(event)
            except Exception:
                pass
        if event_type in AUDITABLE_EVENTS:
            append_audit_event(event, audit_path=self.audit_path)

    def _spec_filename(self, name: str, version: int) -> str:
        return os.path.join(self.specs_dir, f"{name}_v{version}.json")

    def _review_filename(self, name: str, version: int) -> str:
        return os.path.join(self.reviews_dir, f"{name}_v{version}.json")

    def _spec_to_dict(self, spec: ToolSpec) -> Dict[str, Any]:
        return {
            "name": spec.name,
            "description": spec.description,
            "json_schema_parameters": spec.json_schema_parameters,
            "capabilities": list(spec.capabilities),
            "allowed_paths": list(spec.allowed_paths),
            "allowed_domains": list(spec.allowed_domains),
            "required_secrets": list(spec.required_secrets),
            "timeout_s": spec.timeout_s,
            "version": spec.version,
            "created_at": spec.created_at,
        }

    def _spec_from_dict(self, data: Dict[str, Any]) -> ToolSpec:
        return ToolSpec(
            name=str(data["name"]),
            description=str(data.get("description", "")),
            json_schema_parameters=dict(data.get("json_schema_parameters") or {}),
            capabilities=[str(x) for x in data.get("capabilities", [])],
            allowed_paths=[str(x) for x in data.get("allowed_paths", [])],
            allowed_domains=[str(x) for x in data.get("allowed_domains", [])],
            required_secrets=[str(x) for x in data.get("required_secrets", [])],
            timeout_s=float(data.get("timeout_s", 30.0) or 30.0),
            version=int(data.get("version", 1)),
            created_at=float(data.get("created_at", time.time())),
        )

    def _save_record(self, record: DynamicToolRecord) -> None:
        payload = {
            "spec": self._spec_to_dict(record.spec),
            "status": record.status,
            "code_path": record.code_path,
            "review_path": record.review_path,
            "validation": record.validation,
            "request": dataclasses.asdict(record.request),
        }
        os.makedirs(os.path.dirname(record.spec_path) or ".", exist_ok=True)
        with open(record.spec_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)

    def _load_records(self) -> None:
        for filename in os.listdir(self.specs_dir):
            if not filename.endswith(".json"):
                continue
            path = os.path.join(self.specs_dir, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                spec = self._spec_from_dict(payload.get("spec") or payload)
                request_payload = payload.get("request") or {
                    "name": spec.name,
                    "purpose": spec.description,
                    "inputs": spec.json_schema_parameters,
                    "outputs": [],
                    "capabilities": list(spec.capabilities),
                    "risk": "unknown",
                    "allowed_paths": list(spec.allowed_paths),
                    "allowed_domains": list(spec.allowed_domains),
                    "required_secrets": list(spec.required_secrets),
                    "timeout_s": spec.timeout_s,
                }
                request = ToolBuildRequest(**request_payload)
                record = DynamicToolRecord(
                    spec=spec,
                    status=str(payload.get("status", "draft")),
                    code_path=str(payload.get("code_path") or self._builder.code_path_for(spec)),
                    spec_path=path,
                    review_path=str(payload.get("review_path") or self._review_filename(spec.name, spec.version)),
                    validation=dict(payload.get("validation") or {}),
                    request=request,
                )
                self._dynamic.setdefault(spec.name, {})[spec.version] = record
            except Exception:
                continue

    def _save_active_versions(self) -> None:
        os.makedirs(os.path.dirname(self.active_state_path) or ".", exist_ok=True)
        with open(self.active_state_path, "w", encoding="utf-8") as f:
            json.dump(self._active_versions, f, ensure_ascii=False, indent=2, sort_keys=True)

    def _load_active_versions(self) -> None:
        if not os.path.exists(self.active_state_path):
            return
        try:
            with open(self.active_state_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                self._active_versions = {str(k): int(v) for k, v in payload.items()}
        except Exception:
            self._active_versions = {}

    def _restore_active_tools(self) -> None:
        for name, version in list(self._active_versions.items()):
            record = self._dynamic.get(name, {}).get(version)
            if record is None:
                self._active_versions.pop(name, None)
                continue
            if record.status not in {"approved", "activated"}:
                self._active_versions.pop(name, None)
                continue
            try:
                loaded = self._loader.load_tool(record.code_path, record.spec, self._policy_enforcer)
                policy = ToolPolicy(
                    capabilities=list(record.spec.capabilities),
                    allowed_paths=list(record.spec.allowed_paths),
                    allowed_domains=list(record.spec.allowed_domains),
                    required_secrets=list(record.spec.required_secrets),
                    timeout_s=record.spec.timeout_s,
                )
                record.active_tool = PolicyGuardedTool(loaded, record.spec, policy)
                record.status = "activated"
                self._save_record(record)
            except Exception:
                self._active_versions.pop(name, None)
        self._save_active_versions()

    def _next_version(self, name: str) -> int:
        versions = self._dynamic.get(name, {})
        if not versions:
            return 1
        return max(versions.keys()) + 1

    def _policy_diff(self, name: str, version: int, spec: ToolSpec) -> Dict[str, Any]:
        previous = self._dynamic.get(name, {}).get(version - 1)
        if previous is None:
            return {"type": "initial_version"}
        prev_spec = previous.spec
        return {
            "capabilities": {
                "before": sorted(prev_spec.capabilities),
                "after": sorted(spec.capabilities),
            },
            "allowed_paths": {"before": prev_spec.allowed_paths, "after": spec.allowed_paths},
            "allowed_domains": {"before": prev_spec.allowed_domains, "after": spec.allowed_domains},
            "required_secrets": {"before": prev_spec.required_secrets, "after": spec.required_secrets},
            "timeout_s": {"before": prev_spec.timeout_s, "after": spec.timeout_s},
        }

    def replace_static_tools(self, tools: Sequence[AgentTool]) -> None:
        self._static_tools = {getattr(t, "name", f"tool_{i}"): t for i, t in enumerate(tools)}

    def add_static_tool(self, tool: AgentTool) -> None:
        self._static_tools[tool.name] = tool

    def remove_tool(self, name: str) -> bool:
        removed = False
        if name in self._static_tools:
            self._static_tools.pop(name, None)
            removed = True
        if name in self._dynamic:
            self._dynamic.pop(name, None)
            removed = True
        if name in self._active_versions:
            self._active_versions.pop(name, None)
            self._save_active_versions()
            removed = True
        return removed

    def list_tools(self) -> List[str]:
        names = set(self._static_tools.keys())
        names.update(self._active_versions.keys())
        return sorted(names)

    def snapshot_tools(self) -> List[AgentTool]:
        result: List[AgentTool] = list(self._static_tools.values())
        for name, version in sorted(self._active_versions.items()):
            record = self._dynamic.get(name, {}).get(version)
            if record and record.active_tool is not None:
                result.append(record.active_tool)
        return result

    def create_build_request_from_intent(self, intent: Dict[str, Any]) -> ToolBuildRequest:
        return self._builder.create_build_request_from_intent(intent)

    def build_tool(self, request: ToolBuildRequest) -> ToolBuildResult:
        request.name = sanitize_tool_name(request.name)
        self._emit(
            "tool_build_requested",
            toolName=request.name,
            request={
                "purpose": request.purpose,
                "inputs": request.inputs,
                "outputs": request.outputs,
                "capabilities": request.capabilities,
                "risk": request.risk,
            },
        )

        version = self._next_version(request.name)
        spec = self._builder.build_spec(request, version=version)
        code = self._builder.render_code(spec, request)
        code_path = self._builder.write_code(spec, code)
        self._emit("tool_build_generated", toolName=spec.name, version=spec.version, codePath=code_path)

        spec_validation = validate_tool_spec(spec)
        code_validation = validate_generated_code(code)
        validation = {"spec": spec_validation, "code": code_validation}

        status = "draft"
        status = self._state_machine.transition(status, "validated") if spec_validation["ok"] and code_validation["ok"] else "rejected"
        if status == "validated":
            self._emit("tool_build_validated", toolName=spec.name, version=spec.version, validation=validation)
            status = self._state_machine.transition(status, "approval_required")
            self._emit("tool_approval_required", toolName=spec.name, version=spec.version)
        else:
            self._emit(
                "tool_build_rejected",
                toolName=spec.name,
                version=spec.version,
                validation=validation,
                reason="validation_failed",
            )

        spec_path = self._spec_filename(spec.name, spec.version)
        review_path = self._review_filename(spec.name, spec.version)
        review_payload = {
            "tool_name": spec.name,
            "version": spec.version,
            "status": status,
            "summary": {
                "description": spec.description,
                "capabilities": spec.capabilities,
                "allowed_paths": spec.allowed_paths,
                "allowed_domains": spec.allowed_domains,
                "required_secrets": spec.required_secrets,
                "timeout_s": spec.timeout_s,
            },
            "policy_capability_diff": self._policy_diff(spec.name, spec.version, spec),
            "validation": validation,
            "code_path": code_path,
        }
        write_review_artifact(review_path, review_payload)

        record = DynamicToolRecord(
            spec=spec,
            status=status,
            code_path=code_path,
            spec_path=spec_path,
            review_path=review_path,
            validation=validation,
            request=request,
        )
        self._dynamic.setdefault(spec.name, {})[spec.version] = record
        self._save_record(record)
        return ToolBuildResult(
            spec=spec,
            status=status,
            code_path=code_path,
            spec_path=spec_path,
            review_path=review_path,
            validation=validation,
        )

    def approve_tool(self, name: str, version: int) -> ToolBuildResult:
        record = self._dynamic.get(name, {}).get(int(version))
        if record is None:
            raise KeyError(f"Tool version not found: {name} v{version}")
        record.status = self._state_machine.transition(record.status, "approved")
        self._save_record(record)
        return ToolBuildResult(
            spec=record.spec,
            status=record.status,  # type: ignore[arg-type]
            code_path=record.code_path,
            spec_path=record.spec_path,
            review_path=record.review_path,
            validation=record.validation,
        )

    def reject_tool(self, name: str, version: int, reason: str = "rejected_by_reviewer") -> ToolBuildResult:
        record = self._dynamic.get(name, {}).get(int(version))
        if record is None:
            raise KeyError(f"Tool version not found: {name} v{version}")
        target = "rejected"
        if record.status != "rejected":
            record.status = self._state_machine.transition(record.status, target)
        self._save_record(record)
        self._emit(
            "tool_build_rejected",
            toolName=record.spec.name,
            version=record.spec.version,
            validation=record.validation,
            reason=reason,
        )
        return ToolBuildResult(
            spec=record.spec,
            status=record.status,  # type: ignore[arg-type]
            code_path=record.code_path,
            spec_path=record.spec_path,
            review_path=record.review_path,
            validation=record.validation,
        )

    def activate_tool_version(self, name: str, version: int, max_output_chars: int = 20_000) -> AgentTool:
        record = self._dynamic.get(name, {}).get(int(version))
        if record is None:
            raise KeyError(f"Tool version not found: {name} v{version}")
        if record.status not in {"approved", "activated"}:
            raise ValueError(f"Tool must be approved before activation. Current status: {record.status}")
        if name in self._static_tools:
            raise ValueError(f"Cannot activate dynamic tool '{name}' because a static tool with the same name exists.")

        previous_version = self._active_versions.get(name)
        if previous_version is not None and previous_version != int(version):
            previous = self._dynamic.get(name, {}).get(int(previous_version))
            if previous is not None and previous.status == "activated":
                previous.status = "approved"
                self._save_record(previous)

        loaded = self._loader.load_tool(record.code_path, record.spec, self._policy_enforcer)
        policy = ToolPolicy(
            capabilities=list(record.spec.capabilities),
            allowed_paths=list(record.spec.allowed_paths),
            allowed_domains=list(record.spec.allowed_domains),
            required_secrets=list(record.spec.required_secrets),
            timeout_s=record.spec.timeout_s,
            max_output_chars=max_output_chars,
        )
        wrapped = PolicyGuardedTool(loaded, record.spec, policy)
        record.active_tool = wrapped
        record.status = self._state_machine.transition(record.status, "activated")
        self._active_versions[name] = int(version)
        self._save_record(record)
        self._save_active_versions()
        self._emit("tool_activated", toolName=name, version=version)
        return wrapped

    def get_record(self, name: str, version: int) -> Optional[DynamicToolRecord]:
        return self._dynamic.get(name, {}).get(int(version))
