from typing import Any, Dict, List, Optional, Sequence, Tuple

from agent import Agent
from agent.agent_types import AgentTool, AgentToolResult, TextContent

from .constants import TOOL_CAPABILITIES
from .memory import MemoryRetriever


class MemorySearchTool(AgentTool):
    name = "memory_search"
    description = "Semantic search over stored chat memory."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "top_k": {"type": "integer", "description": "Number of results"},
            "scope": {
                "type": "string",
                "description": "Memory scope: short, long, or both",
                "enum": ["short", "long", "both"],
            },
        },
        "required": ["query"],
    }
    label = "Memory Search"

    def __init__(self, retriever: MemoryRetriever) -> None:
        self.retriever = retriever

    async def execute(
        self,
        tool_call_id: str,
        params: Dict[str, Any],
        signal: Optional[Any] = None,
        on_update: Optional[Any] = None,
    ) -> AgentToolResult:
        del tool_call_id, signal, on_update
        query = str(params.get("query", "")).strip()
        if not query:
            return AgentToolResult(
                content=[TextContent(type="text", text="memory_search error: missing query")],
                details={"error": "missing query"},
            )
        try:
            top_k = int(params.get("top_k", 3) or 3)
        except Exception:
            top_k = 3
        top_k = max(1, top_k)
        scope = str(params.get("scope", "both")).strip().lower()
        if scope not in {"short", "long", "both"}:
            scope = "both"
        try:
            hits = self.retriever.retrieve(query=query, top_k=top_k, scope=scope)
        except Exception as exc:
            return AgentToolResult(
                content=[TextContent(type="text", text=f"memory_search error: {exc}")],
                details={"error": str(exc)},
            )
        if not hits:
            text = "No matching memories found."
        else:
            text = "Memory search results:\n" + "\n".join(f"{i + 1}. {h}" for i, h in enumerate(hits))
        return AgentToolResult(
            content=[TextContent(type="text", text=text)],
            details={"query": query, "top_k": top_k, "scope": scope, "count": len(hits)},
        )


class ToolsmakerTool(AgentTool):
    name = "toolsmaker"
    description = "Manage generated tools via create, approve, activate, reject, or list actions."
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "approve", "activate", "reject", "list"],
                "description": "Lifecycle action to run",
            },
            "intent": {
                "type": "object",
                "description": "Structured tool intent payload used by action=create",
            },
            "name": {
                "type": "string",
                "description": "Tool name used by approve/activate/reject",
            },
            "version": {
                "type": "integer",
                "description": "Tool version used by approve/activate/reject",
            },
            "reason": {
                "type": "string",
                "description": "Optional rejection reason for action=reject",
            },
            "max_output_chars": {
                "type": "integer",
                "description": "Optional output cap used by action=activate",
            },
        },
        "required": ["action"],
    }
    label = "Toolsmaker"

    def __init__(self, agent: Agent, allowed_capabilities: Sequence[str]) -> None:
        self.agent = agent
        self.allowed_capabilities = [str(x) for x in allowed_capabilities if x in TOOL_CAPABILITIES]

    @staticmethod
    def _error(text: str, details: Dict[str, Any]) -> AgentToolResult:
        return AgentToolResult(
            content=[TextContent(type="text", text=text)],
            details={"ok": False, **details},
        )

    @staticmethod
    def _ok(text: str, details: Dict[str, Any]) -> AgentToolResult:
        return AgentToolResult(
            content=[TextContent(type="text", text=text)],
            details={"ok": True, **details},
        )

    @staticmethod
    def _result_summary(status: str, name: str, version: int, review_path: str = "") -> str:
        lines = [f"tool={name}", f"version={version}", f"status={status}"]
        if review_path:
            lines.append(f"review_path={review_path}")
        return "\n".join(lines)

    @staticmethod
    def _parse_name_version(params: Dict[str, Any]) -> Tuple[str, int]:
        name = str(params.get("name", "")).strip()
        if not name:
            raise ValueError("missing required field: name")
        version_raw = params.get("version", None)
        if version_raw is None:
            raise ValueError("missing required field: version")
        try:
            version = int(version_raw)
        except Exception as exc:
            raise ValueError("version must be an integer") from exc
        if version <= 0:
            raise ValueError("version must be > 0")
        return name, version

    @staticmethod
    def _infer_required_capabilities(intent: Dict[str, Any]) -> List[str]:
        name_text = str(intent.get("name", "")).strip().lower()
        purpose_text = str(intent.get("purpose", "")).strip().lower()
        text = f"{name_text} {purpose_text}"
        required: List[str] = []
        if any(token in text for token in ["write", "writer", "save", "append"]):
            required.append("fs_write")
        if any(token in text for token in ["read", "reader", "load"]):
            required.append("fs_read")
        if any(token in text for token in ["http", "url", "fetch", "request", "download"]):
            required.append("http")
        if any(token in text for token in ["secret", "token", "password", "api key", "env"]):
            required.append("secrets")
        return sorted(set(required))

    def _validate_intent_contract(self, intent: Dict[str, Any]) -> Optional[AgentToolResult]:
        raw_caps = intent.get("capabilities", [])
        if raw_caps is None:
            raw_caps = []
        if not isinstance(raw_caps, list):
            return self._error(
                "toolsmaker create error: intent.capabilities must be an array.",
                {"action": "create", "error": "invalid_capabilities"},
            )

        requested = sorted({str(x).strip() for x in raw_caps if str(x).strip()})
        if not requested:
            inferred = self._infer_required_capabilities(intent)
            return self._error(
                "toolsmaker create blocked: intent.capabilities is empty; this would generate a no-op tool.",
                {
                    "action": "create",
                    "error": "missing_capabilities",
                    "requested": requested,
                    "inferred_required": inferred,
                    "hint": "Include capabilities such as fs_write/fs_read/http/secrets in the intent.",
                },
            )

        inferred_required = self._infer_required_capabilities(intent)
        missing = sorted(set(inferred_required) - set(requested))
        if missing:
            return self._error(
                "toolsmaker create blocked: intent likely needs capabilities that are missing.",
                {
                    "action": "create",
                    "error": "inferred_capabilities_missing",
                    "requested": requested,
                    "inferred_required": inferred_required,
                    "missing": missing,
                },
            )

        if "fs_write" in requested and not list(intent.get("allowed_paths") or []):
            return self._error(
                "toolsmaker create blocked: fs_write requires allowed_paths in intent.",
                {
                    "action": "create",
                    "error": "missing_allowed_paths",
                    "requested": requested,
                },
            )
        if "fs_read" in requested and not list(intent.get("allowed_paths") or []):
            return self._error(
                "toolsmaker create blocked: fs_read requires allowed_paths in intent.",
                {
                    "action": "create",
                    "error": "missing_allowed_paths",
                    "requested": requested,
                },
            )
        if "http" in requested and not list(intent.get("allowed_domains") or []):
            return self._error(
                "toolsmaker create blocked: http requires allowed_domains in intent.",
                {
                    "action": "create",
                    "error": "missing_allowed_domains",
                    "requested": requested,
                },
            )
        return None

    def _check_capability_guard(self, intent: Dict[str, Any]) -> Optional[AgentToolResult]:
        raw_caps = intent.get("capabilities", [])
        if raw_caps is None:
            raw_caps = []
        if not isinstance(raw_caps, list):
            return self._error(
                "toolsmaker create error: intent.capabilities must be an array.",
                {"action": "create", "error": "invalid capabilities"},
            )

        requested = sorted({str(x).strip() for x in raw_caps if str(x).strip()})
        allowed = sorted(set(self.allowed_capabilities))
        disallowed = sorted(set(requested) - set(allowed))
        if disallowed:
            return self._error(
                "toolsmaker create blocked: requested capabilities are not allowed by current runtime policy.",
                {
                    "action": "create",
                    "error": "capability_not_allowed",
                    "requested": requested,
                    "allowed": allowed,
                    "disallowed": disallowed,
                    "hint": "Set POP_AGENT_TOOLSMAKER_ALLOWED_CAPS to include the required capabilities.",
                },
            )
        return None

    def _handle_create(self, params: Dict[str, Any]) -> AgentToolResult:
        intent = params.get("intent", None)
        if not isinstance(intent, dict):
            return self._error(
                "toolsmaker create error: missing or invalid intent object.",
                {"action": "create", "error": "missing_intent"},
            )

        invalid = self._validate_intent_contract(intent)
        if invalid is not None:
            return invalid

        blocked = self._check_capability_guard(intent)
        if blocked is not None:
            return blocked

        result = self.agent.build_dynamic_tool_from_intent(intent)
        next_steps = "Next steps: approve this version, then activate it."
        text = "\n".join(
            [
                self._result_summary(
                    status=str(result.status),
                    name=result.spec.name,
                    version=result.spec.version,
                    review_path=result.review_path,
                ),
                next_steps,
            ]
        )
        return self._ok(
            text,
            {
                "action": "create",
                "status": result.status,
                "name": result.spec.name,
                "version": result.spec.version,
                "review_path": result.review_path,
                "spec_path": result.spec_path,
                "code_path": result.code_path,
                "validation": result.validation,
                "requested_capabilities": list(intent.get("capabilities") or []),
                "requested_allowed_paths": list(intent.get("allowed_paths") or []),
                "requested_allowed_domains": list(intent.get("allowed_domains") or []),
                "next_steps": ["approve", "activate"],
            },
        )

    def _handle_approve(self, params: Dict[str, Any]) -> AgentToolResult:
        name, version = self._parse_name_version(params)
        result = self.agent.approve_dynamic_tool(name=name, version=version)
        text = self._result_summary(status=str(result.status), name=result.spec.name, version=result.spec.version)
        return self._ok(
            text,
            {
                "action": "approve",
                "status": result.status,
                "name": result.spec.name,
                "version": result.spec.version,
                "review_path": result.review_path,
                "validation": result.validation,
            },
        )

    def _handle_activate(self, params: Dict[str, Any]) -> AgentToolResult:
        name, version = self._parse_name_version(params)
        try:
            max_output_chars = int(params.get("max_output_chars", 20_000) or 20_000)
        except Exception as exc:
            raise ValueError("max_output_chars must be an integer") from exc
        max_output_chars = max(1, max_output_chars)

        tool = self.agent.activate_tool_version(name=name, version=version, max_output_chars=max_output_chars)
        tools = self.agent.list_tools()
        text = "\n".join(
            [
                f"activated={tool.name}",
                f"version={version}",
                "available_tools=" + ", ".join(tools),
            ]
        )
        return self._ok(
            text,
            {
                "action": "activate",
                "status": "activated",
                "name": name,
                "version": version,
                "activated_tool": tool.name,
                "max_output_chars": max_output_chars,
                "tools": tools,
            },
        )

    def _handle_reject(self, params: Dict[str, Any]) -> AgentToolResult:
        name, version = self._parse_name_version(params)
        reason = str(params.get("reason", "rejected_by_reviewer")).strip() or "rejected_by_reviewer"
        result = self.agent.reject_dynamic_tool(name=name, version=version, reason=reason)
        text = self._result_summary(status=str(result.status), name=result.spec.name, version=result.spec.version)
        return self._ok(
            text,
            {
                "action": "reject",
                "status": result.status,
                "name": result.spec.name,
                "version": result.spec.version,
                "reason": reason,
            },
        )

    def _handle_list(self) -> AgentToolResult:
        tools = self.agent.list_tools()
        if tools:
            text = "tools:\n" + "\n".join(f"{i + 1}. {name}" for i, name in enumerate(tools))
        else:
            text = "tools: (none)"
        return self._ok(text, {"action": "list", "tools": tools, "count": len(tools)})

    async def execute(
        self,
        tool_call_id: str,
        params: Dict[str, Any],
        signal: Optional[Any] = None,
        on_update: Optional[Any] = None,
    ) -> AgentToolResult:
        del tool_call_id, signal, on_update
        action = str(params.get("action", "")).strip().lower()
        if action not in {"create", "approve", "activate", "reject", "list"}:
            return self._error(
                "toolsmaker error: action must be one of create|approve|activate|reject|list.",
                {"action": action or None, "error": "invalid_action"},
            )
        try:
            if action == "create":
                return self._handle_create(params)
            if action == "approve":
                return self._handle_approve(params)
            if action == "activate":
                return self._handle_activate(params)
            if action == "reject":
                return self._handle_reject(params)
            return self._handle_list()
        except Exception as exc:
            return self._error(
                f"toolsmaker {action} error: {exc}",
                {"action": action, "error": str(exc)},
            )
