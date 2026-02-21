from __future__ import annotations

import asyncio
import os
from dataclasses import replace
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests

from agent.agent_types import AgentTool, AgentToolResult, TextContent, ToolPolicy, ToolSpec


class ToolPolicyViolation(PermissionError):
    """Raised when a dynamic tool attempts an out-of-policy operation."""

    policy_blocked = True

    def __init__(self, message: str, code: str = "policy_violation", details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


class ToolPolicyEnforcer:
    """Capability and scope checks for dynamic tools."""

    def __init__(self, project_root: Optional[str] = None) -> None:
        self.project_root = os.path.abspath(project_root or os.getcwd())

    def _ensure_capability(self, spec: ToolSpec, capability: str) -> None:
        if capability not in set(spec.capabilities):
            raise ToolPolicyViolation(
                f"Tool '{spec.name}' missing required capability '{capability}'.",
                code="missing_capability",
                details={"tool": spec.name, "capability": capability},
            )

    def _allowed_abs_paths(self, spec: ToolSpec) -> List[str]:
        allowed: List[str] = []
        for candidate in spec.allowed_paths:
            if not candidate:
                continue
            if os.path.isabs(candidate):
                allowed.append(os.path.abspath(candidate))
            else:
                allowed.append(os.path.abspath(os.path.join(self.project_root, candidate)))
        return allowed

    def _resolve_allowed_path(self, spec: ToolSpec, path: str, capability: str) -> str:
        self._ensure_capability(spec, capability)
        if not path:
            raise ToolPolicyViolation("Path is required.", code="missing_path", details={"tool": spec.name})

        absolute = os.path.abspath(path if os.path.isabs(path) else os.path.join(self.project_root, path))
        allowlist = self._allowed_abs_paths(spec)
        if not allowlist:
            raise ToolPolicyViolation(
                f"Tool '{spec.name}' has no allowed path entries.",
                code="no_allowed_paths",
                details={"tool": spec.name},
            )

        for allowed in allowlist:
            try:
                if os.path.commonpath([absolute, allowed]) == allowed:
                    return absolute
            except Exception:
                continue

        raise ToolPolicyViolation(
            f"Path '{path}' is not in the allowed scope.",
            code="path_not_allowed",
            details={"tool": spec.name, "path": path, "allowed_paths": spec.allowed_paths},
        )

    def read_text(self, spec: ToolSpec, path: str, encoding: str = "utf-8") -> str:
        target = self._resolve_allowed_path(spec, path, "fs_read")
        try:
            with open(target, "r", encoding=encoding) as f:
                return f.read()
        except FileNotFoundError:
            raise ToolPolicyViolation(
                f"File not found: {path}",
                code="path_not_found",
                details={"tool": spec.name, "path": path},
            )

    def write_text(self, spec: ToolSpec, path: str, content: str, encoding: str = "utf-8") -> None:
        target = self._resolve_allowed_path(spec, path, "fs_write")
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        with open(target, "w", encoding=encoding) as f:
            f.write(content)

    def _is_domain_allowed(self, host: str, allowed_domain: str) -> bool:
        candidate = (allowed_domain or "").lower().strip()
        if not candidate:
            return False
        host = (host or "").lower().strip()
        if host == candidate:
            return True
        if host.endswith("." + candidate.lstrip(".")):
            return True
        return False

    def http_get(self, spec: ToolSpec, url: str, timeout_s: float = 10.0) -> str:
        self._ensure_capability(spec, "http")
        parsed = urlparse(url or "")
        host = parsed.netloc.split("@")[-1].split(":")[0]
        if parsed.scheme not in {"http", "https"} or not host:
            raise ToolPolicyViolation(
                f"URL '{url}' is invalid.",
                code="invalid_url",
                details={"tool": spec.name, "url": url},
            )
        if not spec.allowed_domains:
            raise ToolPolicyViolation(
                f"Tool '{spec.name}' has no allowed domains.",
                code="no_allowed_domains",
                details={"tool": spec.name},
            )
        if not any(self._is_domain_allowed(host, item) for item in spec.allowed_domains):
            raise ToolPolicyViolation(
                f"Domain '{host}' is not allowed.",
                code="domain_not_allowed",
                details={"tool": spec.name, "domain": host, "allowed_domains": spec.allowed_domains},
            )
        response = requests.get(url, timeout=timeout_s)
        response.raise_for_status()
        return response.text

    def get_secret(self, spec: ToolSpec, secret_name: str) -> str:
        self._ensure_capability(spec, "secrets")
        if secret_name not in set(spec.required_secrets):
            raise ToolPolicyViolation(
                f"Secret '{secret_name}' is not declared for tool '{spec.name}'.",
                code="secret_not_declared",
                details={"tool": spec.name, "secret": secret_name, "required_secrets": spec.required_secrets},
            )
        value = os.getenv(secret_name)
        if value is None:
            raise ToolPolicyViolation(
                f"Secret '{secret_name}' is not set in environment.",
                code="secret_missing",
                details={"tool": spec.name, "secret": secret_name},
            )
        return value


class GeneratedToolBase(AgentTool):
    """Base class for generated tools to call policy-checked operations."""

    name: str = "generated_tool"
    description: str = ""
    parameters: Dict[str, Any] = {"type": "object", "properties": {}}
    label: str = "Generated Tool"

    def __init__(self, spec: ToolSpec, policy: ToolPolicyEnforcer) -> None:
        self.spec = spec
        self._policy = policy

    def _read_text(self, path: str) -> str:
        return self._policy.read_text(self.spec, path)

    def _write_text(self, path: str, content: str) -> None:
        self._policy.write_text(self.spec, path, content)

    def _http_get(self, url: str, timeout_s: float = 10.0) -> str:
        return self._policy.http_get(self.spec, url, timeout_s=timeout_s)

    def _get_secret(self, secret_name: str) -> str:
        return self._policy.get_secret(self.spec, secret_name)


class PolicyGuardedTool(AgentTool):
    """Wrap a tool to enforce timeout and output limits."""

    def __init__(self, wrapped: AgentTool, spec: ToolSpec, policy: ToolPolicy) -> None:
        self._wrapped = wrapped
        self._spec = spec
        self._policy = policy
        self.name = wrapped.name
        self.description = wrapped.description
        self.parameters = wrapped.parameters
        self.label = wrapped.label

    async def execute(
        self,
        tool_call_id: str,
        params: Dict[str, Any],
        signal: Optional[Any] = None,
        on_update: Optional[Any] = None,
    ) -> AgentToolResult:
        timeout_s = float(self._policy.timeout_s or self._spec.timeout_s or 30.0)
        try:
            result = await asyncio.wait_for(
                self._wrapped.execute(tool_call_id, params, signal=signal, on_update=on_update),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            raise RuntimeError(f"Tool '{self._wrapped.name}' timed out after {timeout_s:.1f}s")

        max_chars = int(self._policy.max_output_chars or 20_000)
        if max_chars <= 0:
            return result

        total = 0
        adjusted: List[Any] = []
        truncated = False
        for item in result.content:
            if isinstance(item, TextContent):
                text = item.text or ""
                remaining = max_chars - total
                if remaining <= 0:
                    truncated = True
                    continue
                if len(text) > remaining:
                    adjusted.append(replace(item, text=text[:remaining]))
                    total += remaining
                    truncated = True
                    continue
                adjusted.append(item)
                total += len(text)
                continue

            # Keep non-text blocks untouched while preserving overall cap.
            adjusted.append(item)

        if not truncated:
            return result

        details = dict(result.details or {})
        details["output_truncated"] = True
        details["max_output_chars"] = max_chars
        return AgentToolResult(content=adjusted, details=details)
