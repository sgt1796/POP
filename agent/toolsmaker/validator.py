from __future__ import annotations

import ast
import re
from typing import Any, Dict, List

from agent.agent_types import ToolSpec
from agent.toolsmaker.builder import is_meaningful_tool_name


FORBIDDEN_IMPORTS = {"subprocess", "socket", "ctypes", "importlib", "pathlib"}
FORBIDDEN_CALLS = {"eval", "exec", "__import__"}
ALLOWED_CAPABILITIES = {"fs_read", "fs_write", "http", "secrets"}


def _resolve_call_name(node: ast.Call) -> str:
    target = node.func
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return ""


def validate_generated_code(code: str) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {
            "ok": False,
            "errors": [f"Syntax error at line {exc.lineno}: {exc.msg}"],
            "warnings": warnings,
        }

    has_generated_tool = False
    has_execute_method = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in FORBIDDEN_IMPORTS:
                    errors.append(f"Forbidden import: {alias.name}")

        if isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".")[0]
            if module in FORBIDDEN_IMPORTS:
                errors.append(f"Forbidden import-from module: {node.module}")

        if isinstance(node, ast.Call):
            call_name = _resolve_call_name(node)
            if call_name in FORBIDDEN_CALLS:
                errors.append(f"Forbidden call: {call_name}")

        if isinstance(node, ast.ClassDef) and node.name == "GeneratedTool":
            has_generated_tool = True
            if not any(
                isinstance(base, ast.Name) and base.id == "GeneratedToolBase"
                or isinstance(base, ast.Attribute) and base.attr == "GeneratedToolBase"
                for base in node.bases
            ):
                errors.append("GeneratedTool must inherit from GeneratedToolBase.")
            has_execute_method = any(
                isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == "execute"
                for child in node.body
            )

    if not has_generated_tool:
        errors.append("GeneratedTool class not found.")
    elif not has_execute_method:
        errors.append("GeneratedTool must define execute(...).")

    return {"ok": len(errors) == 0, "errors": errors, "warnings": warnings}


def validate_tool_spec(spec: ToolSpec) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]{0,63}$", spec.name or ""):
        errors.append("Tool name must match ^[a-zA-Z][a-zA-Z0-9_]{0,63}$")
    elif not is_meaningful_tool_name(spec.name):
        errors.append("Tool name must be meaningful and not a placeholder (e.g. generated_tool)")

    schema = spec.json_schema_parameters or {}
    if schema.get("type") != "object":
        errors.append("json_schema_parameters.type must be 'object'")
    if not isinstance(schema.get("properties", {}), dict):
        errors.append("json_schema_parameters.properties must be an object")

    invalid_caps = [cap for cap in spec.capabilities if cap not in ALLOWED_CAPABILITIES]
    if invalid_caps:
        errors.append(f"Unsupported capabilities: {', '.join(invalid_caps)}")

    if spec.timeout_s <= 0:
        errors.append("timeout_s must be > 0")

    if "fs_read" in spec.capabilities or "fs_write" in spec.capabilities:
        if not spec.allowed_paths:
            errors.append("allowed_paths is required for filesystem capabilities")
    if "http" in spec.capabilities:
        if not spec.allowed_domains:
            errors.append("allowed_domains is required for http capability")
    if "secrets" in spec.capabilities and not spec.required_secrets:
        warnings.append("secrets capability is enabled but required_secrets is empty")

    return {"ok": len(errors) == 0, "errors": errors, "warnings": warnings}
