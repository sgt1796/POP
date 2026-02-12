from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List

from agent.agent_types import ToolBuildRequest, ToolCapability, ToolSpec


def sanitize_tool_name(value: str) -> str:
    base = re.sub(r"[^a-zA-Z0-9_]+", "_", (value or "").strip())
    base = base.strip("_")
    if not base:
        return "generated_tool"
    if not re.match(r"^[a-zA-Z]", base):
        base = f"t_{base}"
    return base[:64]


class ToolBuilder:
    """Build ToolSpec and deterministic code templates from structured requests."""

    def __init__(self, generated_dir: str) -> None:
        self.generated_dir = generated_dir
        os.makedirs(self.generated_dir, exist_ok=True)

    @staticmethod
    def create_build_request_from_intent(intent: Dict[str, Any]) -> ToolBuildRequest:
        if not isinstance(intent, dict):
            raise ValueError("Tool intent must be a structured object.")
        name = sanitize_tool_name(str(intent.get("name", "")).strip())
        if not name:
            raise ValueError("Tool intent is missing 'name'.")
        purpose = str(intent.get("purpose", "")).strip()
        if not purpose:
            raise ValueError("Tool intent is missing 'purpose'.")
        inputs = intent.get("inputs") or {}
        if not isinstance(inputs, dict):
            raise ValueError("Tool intent 'inputs' must be an object.")
        outputs = intent.get("outputs") or []
        if not isinstance(outputs, list):
            raise ValueError("Tool intent 'outputs' must be an array.")
        raw_caps = intent.get("capabilities") or []
        if not isinstance(raw_caps, list):
            raise ValueError("Tool intent 'capabilities' must be an array.")
        capabilities: List[ToolCapability] = [str(item) for item in raw_caps if item]  # type: ignore
        return ToolBuildRequest(
            name=name,
            purpose=purpose,
            inputs=inputs,
            outputs=[str(x) for x in outputs],
            capabilities=capabilities,
            risk=str(intent.get("risk", "medium")),
            allowed_paths=[str(x) for x in intent.get("allowed_paths", []) or []],
            allowed_domains=[str(x) for x in intent.get("allowed_domains", []) or []],
            required_secrets=[str(x) for x in intent.get("required_secrets", []) or []],
            timeout_s=float(intent.get("timeout_s", 30.0) or 30.0),
        )

    @staticmethod
    def _build_json_schema(request: ToolBuildRequest) -> Dict[str, Any]:
        schema = dict(request.inputs or {})
        if schema.get("type") == "object" and isinstance(schema.get("properties"), dict):
            properties = dict(schema.get("properties") or {})
        else:
            properties = {
                key: {"type": "string", "description": str(value)}
                for key, value in (request.inputs or {}).items()
                if isinstance(key, str)
            }

        if "fs_read" in request.capabilities and "read_path" not in properties:
            properties["read_path"] = {"type": "string", "description": "Path to read."}
        if "fs_write" in request.capabilities:
            if "write_path" not in properties:
                properties["write_path"] = {"type": "string", "description": "Path to write."}
            if "write_content" not in properties:
                properties["write_content"] = {"type": "string", "description": "Content to write."}
        if "http" in request.capabilities and "url" not in properties:
            properties["url"] = {"type": "string", "description": "URL for HTTP GET."}
        if "secrets" in request.capabilities and "secret_name" not in properties:
            properties["secret_name"] = {"type": "string", "description": "Declared secret name to read."}
        if "delay_s" not in properties:
            properties["delay_s"] = {"type": "number", "description": "Optional artificial delay in seconds."}

        return {"type": "object", "properties": properties, "required": []}

    def build_spec(self, request: ToolBuildRequest, version: int) -> ToolSpec:
        description = f"{request.purpose.strip()} (generated tool)"
        return ToolSpec(
            name=sanitize_tool_name(request.name),
            description=description,
            json_schema_parameters=self._build_json_schema(request),
            capabilities=list(request.capabilities),
            allowed_paths=list(request.allowed_paths),
            allowed_domains=list(request.allowed_domains),
            required_secrets=list(request.required_secrets),
            timeout_s=float(request.timeout_s or 30.0),
            version=int(version),
            created_at=time.time(),
        )

    def code_path_for(self, spec: ToolSpec) -> str:
        return os.path.join(self.generated_dir, f"{spec.name}_v{spec.version}.py")

    def render_code(self, spec: ToolSpec, request: ToolBuildRequest) -> str:
        schema_json = json.dumps(spec.json_schema_parameters, indent=4, sort_keys=True)
        outputs_json = json.dumps(request.outputs or [], ensure_ascii=False)
        purpose_json = json.dumps(request.purpose or "", ensure_ascii=False)

        body: List[str] = []
        body.append("        lines = []")
        body.append("        lines.append(f\"tool={self.name}\")")
        body.append(f"        lines.append(\"purpose=\" + {purpose_json})")
        body.append("        lines.append(f\"params={params}\")")
        body.append(f"        expected_outputs = {outputs_json}")
        body.append("        if expected_outputs:")
        body.append("            lines.append(f\"expected_outputs={expected_outputs}\")")
        body.append("        delay_s = float(params.get(\"delay_s\", 0.0) or 0.0)")
        body.append("        if delay_s > 0:")
        body.append("            import asyncio")
        body.append("            await asyncio.sleep(delay_s)")

        if "fs_read" in spec.capabilities:
            body.extend(
                [
                    "        read_path = params.get(\"read_path\")",
                    "        if read_path:",
                    "            file_text = self._read_text(str(read_path))",
                    "            lines.append(f\"read_chars={len(file_text)}\")",
                ]
            )
        if "fs_write" in spec.capabilities:
            body.extend(
                [
                    "        write_path = params.get(\"write_path\")",
                    "        if write_path is not None:",
                    "            write_content = str(params.get(\"write_content\", \"\"))",
                    "            self._write_text(str(write_path), write_content)",
                    "            lines.append(f\"wrote_chars={len(write_content)}\")",
                ]
            )
        if "http" in spec.capabilities:
            body.extend(
                [
                    "        url = params.get(\"url\")",
                    "        if url:",
                    "            timeout_s = float(params.get(\"http_timeout_s\", 10.0) or 10.0)",
                    "            response_text = self._http_get(str(url), timeout_s=timeout_s)",
                    "            lines.append(f\"http_chars={len(response_text)}\")",
                ]
            )
        if "secrets" in spec.capabilities:
            body.extend(
                [
                    "        secret_name = params.get(\"secret_name\")",
                    "        if secret_name:",
                    "            secret_value = self._get_secret(str(secret_name))",
                    "            lines.append(f\"secret_len={len(secret_value)}\")",
                ]
            )

        body.extend(
            [
                "        return AgentToolResult(",
                "            content=[TextContent(type=\"text\", text=\"\\n\".join(lines))],",
                "            details={\"generated\": True, \"version\": self.spec.version},",
                "        )",
            ]
        )
        execute_body = "\n".join(body)

        code = f'''from __future__ import annotations

from typing import Any, Dict, Optional

from agent.agent_types import AgentToolResult, TextContent
from agent.toolsmaker.policy import GeneratedToolBase


class GeneratedTool(GeneratedToolBase):
    name = {json.dumps(spec.name)}
    description = {json.dumps(spec.description)}
    parameters = {schema_json}
    label = {json.dumps(spec.name)}

    async def execute(
        self,
        tool_call_id: str,
        params: Dict[str, Any],
        signal: Optional[Any] = None,
        on_update: Optional[Any] = None,
    ) -> AgentToolResult:
        del tool_call_id, signal, on_update
{execute_body}
'''
        return code

    def write_code(self, spec: ToolSpec, code: str) -> str:
        path = self.code_path_for(spec)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        return path
