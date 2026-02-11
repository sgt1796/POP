from __future__ import annotations

import hashlib
import importlib.util
import os
from typing import Any

from agent.agent_types import AgentTool, ToolSpec
from agent.dynamic_tools.policy import ToolPolicyEnforcer


class ToolLoader:
    """Load generated tool modules from disk."""

    def load_tool(self, code_path: str, spec: ToolSpec, policy: ToolPolicyEnforcer) -> AgentTool:
        if not os.path.exists(code_path):
            raise FileNotFoundError(f"Generated tool file not found: {code_path}")

        token = hashlib.md5(f"{code_path}:{spec.version}".encode("utf-8")).hexdigest()[:10]
        module_name = f"agent_dynamic_{spec.name}_v{spec.version}_{token}"
        module_spec = importlib.util.spec_from_file_location(module_name, code_path)
        if module_spec is None or module_spec.loader is None:
            raise RuntimeError(f"Unable to load module spec from {code_path}")

        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)

        tool_cls: Any = getattr(module, "GeneratedTool", None)
        if tool_cls is None:
            raise RuntimeError(f"GeneratedTool class missing in {code_path}")

        tool = tool_cls(spec, policy)
        if not hasattr(tool, "execute") or not callable(getattr(tool, "execute")):
            raise RuntimeError("Loaded tool does not implement execute(...)")
        return tool
