import asyncio
import time

from ..agent_types import AgentTool, AgentToolResult, TextContent
from POP.utils import web_snapshot as websnapshot


class SlowTool(AgentTool):
    name = "slow"
    description = "Sleep a bit"
    parameters = {"type": "object", "properties": {"seconds": {"type": "number"}}}
    label = "Slow"

    async def execute(self, tool_call_id, params, signal=None, on_update=None):
        t0 = time.time()
        seconds = float(params.get("seconds", 1.0))
        steps = max(1, int(seconds * 10))
        for _ in range(steps):
            if signal and signal.is_set():
                break
            await asyncio.sleep(0.1)
        return AgentToolResult(
            content=[TextContent(type="text", text=f"slow done {seconds}s")],
            details={"time_elapsed": time.time() - t0},
        )


class FastTool(AgentTool):
    name = "fast"
    description = "Return quickly"
    parameters = {"type": "object", "properties": {}}
    label = "Fast"

    async def execute(self, tool_call_id, params, signal=None, on_update=None):
        return AgentToolResult(
            content=[TextContent(type="text", text="fast done")],
            details={},
        )


class WebSnapshotTool(AgentTool):
    name = "websnapshot"
    description = "Fetch a text snapshot of a webpage using POP.utils.websnapshot"
    parameters = {
        "type": "object",
        "properties": {
            "web_url": {"type": "string", "description": "URL to snapshot"},
            "url": {"type": "string", "description": "Alias for web_url"},
            "use_api_key": {"type": "boolean"},
            "return_format": {"type": "string"},
            "timeout": {"type": "number"},
            "target_selector": {"type": "array", "items": {"type": "string"}},
            "wait_for_selector": {"type": "array", "items": {"type": "string"}},
            "exclude_selector": {"type": "array", "items": {"type": "string"}},
            "remove_image": {"type": "boolean"},
            "links_at_end": {"type": "boolean"},
            "images_at_end": {"type": "boolean"},
            "json_response": {"type": "boolean"},
            "image_caption": {
                "type": "boolean",
                "description": (
                    "Caption images in the snapshot using AI. Note: this may consume additional tokens and time."
                ),
            },
            "cookie": {"type": "string"},
        },
        "required": ["web_url"],
    }
    label = "Web Snapshot"

    async def execute(self, tool_call_id, params, signal=None, on_update=None):
        web_url = params.get("web_url") or params.get("url")
        if not web_url:
            return AgentToolResult(
                content=[TextContent(type="text", text="websnapshot error: missing web_url")],
                details={"error": "missing web_url"},
            )

        kwargs = {
            "use_api_key": bool(params.get("use_api_key", True)),
            "return_format": params.get("return_format", "default"),
            "timeout": int(params.get("timeout", 0) or 0),
            "target_selector": params.get("target_selector") or None,
            "wait_for_selector": params.get("wait_for_selector") or None,
            "exclude_selector": params.get("exclude_selector") or None,
            "remove_image": bool(params.get("remove_image", False)),
            "links_at_end": bool(params.get("links_at_end", False)),
            "images_at_end": bool(params.get("images_at_end", False)),
            "json_response": bool(params.get("json_response", False)),
            "image_caption": bool(params.get("image_caption", False)),
            "cookie": params.get("cookie"),
        }
        try:
            snapshot = websnapshot.get_text_snapshot(web_url=web_url, **kwargs)
        except Exception as exc:
            return AgentToolResult(
                content=[TextContent(type="text", text=f"websnapshot error: {exc}")],
                details={"error": str(exc)},
            )
        return AgentToolResult(
            content=[TextContent(type="text", text=snapshot)],
            details={"url": web_url},
        )
