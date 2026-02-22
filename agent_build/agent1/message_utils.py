from typing import Any, Dict, List

from agent import Agent
from agent.agent_types import TextContent

from .constants import USER_PROMPT_MARKER


def extract_texts(message: Any) -> List[str]:
    texts: List[str] = []
    if not message:
        return texts
    content = getattr(message, "content", None)
    if not content:
        return texts
    for item in content:
        if isinstance(item, TextContent):
            texts.append(item.text or "")
        elif isinstance(item, dict) and item.get("type") == "text":
            texts.append(str(item.get("text", "")))
    return texts


def extract_latest_assistant_text(agent: Agent) -> str:
    for message in reversed(agent.state.messages):
        if getattr(message, "role", None) != "assistant":
            continue
        text = "\n".join([t for t in extract_texts(message) if t.strip()]).strip()
        if text:
            return text
    return ""


def extract_original_user_message(text: str) -> str:
    if USER_PROMPT_MARKER in text:
        return text.split(USER_PROMPT_MARKER, 1)[1].strip()
    return text.strip()


def format_message_line(message: Any) -> str:
    role = getattr(message, "role", "unknown")
    text = "\n".join(extract_texts(message)).strip()
    return f"[event] {role}: {text}"


def extract_bash_exec_command(event: Dict[str, Any]) -> str:
    args = event.get("args")
    if isinstance(args, dict):
        cmd = args.get("cmd")
        if isinstance(cmd, str) and cmd.strip():
            return cmd.strip()

    result = event.get("result")
    details = getattr(result, "details", None)
    if isinstance(details, dict):
        cmd = details.get("command")
        if isinstance(cmd, str) and cmd.strip():
            return cmd.strip()
    if isinstance(result, dict):
        details = result.get("details")
        if isinstance(details, dict):
            cmd = details.get("command")
            if isinstance(cmd, str) and cmd.strip():
                return cmd.strip()
    return ""
