from typing import Any, Dict

from .constants import LOG_LEVELS
from .message_utils import extract_bash_exec_command, format_message_line


def resolve_log_level(value: str) -> int:
    if not value:
        return LOG_LEVELS["quiet"]
    key = str(value).strip().lower()
    if key.isdigit():
        return int(key)
    return LOG_LEVELS.get(key, LOG_LEVELS["quiet"])


def make_event_logger(level: str = "quiet"):
    """Create an event logger function for agent events.
    Levels:
    - quiet: no logging
    - messages: log completed messages
    - stream: log message updates/streams
    - debug: log all events
    """
    level_value = resolve_log_level(level)

    def log(event: Dict[str, Any]) -> None:
        etype = event.get("type")
        if level_value <= LOG_LEVELS["quiet"]:
            if etype == "tool_execution_end" and str(event.get("toolName", "")).strip() == "bash_exec":
                command = extract_bash_exec_command(event)
                if command:
                    print(f"Ran command {command}")
                else:
                    print("Ran command")
            return

        if etype == "tool_execution_start":
            print(f"[tool:start] {event.get('toolName')} args={event.get('args')}")
            return
        if etype == "tool_execution_end":
            print(f"[tool:end] {event.get('toolName')} error={event.get('isError')}")
            return
        if etype == "message_end" and level_value >= LOG_LEVELS["messages"]:
            message = event.get("message")
            if message:
                print(format_message_line(message))
            return
        if etype == "message_update" and level_value >= LOG_LEVELS["stream"]:
            assistant_event = event.get("assistantMessageEvent") or {}
            if assistant_event.get("type") == "text_delta":
                delta = assistant_event.get("delta")
                if delta:
                    print(f"[stream] {delta}")
            return
        if level_value >= LOG_LEVELS["debug"]:
            print(f"[debug] {event}")

    return log
