from typing import Any, Dict, Optional, Set, Tuple

from agent import Agent


class ToolsmakerApprovalSubscriber:
    """Prompt the terminal user for tool approval after toolsmaker create calls."""

    def __init__(self, agent: Agent, auto_activate_default: bool = True) -> None:
        self.agent = agent
        self.auto_activate_default = auto_activate_default
        self._handled: Set[Tuple[str, int]] = set()

    def _read_details(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if event.get("type") != "tool_execution_end":
            return None
        if str(event.get("toolName", "")).strip() != "toolsmaker":
            return None
        result = event.get("result")
        details = getattr(result, "details", None)
        if not isinstance(details, dict):
            return None
        if not bool(details.get("ok")):
            return None
        if str(details.get("action", "")).strip().lower() != "create":
            return None
        if str(details.get("status", "")).strip().lower() != "approval_required":
            return None
        return details

    def on_event(self, event: Dict[str, Any]) -> None:
        try:
            details = self._read_details(event)
            if details is None:
                return
            name = str(details.get("name", "")).strip()
            version = int(details.get("version", 0) or 0)
            if not name or version <= 0:
                return

            key = (name, version)
            if key in self._handled:
                return
            self._handled.add(key)

            review_path = str(details.get("review_path", "")).strip()
            print("\n[toolsmaker] Manual approval requested.")
            print(f"[toolsmaker] tool={name} version={version}")
            if review_path:
                print(f"[toolsmaker] review={review_path}")
            requested_capabilities = list(details.get("requested_capabilities") or [])
            if requested_capabilities:
                print(f"[toolsmaker] requested_capabilities={requested_capabilities}")
            else:
                print("[toolsmaker] requested_capabilities=(none)")

            decision = input("[toolsmaker] Approve this tool version? [y/N]: ").strip().lower()
            if decision in {"y", "yes"}:
                approved = self.agent.approve_dynamic_tool(name=name, version=version)
                print(f"[toolsmaker] approved status={approved.status}")

                if self.auto_activate_default:
                    activation_prompt = "[toolsmaker] Activate now? [Y/n]: "
                else:
                    activation_prompt = "[toolsmaker] Activate now? [y/N]: "
                activate_choice = input(activation_prompt).strip().lower()
                should_activate = activate_choice in {"y", "yes"} or (
                    activate_choice == "" and self.auto_activate_default
                )
                if should_activate:
                    activated_tool = self.agent.activate_tool_version(name=name, version=version)
                    print(f"[toolsmaker] activated tool={activated_tool.name} version={version}")
                else:
                    print("[toolsmaker] activation skipped")
            else:
                reason = input("[toolsmaker] Reject reason (enter for default): ").strip() or "rejected_by_reviewer"
                rejected = self.agent.reject_dynamic_tool(name=name, version=version, reason=reason)
                print(f"[toolsmaker] rejected status={rejected.status} reason={reason}")
        except Exception as exc:
            print(f"[toolsmaker] manual approval warning: {exc}")


class BashExecApprovalPrompter:
    """Prompt the terminal user for medium/high risk bash_exec commands."""

    def __call__(self, request: Dict[str, Any]) -> bool:
        try:
            command = str(request.get("command", "")).strip()
            cwd = str(request.get("cwd", "")).strip()
            risk = str(request.get("risk", "")).strip() or "unknown"
            justification = str(request.get("justification", "")).strip()

            print("\n[bash_exec] Approval requested.")
            print(f"[bash_exec] risk={risk}")
            print(f"[bash_exec] cwd={cwd}")
            print(f"[bash_exec] command={command}")
            if justification:
                print(f"[bash_exec] justification={justification}")
            else:
                print("[bash_exec] justification=(none)")

            decision = input("[bash_exec] Allow this command? [y/N]: ").strip().lower()
            return decision in {"y", "yes"}
        except Exception as exc:
            print(f"[bash_exec] approval prompt warning: {exc}")
            return False
