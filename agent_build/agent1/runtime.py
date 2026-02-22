import os

from POP.embedder import Embedder
from POP.stream import stream

from agent import Agent
from agent.tools import BashExecConfig, BashExecTool, FastTool, SlowTool, WebSnapshotTool

from .approvals import BashExecApprovalPrompter, ToolsmakerApprovalSubscriber
from .constants import BASH_GIT_READ_SUBCOMMANDS, BASH_READ_COMMANDS, BASH_WRITE_COMMANDS
from .env_utils import (
    parse_bool_env,
    parse_float_env,
    parse_int_env,
    parse_path_list_env,
    parse_toolsmaker_allowed_capabilities,
    sorted_csv,
)
from .event_logger import make_event_logger
from .memory import (
    ConversationMemory,
    DiskMemory,
    EmbeddingIngestionWorker,
    MemoryRetriever,
    MemorySubscriber,
    build_augmented_prompt,
    format_memory_sections,
)
from .message_utils import extract_latest_assistant_text
from .tools import MemorySearchTool, ToolsmakerTool


async def _read_input(prompt: str) -> str:
    return input(prompt)


async def main() -> None:
    agent = Agent({"stream_fn": stream})
    agent.set_model({"provider": "gemini", "id": "gemini-3-flash-preview", "api": None})
    agent.set_timeout(120)

    embedder = Embedder(use_api="openai")
    short_memory = ConversationMemory(embedder=embedder, max_entries=100)
    long_memory = DiskMemory(filepath=os.path.join("agent", "mem", "chat"), embedder=embedder, max_entries=1000)
    retriever = MemoryRetriever(short_term=short_memory, long_term=long_memory)

    ingestion_worker = EmbeddingIngestionWorker(memory=short_memory, long_term=long_memory)
    ingestion_worker.start()
    memory_subscriber = MemorySubscriber(ingestion_worker=ingestion_worker)

    memory_search_tool = MemorySearchTool(retriever=retriever)
    toolsmaker_caps = parse_toolsmaker_allowed_capabilities(os.getenv("POP_AGENT_TOOLSMAKER_ALLOWED_CAPS"))
    toolsmaker_tool = ToolsmakerTool(agent=agent, allowed_capabilities=toolsmaker_caps)
    workspace_root = os.path.realpath(os.getcwd())
    bash_allowed_roots = parse_path_list_env(
        "POP_AGENT_BASH_ALLOWED_ROOTS",
        default_paths=[workspace_root],
        base_dir=workspace_root,
    )
    bash_writable_roots = parse_path_list_env(
        "POP_AGENT_BASH_WRITABLE_ROOTS",
        default_paths=[workspace_root],
        base_dir=workspace_root,
    )
    bash_timeout_s = parse_float_env("POP_AGENT_BASH_TIMEOUT_S", 15.0)
    bash_max_output_chars = parse_int_env("POP_AGENT_BASH_MAX_OUTPUT_CHARS", 20_000)
    bash_prompt_approval = parse_bool_env("POP_AGENT_BASH_PROMPT_APPROVAL", True)
    bash_approval_fn = BashExecApprovalPrompter() if bash_prompt_approval else None
    bash_exec_tool = BashExecTool(
        BashExecConfig(
            project_root=workspace_root,
            allowed_roots=bash_allowed_roots,
            writable_roots=bash_writable_roots,
            read_commands=BASH_READ_COMMANDS,
            write_commands=BASH_WRITE_COMMANDS,
            git_read_subcommands=BASH_GIT_READ_SUBCOMMANDS,
            default_timeout_s=bash_timeout_s,
            max_timeout_s=60.0,
            default_max_output_chars=bash_max_output_chars,
            max_output_chars_limit=100_000,
        ),
        approval_fn=bash_approval_fn,
    )
    bash_read_csv = sorted_csv(BASH_READ_COMMANDS)
    bash_write_csv = sorted_csv(BASH_WRITE_COMMANDS)
    bash_git_csv = sorted_csv(BASH_GIT_READ_SUBCOMMANDS)
    bash_exec_tool.description = (
        "Run one safe shell command without a shell. "
        f"Allowed read commands: {bash_read_csv}. "
        f"Allowed write commands: {bash_write_csv}. "
        f"For git, allowed subcommands: {bash_git_csv}. "
        "Write commands require approval."
    )
    agent.set_system_prompt(
        "You are a helpful assistant. "
        "Use tools when they improve accuracy or when the user asks for external actions. "
        "Prefer existing tools first, especially bash_exec for filesystem/shell inspection tasks. "
        "Use toolsmaker only when no existing tool can safely complete the request. "
        "When existing tools are insufficient, use the toolsmaker tool to create minimal-capability tools. "
        "Follow the lifecycle: create first, then approve, then activate. "
        "When calling toolsmaker create, always include intent.capabilities; "
        "fs_read/fs_write require allowed_paths and http requires allowed_domains. "
        f"Bash_exec allowlist (read): {bash_read_csv}. "
        f"Bash_exec allowlist (write): {bash_write_csv}. "
        f"Bash_exec git subcommands: {bash_git_csv}. "
        "Never call bash_exec with a command/subcommand outside those allowlists."
    )
    agent.set_tools([SlowTool(), FastTool(), WebSnapshotTool(), memory_search_tool, toolsmaker_tool, bash_exec_tool])

    log_level = os.getenv("POP_AGENT_LOG_LEVEL", "quiet")
    toolsmaker_manual_approval = parse_bool_env("POP_AGENT_TOOLSMAKER_PROMPT_APPROVAL", True)
    toolsmaker_auto_activate = parse_bool_env("POP_AGENT_TOOLSMAKER_AUTO_ACTIVATE", True)

    unsubscribe_log = agent.subscribe(make_event_logger(log_level))
    unsubscribe_memory = agent.subscribe(memory_subscriber.on_event)
    if toolsmaker_manual_approval:
        approval_subscriber = ToolsmakerApprovalSubscriber(
            agent=agent,
            auto_activate_default=toolsmaker_auto_activate,
        )
        unsubscribe_approval = agent.subscribe(approval_subscriber.on_event)
    else:
        unsubscribe_approval = lambda: None

    try:
        top_k = max(1, int(os.getenv("POP_AGENT_MEMORY_TOP_K", "3") or "3"))
    except Exception:
        top_k = 3

    print("POP Chatroom Agent (tools + embedding memory)")
    if toolsmaker_manual_approval:
        print(
            "[toolsmaker] manual approval prompts: on "
            f"(default auto-activate={'on' if toolsmaker_auto_activate else 'off'})"
        )
    else:
        print("[toolsmaker] manual approval prompts: off")
    if bash_prompt_approval:
        print("[bash_exec] approval prompts: on")
    else:
        print("[bash_exec] approval prompts: off (medium/high commands will be denied)")
    print("Type 'exit' or 'quit' to stop.\n")
    try:
        while True:
            try:
                user_message = (await _read_input("User: ")).strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting chat.")
                break

            if not user_message:
                continue
            if user_message.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break

            await ingestion_worker.flush()

            memory_text = "(no relevant memories)"
            try:
                short_hits, long_hits = retriever.retrieve_sections(user_message, top_k=top_k, scope="both")
                memory_text = format_memory_sections(short_hits, long_hits)
            except Exception as exc:
                print(f"[memory] retrieval warning: {exc}")

            augmented_prompt = build_augmented_prompt(user_message, memory_text)
            try:
                await agent.prompt(augmented_prompt)
            except Exception as exc:
                print(f"Assistant error: {exc}\n")
                continue

            reply = extract_latest_assistant_text(agent)
            if not reply:
                reply = "(no assistant text returned)"
            print(f"Assistant: {reply}\n")
    finally:
        try:
            await ingestion_worker.shutdown()
        except Exception as exc:
            print(f"[memory] shutdown warning: {exc}")
        unsubscribe_memory()
        unsubscribe_log()
        unsubscribe_approval()
