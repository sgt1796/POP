from typing import Set

LOG_LEVELS = {
    "quiet": 0,
    "messages": 1,
    "stream": 2,
    "debug": 3,
}

USER_PROMPT_MARKER = "|Current user message|:\n"
DEFAULT_TOOLSMAKER_ALLOWED_CAPS = "fs_read,fs_write,http"
TOOL_CAPABILITIES: Set[str] = {"fs_read", "fs_write", "http", "secrets"}

BASH_READ_COMMANDS: Set[str] = {
    "pwd",
    "ls",
    "cat",
    "head",
    "tail",
    "wc",
    "find",
    "rg",
    "git",
    "echo",
    "df",
    "du",
}
BASH_WRITE_COMMANDS: Set[str] = {"mkdir", "touch", "cp", "mv", "rm"}
BASH_GIT_READ_SUBCOMMANDS: Set[str] = {"status", "diff", "log", "show", "branch"}
