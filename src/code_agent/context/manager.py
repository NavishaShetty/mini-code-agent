"""Context management for handling large outputs and context windows.

This module provides utilities for:
1. Truncating large command outputs (head + tail pattern)
2. Managing conversation context to fit within token limits
3. Injecting project context (like Ansible Lightspeed does)

Interview talking points:
- Truncation strategies:
  - Head + tail (keep beginning and end)
  - Sliding window (keep recent)
  - Summarization (compress old messages)

- Ansible Lightspeed pattern:
  - Captures context from VS Code (open files, cursor position)
  - Sends context + prompt to service
  - Service enriches with RAG if needed

- Token budgeting:
  - System prompt: ~1000 tokens
  - Context: ~4000 tokens
  - User messages: ~2000 tokens
  - Response: ~4000 tokens
"""

import os
import subprocess
from pathlib import Path


def truncate_output(output: str, max_chars: int = 10000) -> str:
    """Truncate large output using head + tail pattern.

    Keeps the beginning and end of the output, which typically
    contain the most useful information (headers and final results).

    Args:
        output: The full output string
        max_chars: Maximum characters to keep (default 10000)

    Returns:
        Truncated output with middle replaced by [...truncated...]
    """
    if len(output) <= max_chars:
        return output

    # Keep more from the end (usually has results/errors)
    head_size = max_chars // 3
    tail_size = max_chars - head_size - 50  # 50 chars for the truncation message

    head = output[:head_size]
    tail = output[-tail_size:]

    truncated_lines = len(output[head_size:-tail_size].splitlines())

    return f"{head}\n\n[...{truncated_lines} lines truncated...]\n\n{tail}"


def truncate_messages(
    messages: list[dict],
    max_messages: int = 20,
    keep_system: bool = True,
) -> list[dict]:
    """Truncate message history to fit within limits.

    Uses sliding window approach - keeps the most recent messages.

    Args:
        messages: Full message history
        max_messages: Maximum messages to keep
        keep_system: Always keep the system message

    Returns:
        Truncated message list
    """
    if len(messages) <= max_messages:
        return messages

    if keep_system and messages and messages[0].get("role") == "system":
        system = [messages[0]]
        rest = messages[1:]
        return system + rest[-(max_messages - 1) :]
    else:
        return messages[-max_messages:]


def get_project_context(cwd: str | None = None, max_depth: int = 2) -> dict:
    """Gather project context for injection into prompts.

    This mirrors what Ansible Lightspeed does - capture surrounding
    context to help the LLM understand the project.

    Args:
        cwd: Working directory (default: current)
        max_depth: How deep to scan for structure

    Returns:
        Dict with project context information
    """
    cwd = cwd or os.getcwd()
    cwd_path = Path(cwd)

    context = {
        "cwd": cwd,
        "project_name": cwd_path.name,
        "files": [],
        "git_status": None,
        "readme": None,
    }

    # Get file listing (limited)
    try:
        files = []
        for item in cwd_path.iterdir():
            if item.name.startswith("."):
                continue
            if item.is_dir():
                files.append(f"{item.name}/")
            else:
                files.append(item.name)
        context["files"] = sorted(files)[:30]
    except PermissionError:
        pass

    # Get git status if available
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            context["git_status"] = result.stdout.strip() or "clean"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Get README if exists
    for readme_name in ["README.md", "README.txt", "README"]:
        readme_path = cwd_path / readme_name
        if readme_path.exists():
            try:
                content = readme_path.read_text()[:1000]  # First 1000 chars
                context["readme"] = content
                break
            except (PermissionError, UnicodeDecodeError):
                pass

    return context


def format_context_for_prompt(context: dict) -> str:
    """Format project context for injection into system prompt.

    Args:
        context: Dict from get_project_context()

    Returns:
        Formatted string for prompt injection
    """
    lines = [
        f"Project: {context['project_name']}",
        f"Directory: {context['cwd']}",
    ]

    if context.get("files"):
        lines.append(f"Files: {', '.join(context['files'][:10])}")
        if len(context["files"]) > 10:
            lines.append(f"  ... and {len(context['files']) - 10} more")

    if context.get("git_status"):
        lines.append(f"Git status: {context['git_status'][:200]}")

    if context.get("readme"):
        lines.append(f"README preview: {context['readme'][:300]}...")

    return "\n".join(lines)


def estimate_tokens(text: str) -> int:
    """Rough token estimation (4 chars per token average).

    For accurate counting, use tiktoken, but this is fast and
    good enough for budgeting.

    Args:
        text: Text to estimate tokens for

    Returns:
        Estimated token count
    """
    return len(text) // 4


def budget_context(
    messages: list[dict],
    max_tokens: int = 100000,
    reserve_for_response: int = 4000,
) -> list[dict]:
    """Budget messages to fit within token limit.

    Prioritizes:
    1. System message (always keep)
    2. Most recent messages
    3. Truncates old messages if needed

    Args:
        messages: Full message history
        max_tokens: Maximum context tokens
        reserve_for_response: Tokens to reserve for response

    Returns:
        Budgeted message list
    """
    available = max_tokens - reserve_for_response

    # Always include system message
    if not messages:
        return messages

    result = []
    total_tokens = 0

    # Add system message first
    if messages[0].get("role") == "system":
        system_tokens = estimate_tokens(messages[0].get("content", ""))
        result.append(messages[0])
        total_tokens += system_tokens
        messages = messages[1:]

    # Add messages from most recent, going backwards
    for msg in reversed(messages):
        msg_tokens = estimate_tokens(msg.get("content", ""))
        if total_tokens + msg_tokens <= available:
            result.insert(1 if result else 0, msg)
            total_tokens += msg_tokens
        else:
            # We've hit the limit, stop adding
            break

    return result
