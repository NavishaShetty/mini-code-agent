"""Interactive execution modes for the agent.

This module provides three execution modes (from mini-swe-agent):
1. human - User provides commands, agent just observes
2. confirm - Agent proposes, user approves before execution
3. yolo - Full autonomy, no human intervention

Interview talking points:
- Human-in-the-loop patterns:
  - Trust boundaries: What actions need approval?
  - Safe commands: Read-only operations can run autonomously
  - Dangerous commands: Writes, deletes, network calls need review

- Production considerations:
  - Default to safest mode (confirm)
  - Audit logging for all actions
  - Ability to interrupt long-running tasks
  - Rollback mechanisms for mistakes
"""

import re
from enum import Enum
from typing import Any

from code_agent.agent.base import (
    Agent,
    Environment,
    FormatError,
    Model,
    TerminatingException,
)


class ExecutionMode(str, Enum):
    """Execution modes for the interactive agent."""

    HUMAN = "human"  # User provides commands
    CONFIRM = "confirm"  # Agent proposes, user approves
    YOLO = "yolo"  # Full autonomy


# Safe command patterns (read-only operations)
SAFE_PATTERNS = [
    r"^ls\b",
    r"^cat\b",
    r"^head\b",
    r"^tail\b",
    r"^grep\b",
    r"^find\b",
    r"^pwd$",
    r"^echo\b",
    r"^wc\b",
    r"^tree\b",
    r"^git status",
    r"^git log",
    r"^git diff",
    r"^git branch",
]

# Dangerous command patterns
DANGEROUS_PATTERNS = [
    r"^rm\s+-rf",
    r"^sudo\b",
    r"curl.*\|\s*(sh|bash)",
]


def is_safe_command(command: str) -> bool:
    """Check if command is read-only/safe."""
    return any(re.match(p, command.strip()) for p in SAFE_PATTERNS)


def is_dangerous_command(command: str) -> bool:
    """Check if command is potentially dangerous."""
    return any(re.search(p, command.strip()) for p in DANGEROUS_PATTERNS)


class UserAbort(TerminatingException):
    """Raised when user aborts the session."""


class InteractiveAgent(Agent):
    """Agent with human-in-the-loop execution modes."""

    def __init__(
        self,
        model: Model,
        env: Environment,
        *,
        mode: ExecutionMode = ExecutionMode.CONFIRM,
        auto_approve_safe: bool = True,
        **kwargs,
    ):
        super().__init__(model, env, **kwargs)
        self.mode = mode
        self.auto_approve_safe = auto_approve_safe

    def execute_action(self, action: str) -> dict[str, Any]:
        """Execute with mode-specific handling."""
        if self.mode == ExecutionMode.HUMAN:
            action = self._get_user_command(action)
        elif self.mode == ExecutionMode.CONFIRM:
            action = self._confirm_action(action)
        elif self.mode == ExecutionMode.YOLO and is_dangerous_command(action):
            print(f"\n⚠️  DANGEROUS: {action}")

        return super().execute_action(action)

    def _get_user_command(self, suggested: str) -> str:
        """Get command from user in HUMAN mode."""
        print(f"\n🤖 Suggests: {suggested}")
        print("Enter command ('quit' to abort, 'skip' to continue):")
        try:
            user_input = input(">>> ").strip()
            if user_input.lower() == "quit":
                raise UserAbort("User aborted")
            if user_input.lower() == "skip":
                return "echo '[skipped]'"
            return user_input or suggested
        except (KeyboardInterrupt, EOFError):
            raise UserAbort("User interrupted")

    def _confirm_action(self, action: str) -> str:
        """Confirm action in CONFIRM mode."""
        if self.auto_approve_safe and is_safe_command(action):
            print(f"\n✅ Auto-approved: {action}")
            return action

        if is_dangerous_command(action):
            print(f"\n🚨 DANGEROUS!")
        print(f"\n🤖 Run: {action}")
        print("[y]es / [n]o / [e]dit / [q]uit")

        try:
            choice = input(">>> ").strip().lower()
            if choice in ("y", "yes", ""):
                return action
            if choice in ("n", "no"):
                raise FormatError("Rejected. Try different approach.")
            if choice in ("e", "edit"):
                return input("New command: ").strip() or action
            if choice in ("q", "quit"):
                raise UserAbort("User aborted")
            return action
        except (KeyboardInterrupt, EOFError):
            raise UserAbort("User interrupted")

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Run with mode announcement."""
        emoji = {"human": "👤", "confirm": "🤝", "yolo": "🚀"}
        print(f"\n{emoji.get(self.mode.value, '🤖')} Mode: {self.mode.value.upper()}")
        print(f"📋 Task: {task}\n")
        return super().run(task, **kwargs)
