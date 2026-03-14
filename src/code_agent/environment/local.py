"""Local environment for executing bash commands.

This module provides a simple environment that executes commands directly
on the local machine via subprocess. Used by the agent to interact with
the filesystem and run code.

Interview talking point: Why bash-only (no tool-calling API)?
- Model-agnostic: Works with any LLM
- Simpler debugging: Just regex parsing
- Natural language: Model already knows bash
- Fewer failure modes: No schema validation issues
"""

import os
import platform
import subprocess
from typing import Any

from pydantic import BaseModel


class LocalEnvironmentConfig(BaseModel):
    """Configuration for the local environment."""

    cwd: str = ""
    env: dict[str, str] = {}
    timeout: int = 30


class LocalEnvironment:
    """Executes bash commands directly on the local machine.

    This is the simplest environment - it just runs commands via subprocess.
    More sophisticated environments could use Docker, sandboxing, etc.
    """

    def __init__(self, *, config_class: type = LocalEnvironmentConfig, **kwargs):
        self.config = config_class(**kwargs)

    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command and return the result.

        Returns:
            dict with 'output' (stdout+stderr combined) and 'returncode'
        """
        cwd = cwd or self.config.cwd or os.getcwd()
        try:
            result = subprocess.run(
                command,
                shell=True,
                text=True,
                cwd=cwd,
                env=os.environ | self.config.env,
                timeout=timeout or self.config.timeout,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            return {"output": result.stdout, "returncode": result.returncode}
        except subprocess.TimeoutExpired as e:
            # Capture partial output if available
            output = e.output.decode("utf-8", errors="replace") if e.output else ""
            raise TimeoutError(f"Command timed out after {timeout or self.config.timeout}s. Partial output:\n{output}")

    def get_template_vars(self) -> dict[str, Any]:
        """Return variables for template rendering."""
        return self.config.model_dump() | platform.uname()._asdict() | {"cwd": os.getcwd()}
