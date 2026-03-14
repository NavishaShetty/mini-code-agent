"""Core agent implementation using the ReAct pattern.

The ReAct (Reasoning + Acting) pattern:
    Thought -> Action -> Observation -> Thought -> ...

This is the core loop that powers code agents like Claude Code, OpenCode,
and mini-swe-agent. The key insight is that we don't need special tool-calling
APIs - we can just parse bash blocks from the model's response.

Interview talking points:
- Why bash-only (no tool-calling API)?
  - Model-agnostic: Works with any LLM
  - Simpler debugging: Just regex parsing
  - Natural language: Model already knows bash
  - Fewer failure modes: No schema validation issues

- State management:
  - Linear message history (simplest)
  - No complex state machine
  - Each step is independent
  - Easy to replay/debug

- Failure modes:
  - Infinite loops: Same action repeated -> detect with action history
  - Parse errors: No bash block found -> retry with format reminder
  - Execution timeout: Command hangs -> kill after timeout, return partial
  - Context overflow: Too many messages -> truncate or summarize
"""

import re
import time
from typing import Any, Protocol

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel

from code_agent.context.manager import (
    budget_context,
    format_context_for_prompt,
    get_project_context,
    truncate_output,
)


# Protocol definitions for dependency injection
class Model(Protocol):
    """Protocol for LLM models."""

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict[str, Any]: ...
    def get_template_vars(self) -> dict[str, Any]: ...


class Environment(Protocol):
    """Protocol for execution environments."""

    def execute(self, command: str, **kwargs) -> dict[str, Any]: ...
    def get_template_vars(self) -> dict[str, Any]: ...


# Exception hierarchy
class NonTerminatingException(Exception):
    """Exceptions that the agent can recover from by retrying."""


class FormatError(NonTerminatingException):
    """Raised when the LLM's output is not in the expected format."""


class ExecutionTimeoutError(NonTerminatingException):
    """Raised when command execution times out."""


class TerminatingException(Exception):
    """Exceptions that end the agent's run."""


class Submitted(TerminatingException):
    """Raised when the agent declares it has finished the task."""


class LimitsExceeded(TerminatingException):
    """Raised when step or cost limits are reached."""


class LoopDetected(TerminatingException):
    """Raised when the agent is stuck in a loop."""


# Default templates
DEFAULT_SYSTEM_TEMPLATE = """You are a helpful coding assistant that can execute bash commands to accomplish tasks.

You have access to a bash shell. To execute a command, wrap it in a bash code block:

```bash
your command here
```

Guidelines:
- Execute ONE command at a time
- After each command, observe the output before deciding next steps
- When you have completed the task, output: TASK_COMPLETE
- If you cannot complete the task, explain why

Current working directory: {{ cwd }}
Operating system: {{ system }} {{ release }}
{% if project_context %}

Project Context:
{{ project_context }}
{% endif %}
"""

DEFAULT_INSTANCE_TEMPLATE = """Task: {{ task }}

Please complete this task step by step."""

DEFAULT_FORMAT_ERROR_TEMPLATE = """Error: Expected exactly one bash code block.

Found {{ actions|length }} bash blocks. Please provide exactly one command:

```bash
your single command here
```"""

DEFAULT_OBSERVATION_TEMPLATE = """Output:
{{ output }}"""

DEFAULT_TIMEOUT_TEMPLATE = """Command timed out after {{ timeout }}s.
Partial output:
{{ output }}

Please try a different approach or break the task into smaller steps."""


class AgentConfig(BaseModel):
    """Configuration for the agent."""

    system_template: str = DEFAULT_SYSTEM_TEMPLATE
    instance_template: str = DEFAULT_INSTANCE_TEMPLATE
    format_error_template: str = DEFAULT_FORMAT_ERROR_TEMPLATE
    observation_template: str = DEFAULT_OBSERVATION_TEMPLATE
    timeout_template: str = DEFAULT_TIMEOUT_TEMPLATE
    action_regex: str = r"```bash\s*\n(.*?)\n```"
    step_limit: int = 50  # 0 = unlimited
    cost_limit: float = 1.0  # 0 = unlimited
    loop_threshold: int = 3  # Number of repeated actions before detecting loop

    # Context management settings
    max_output_chars: int = 10000  # Truncate command output beyond this
    max_context_tokens: int = 100000  # Budget messages to fit this limit
    inject_project_context: bool = True  # Add project info to system prompt


class Agent:
    """Core agent implementing the ReAct pattern.

    The agent loop:
    1. run() initializes messages and calls step() repeatedly
    2. step() calls query() -> parse_action() -> execute_action()
    3. Observations are added to message history
    4. Loop continues until Submitted or LimitsExceeded

    Example usage:
        model = LiteLLMModel(model_name="claude-3-5-sonnet-20241022")
        env = LocalEnvironment()
        agent = Agent(model, env)
        status, message = agent.run("List all Python files")
    """

    def __init__(
        self,
        model: Model,
        env: Environment,
        *,
        config_class: type = AgentConfig,
        **kwargs,
    ):
        self.config = config_class(**kwargs)
        self.model = model
        self.env = env
        self.messages: list[dict[str, Any]] = []
        self.action_history: list[str] = []
        self.extra_template_vars: dict[str, Any] = {}

    def render_template(self, template: str, **kwargs) -> str:
        """Render a Jinja2 template with context variables."""
        template_vars = (
            self.config.model_dump()
            | self.env.get_template_vars()
            | self.model.get_template_vars()
            | self.extra_template_vars
        )
        return Template(template, undefined=StrictUndefined).render(**kwargs, **template_vars)

    def add_message(self, role: str, content: str, **kwargs):
        """Add a message to the conversation history."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": time.time(),
            **kwargs,
        })

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Run the agent until completion.

        Args:
            task: The task description for the agent

        Returns:
            Tuple of (exit_status, message)
            - exit_status: "Submitted", "LimitsExceeded", "LoopDetected", etc.
            - message: Final output or error message
        """
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.action_history = []

        # Optionally inject project context (like Ansible Lightspeed pattern)
        if self.config.inject_project_context:
            project_ctx = get_project_context()
            self.extra_template_vars["project_context"] = format_context_for_prompt(project_ctx)

        # Initialize conversation
        self.add_message("system", self.render_template(self.config.system_template))
        self.add_message("user", self.render_template(self.config.instance_template))

        while True:
            try:
                self.step()
            except NonTerminatingException as e:
                # Recoverable error - add to messages and continue
                self.add_message("user", str(e))
            except TerminatingException as e:
                # Terminal state - return result
                self.add_message("user", str(e))
                return type(e).__name__, str(e)

    def step(self) -> dict[str, Any]:
        """Execute one iteration of the agent loop.

        Returns:
            The observation dict from execute_action
        """
        response = self.query()
        return self.get_observation(response)

    def query(self) -> dict[str, Any]:
        """Query the model and check limits.

        Returns:
            Model response dict with 'content' key
        """
        # Check limits before querying
        model_stats = self.model.get_template_vars()
        if 0 < self.config.step_limit <= model_stats.get("n_calls", 0):
            raise LimitsExceeded(f"Step limit ({self.config.step_limit}) exceeded")
        if 0 < self.config.cost_limit <= model_stats.get("total_cost", 0):
            raise LimitsExceeded(f"Cost limit (${self.config.cost_limit}) exceeded")

        # Budget messages to fit within context window
        budgeted_messages = budget_context(
            self.messages,
            max_tokens=self.config.max_context_tokens,
        )

        response = self.model.query(budgeted_messages)
        self.add_message("assistant", **response)
        return response

    def get_observation(self, response: dict[str, Any]) -> dict[str, Any]:
        """Parse and execute action, return observation."""
        action = self.parse_action(response)
        output = self.execute_action(action)

        # Truncate large outputs to avoid blowing context window
        truncated = truncate_output(
            output["output"],
            max_chars=self.config.max_output_chars,
        )

        observation = self.render_template(self.config.observation_template, output=truncated)
        self.add_message("user", observation)
        return output

    def parse_action(self, response: dict[str, Any]) -> str:
        """Extract the bash command from the model's response.

        Args:
            response: Model response with 'content' key

        Returns:
            The bash command to execute

        Raises:
            FormatError: If not exactly one bash block found
            Submitted: If the task is complete
        """
        content = response.get("content", "")

        # Check for task completion
        if "TASK_COMPLETE" in content:
            raise Submitted(content)

        # Extract bash blocks
        actions = re.findall(self.config.action_regex, content, re.DOTALL)

        if len(actions) == 1:
            return actions[0].strip()

        raise FormatError(self.render_template(self.config.format_error_template, actions=actions))

    def execute_action(self, action: str) -> dict[str, Any]:
        """Execute the bash command and check for loops.

        Args:
            action: The bash command to execute

        Returns:
            Dict with 'output', 'returncode', and 'action' keys
        """
        # Loop detection
        self.action_history.append(action)
        if len(self.action_history) >= self.config.loop_threshold:
            recent = self.action_history[-self.config.loop_threshold :]
            if len(set(recent)) == 1:
                raise LoopDetected(f"Agent repeating same action: {action}")

        try:
            output = self.env.execute(action)
        except TimeoutError as e:
            raise ExecutionTimeoutError(
                self.render_template(self.config.timeout_template, action=action, output=str(e))
            )

        return output | {"action": action}
