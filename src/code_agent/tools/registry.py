"""Tool Registry for extensible tool management.

This module provides a decorator-based system for registering tools that
the agent can use. It demonstrates "platform thinking" - making it easy
for other teams to extend the system.

Interview talking points:
- Platform Thinking: Internal teams can add domain-specific tools
  - Ansible team adds Ansible-specific tools
  - OpenShift team adds cluster tools
  - Each team extends without modifying core code

- MCP (Model Context Protocol):
  - Open standard for tool integration
  - Red Hat exploring MCP for Ansible Lightspeed
  - Tools can be provided by external servers

Example usage:
    from code_agent.tools import tool, ToolRegistry

    @tool(name="read_file", description="Read contents of a file")
    def read_file(path: str) -> str:
        return open(path).read()

    registry = ToolRegistry()
    registry.register(read_file)
    result = registry.execute("read_file", {"path": "config.yaml"})
"""

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Tool:
    """Represents a registered tool."""

    name: str
    description: str
    func: Callable
    parameters: dict[str, Any] = field(default_factory=dict)

    def execute(self, **kwargs) -> Any:
        """Execute the tool with given arguments."""
        return self.func(**kwargs)

    def to_schema(self) -> dict[str, Any]:
        """Convert to JSON schema format (for tool-calling APIs)."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self.parameters,
            },
        }


def tool(name: str, description: str, parameters: dict[str, Any] | None = None):
    """Decorator to mark a function as a tool.

    Args:
        name: Tool name (used to invoke it)
        description: Human-readable description
        parameters: Optional JSON schema for parameters

    Example:
        @tool("read_file", "Read contents of a file")
        def read_file(path: str) -> str:
            return open(path).read()
    """

    def decorator(func: Callable) -> Callable:
        # Attach metadata to the function
        func._tool_metadata = Tool(
            name=name,
            description=description,
            func=func,
            parameters=parameters or {},
        )
        return func

    return decorator


class ToolRegistry:
    """Registry for managing available tools.

    This is the core extension point for the platform. Teams can:
    1. Define tools with @tool decorator
    2. Register them with the registry
    3. Agent can discover and use them

    Example:
        registry = ToolRegistry()
        registry.register(read_file)
        registry.register(write_file)

        # List available tools
        for name, tool in registry.tools.items():
            print(f"{name}: {tool.description}")

        # Execute a tool
        result = registry.execute("read_file", path="config.yaml")
    """

    def __init__(self):
        self.tools: dict[str, Tool] = {}

    def register(self, func: Callable) -> None:
        """Register a tool function.

        The function must be decorated with @tool.
        """
        if not hasattr(func, "_tool_metadata"):
            raise ValueError(f"Function {func.__name__} is not decorated with @tool")

        tool_meta: Tool = func._tool_metadata
        self.tools[tool_meta.name] = tool_meta

    def register_many(self, *funcs: Callable) -> None:
        """Register multiple tools at once."""
        for func in funcs:
            self.register(func)

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self.tools.get(name)

    def execute(self, name: str, **kwargs) -> Any:
        """Execute a tool by name with given arguments.

        Raises:
            KeyError: If tool not found
        """
        tool = self.tools.get(name)
        if tool is None:
            raise KeyError(f"Tool '{name}' not found. Available: {list(self.tools.keys())}")
        return tool.execute(**kwargs)

    def list_tools(self) -> list[dict[str, str]]:
        """List all registered tools with their descriptions."""
        return [{"name": t.name, "description": t.description} for t in self.tools.values()]

    def get_tools_prompt(self) -> str:
        """Generate a prompt section describing available tools.

        This can be injected into the system prompt so the LLM
        knows what tools are available.
        """
        lines = ["Available tools:"]
        for tool in self.tools.values():
            lines.append(f"  - {tool.name}: {tool.description}")
        return "\n".join(lines)

    def to_schemas(self) -> list[dict[str, Any]]:
        """Convert all tools to JSON schemas (for tool-calling APIs)."""
        return [tool.to_schema() for tool in self.tools.values()]
