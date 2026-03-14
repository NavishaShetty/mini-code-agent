# Tool Calling in AI Agents: Complete Guide

> Notes on implementing tool capabilities in AI coding agents.

---

## Table of Contents

1. [Agent Capabilities: Bash-Only vs Tools](#1-agent-capabilities-bash-only-vs-tools)
2. [Two Ways to Implement Tool Calling](#2-two-ways-to-implement-tool-calling)
3. [From-Scratch Implementation Details](#3-from-scratch-implementation-details)
4. [Global vs Instance-Based Registration](#4-global-vs-instance-based-registration)
5. [Framework Implementation Details](#5-framework-implementation-details)
6. [Comparison: From-Scratch vs Frameworks](#6-comparison-from-scratch-vs-frameworks)
7. [Tool-Calling API vs Bash-Only Parsing](#7-tool-calling-api-vs-bash-only-parsing)
8. [LLM Tool-Calling Support](#8-llm-tool-calling-support)
9. [Interview Preparation Topics](#9-interview-preparation-topics)

---

## 1. Agent Capabilities: Bash-Only vs Tools

### Base Capability: Bash-Only

The simplest agent just parses bash commands from LLM output:

```
LLM Output: "```bash\nls -la\n```"
     │
     ▼
Agent: regex extracts "ls -la"
     │
     ▼
Agent: subprocess.run("ls -la")
```

**Good for**: General system commands, quick tasks, works with any LLM.

### Extended Capability: Tool Registry

Tools are **in addition to** bash-only, not a replacement:

```
┌─────────────────────────────────────────────────────────────────┐
│                    AGENT CAPABILITIES                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  BASE: Bash-Only                                                │
│  ────────────────                                                │
│  • Parse ```bash``` blocks with regex                           │
│  • Execute via subprocess                                        │
│  • Works with ANY LLM                                           │
│  • Covers ~80% of use cases                                     │
│                                                                  │
│  EXTENSION: Tool Registry                                       │
│  ────────────────────────                                        │
│  • Structured Python functions                                  │
│  • Domain-specific capabilities                                  │
│  • Type-safe, validated inputs                                  │
│  • Extensible by other teams                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### When to Use Each

| Use Bash-Only | Use Tool Registry |
|---------------|-------------------|
| General file operations | Domain-specific actions (Ansible, OpenShift) |
| Git commands | Actions that aren't bash commands |
| Running tests | Complex operations with validation |
| Quick ad-hoc tasks | Reusable, documented operations |
| Maximum LLM compatibility | When you need type safety |

---

## 2. Two Ways to Implement Tool Calling

### Option A: From Scratch (Our Approach)

Build your own tool system with Python:

```python
# 1. Define tools with @tool decorator
@tool(name="read_file", description="Read a file")
def read_file(path: str) -> str:
    return open(path).read()

# 2. Create registry and register tools
registry = ToolRegistry()
registry.register(read_file)

# Or use get_builtin_tools() for convenience:
from code_agent.tools import get_builtin_tools
registry.register_many(*get_builtin_tools())

# 3. Execute tools
result = registry.execute("read_file", path="config.yaml")
```

**Pros**: Full control, understand everything, no dependencies
**Cons**: More code to write and maintain

### Option B: Use Frameworks (SmolAgents, LangChain)

Let the framework handle registration and execution:

```python
# SmolAgents
from smolagents import tool, CodeAgent, LiteLLMModel

@tool
def read_file(path: str) -> str:
    """Read contents of a file."""
    return open(path).read()

agent = CodeAgent(tools=[read_file], model=LiteLLMModel(...))
agent.run("Read the config file")
```

```python
# LangChain
from langchain.tools import tool
from langchain.agents import create_react_agent

@tool
def read_file(path: str) -> str:
    """Read contents of a file."""
    return open(path).read()

agent = create_react_agent(llm, tools=[read_file], prompt=...)
```

**Pros**: Less code, handles edge cases, battle-tested
**Cons**: Black box, less control, framework lock-in

---

## 3. From-Scratch Implementation Details

### Components You Need to Build

```
┌─────────────────────────────────────────────────────────────────┐
│                FROM-SCRATCH TOOL SYSTEM                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐                                            │
│  │  @tool Decorator │  Attaches metadata to functions           │
│  └────────┬────────┘                                            │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────┐                                            │
│  │  Tool Dataclass │  Container for name, description, func     │
│  └────────┬────────┘                                            │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────┐                                            │
│  │  ToolRegistry   │  Stores tools, provides execute()          │
│  └────────┬────────┘                                            │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────┐                                            │
│  │  Built-in Tools │  read_file, write_file, glob, grep, etc   │
│  └─────────────────┘                                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### The @tool Decorator Explained

```python
def tool(name: str, description: str, parameters: dict = None):
    def decorator(func):
        # Attach metadata to the function (doesn't change behavior)
        func._tool_metadata = Tool(
            name=name,
            description=description,
            func=func,
            parameters=parameters or {},
        )
        return func
    return decorator
```

**Key insight**: The decorator ONLY attaches metadata. Registration is separate.

### The Registry Pattern

```python
class ToolRegistry:
    def __init__(self):
        self.tools: dict[str, Tool] = {}

    def register(self, func):
        # Extract metadata attached by @tool decorator
        metadata = func._tool_metadata
        self.tools[metadata.name] = metadata

    def execute(self, name, **kwargs):
        return self.tools[name].func(**kwargs)
```

### Convenience: get_builtin_tools()

```python
def get_builtin_tools():
    return [read_file, write_file, glob_files, grep_search, list_directory]

# Usage: register all at once
registry = ToolRegistry()
registry.register_many(*get_builtin_tools())
```

---

## 4. Global vs Instance-Based Registration

### The Key Question

When you use `@tool` decorator, where does the tool go? There are two approaches:

### Approach 1: Global Registration (SmolAgents Style)

```python
# Somewhere inside the framework, there's a global list:
_GLOBAL_TOOLS = []

def tool(func):
    # Decorator automatically adds to GLOBAL list
    _GLOBAL_TOOLS.append(func)
    return func

@tool
def read_file(path: str) -> str:
    """Read a file."""
    return open(path).read()

# The tool is now in _GLOBAL_TOOLS automatically!
# Any agent in the program can access it
```

**"Global" means**: One shared list for the entire program. The decorator itself does the registration.

### Approach 2: Instance-Based Registration (Our Approach)

```python
# NO global list - each registry is independent

@tool(name="read_file", description="Read a file")
def read_file(path: str) -> str:
    return open(path).read()

# The @tool decorator ONLY attaches metadata
# Nothing is registered anywhere yet!

# You create registry instances
registry1 = ToolRegistry()
registry2 = ToolRegistry()

# You explicitly choose what goes where
registry1.register(read_file)
# registry2 doesn't have read_file - it's separate!
```

**"Instance-based" means**: Each registry is separate. You control what goes in each one.

### Visual Comparison

```
GLOBAL REGISTRATION (SmolAgents):
─────────────────────────────────

    @tool                     ┌─────────────────────┐
    def read_file(): ...  ───►│  GLOBAL_TOOLS list  │ (one shared list)
                              │  • read_file        │
    @tool                     │  • write_file       │
    def write_file(): ... ───►│  • grep_search      │
                              └─────────────────────┘
    @tool                              │
    def grep_search(): ... ───►        │
                                       ▼
                              All agents see all tools



INSTANCE-BASED REGISTRATION (Our Approach):
───────────────────────────────────────────

    @tool(...)
    def read_file(): ...     ─── metadata attached, NOT registered

    @tool(...)
    def write_file(): ...    ─── metadata attached, NOT registered

    @tool(...)
    def ansible_lint(): ...  ─── metadata attached, NOT registered


    registry_general = ToolRegistry()
    registry_general.register(read_file)
    registry_general.register(write_file)

    registry_ansible = ToolRegistry()
    registry_ansible.register(ansible_lint)


    ┌─────────────────────┐    ┌─────────────────────┐
    │  registry_general   │    │  registry_ansible   │
    │  • read_file        │    │  • ansible_lint     │
    │  • write_file       │    │                     │
    └─────────────────────┘    └─────────────────────┘
           │                           │
           ▼                           ▼
    General-purpose agent       Ansible-specific agent
```

### Comparison Table

| Global Registration | Instance-Based Registration |
|--------------------|-----------------------------|
| Simpler (automatic) | More explicit (manual) |
| One shared list | Multiple independent registries |
| All tools visible everywhere | Control what each agent sees |
| Can't have different tool sets | Different agents = different tools |
| Hard to test in isolation | Easy to test with mock registry |

### Why Instance-Based Matters (Red Hat Example)

Imagine Red Hat Lightspeed with different specialized agents:

```python
# General coding agent - basic tools only
general_registry = ToolRegistry()
general_registry.register(read_file)
general_registry.register(write_file)

# Ansible agent - shouldn't see OpenShift tools
ansible_registry = ToolRegistry()
ansible_registry.register(read_file)
ansible_registry.register(ansible_lint)
ansible_registry.register(molecule_test)

# OpenShift agent - shouldn't see Ansible tools
openshift_registry = ToolRegistry()
openshift_registry.register(oc_get_pods)
openshift_registry.register(oc_describe)

# Each agent only sees its own tools!
general_agent = Agent(registry=general_registry)
ansible_agent = Agent(registry=ansible_registry)
openshift_agent = Agent(registry=openshift_registry)
```

With **global** registration, ALL agents would see ALL tools - confusing for the LLM.

### Why Register At All?

You might ask: "Why not just call the function directly?"

```python
# Without registry - hardcoded function dispatch
if action == "read_file":
    result = read_file(path)
elif action == "write_file":
    result = write_file(path, content)
elif action == "grep":
    result = grep_search(pattern, path)
# ... endless if/elif chain
```

```python
# With registry - dynamic dispatch by name
result = registry.execute(action_name, **kwargs)
# One line handles any tool!
```

**The registry provides:**

| Benefit | Description |
|---------|-------------|
| **Dynamic lookup** | Call tools by name (string from LLM) |
| **Discovery** | List what tools are available |
| **Prompt generation** | Tell LLM what tools exist |
| **Separation** | Different registries for different agents |
| **Testability** | Mock registry for unit tests |

### Summary

| Term | Meaning |
|------|---------|
| **Global registration** | One shared list, decorator auto-registers |
| **Instance registration** | Each registry is separate, manual registration |
| **Our approach** | Instance-based (more control, more explicit) |
| **Why register at all** | Dynamic lookup, discovery, prompt generation |

---

## 5. Framework Implementation Details

### SmolAgents (HuggingFace)

```python
from smolagents import tool, CodeAgent, LiteLLMModel

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    # Description comes from docstring!
    return do_web_search(query)

# Framework handles:
# - Tool registration (automatic from @tool)
# - Tool schema generation
# - Agent loop
# - Parsing and execution
# - Error handling

agent = CodeAgent(
    tools=[search_web],
    model=LiteLLMModel(model_id="anthropic/claude-3-5-sonnet")
)
agent.run("Find the latest Python version")
```

### LangChain / LangGraph

```python
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_anthropic import ChatAnthropic

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))

llm = ChatAnthropic(model="claude-3-5-sonnet")
agent = create_react_agent(llm, tools=[calculator], prompt=react_prompt)
executor = AgentExecutor(agent=agent, tools=[calculator])
executor.invoke({"input": "What is 25 * 4?"})
```

### What Frameworks Handle For You

| Feature | You Build | Framework Handles |
|---------|-----------|-------------------|
| Tool registration | ✅ Manual | ✅ Automatic |
| Schema generation | ✅ Manual | ✅ Automatic |
| Agent loop | ✅ Manual | ✅ Automatic |
| Parsing LLM output | ✅ Manual | ✅ Automatic |
| Error handling | ✅ Manual | ✅ Automatic |
| Retry logic | ✅ Manual | ✅ Automatic |
| Context management | ✅ Manual | ✅ Automatic |

---

## 6. Comparison: From-Scratch vs Frameworks

### When to Build From Scratch

- ✅ Learning/understanding the fundamentals
- ✅ Interview preparation (shows deep knowledge)
- ✅ Need maximum control
- ✅ Simple use case, don't need full framework
- ✅ Building a platform others will extend
- ✅ Need to work with models that don't support tool-calling

### When to Use Frameworks

- ✅ Production applications
- ✅ Rapid prototyping
- ✅ Complex multi-step agents
- ✅ Need battle-tested error handling
- ✅ Using models with tool-calling API support
- ✅ Don't need to understand internals

### Interview Positioning

> "I've used SmolAgents for rapid prototyping - it handles tool registration, the agent loop, and error handling automatically. But I also built a tool system from scratch to understand the fundamentals: the decorator pattern for metadata, the registry for tool management, and how parsing and execution work. This helps me debug issues and customize behavior when frameworks fall short."

---

## 7. Tool-Calling API vs Bash-Only Parsing

### Two Ways LLMs Can Specify Actions

**Method 1: Bash-Only (Text Parsing)**
```
LLM Output: "I'll read the file.\n```bash\ncat config.yaml\n```"
Agent: Uses regex to extract command
Agent: subprocess.run("cat config.yaml")
```

**Method 2: Tool-Calling API (Structured JSON)**
```python
# You define tools as JSON schema
tools = [{
    "name": "read_file",
    "parameters": {"path": {"type": "string"}}
}]

# LLM returns structured JSON
response.tool_calls = [
    {"name": "read_file", "arguments": {"path": "config.yaml"}}
]
```

### Comparison

| Aspect | Bash-Only | Tool-Calling API |
|--------|-----------|------------------|
| LLM output | Free text | Structured JSON |
| Parsing | Regex | JSON parsing |
| Works with | Any LLM | Only supporting LLMs |
| Validation | None | Schema validation |
| Debugging | Easy (read text) | Harder (inspect JSON) |

### Which LLMs Support Tool-Calling?

| Provider | Tool-Calling Support |
|----------|---------------------|
| OpenAI (GPT-4) | ✅ Yes |
| Anthropic (Claude) | ✅ Yes |
| Google (Gemini) | ✅ Yes |
| Self-hosted Llama | ❌ Usually No |
| Self-hosted Mistral | ❌ Usually No |
| vLLM served models | ❌ Usually No |

**Key insight for Red Hat**: Self-hosted models often lack tool-calling, making bash-only approach valuable.

---

## 8. LLM Tool-Calling Support

### Why Self-Hosted Models Often Lack Tool-Calling

Tool-calling requires the full stack:

```
┌─────────────────────────────────────────┐
│  1. Model Training                      │  Fine-tuned to output JSON?
├─────────────────────────────────────────┤
│  2. Prompt Format                       │  Trained on specific format?
├─────────────────────────────────────────┤
│  3. Serving Layer (vLLM)               │  Parses tool calls?
├─────────────────────────────────────────┤
│  4. API Layer                          │  Exposes tool endpoints?
└─────────────────────────────────────────┘

API Providers: Handle all 4 layers ✅
Self-Hosted: Usually only layers 1-2 ⚠️
```

### Platform Implications

```
Red Hat Lightspeed Platform
├── Team A: Uses Claude API → Tool-calling works ✅
├── Team B: Uses Llama on vLLM → Tool-calling fails ❌
└── Team C: Uses Granite → Maybe works ⚠️

Solution: Bash-only approach works for ALL teams
```

---

## 9. Interview Preparation Topics

### Must-Know Topics

#### A. Agent Fundamentals
- [ ] **ReAct Pattern**: Think → Act → Observe loop
- [ ] **Agent Loop**: How while True + query + parse + execute works
- [ ] **Message History**: How context builds up over iterations
- [ ] **State Management**: Linear history vs complex state machines

#### B. Tool Implementation
- [ ] **Decorator Pattern**: How @tool attaches metadata
- [ ] **Registry Pattern**: Store, discover, execute tools
- [ ] **Tool Design**: Single responsibility, error as strings, bounded output
- [ ] **Built-in vs Custom Tools**: Core tools vs domain-specific

#### C. Tool-Calling Approaches
- [ ] **Bash-Only Parsing**: Regex extraction, subprocess execution
- [ ] **Tool-Calling API**: JSON schemas, structured responses
- [ ] **When to Use Each**: Model support, validation needs, debugging

#### D. Framework Knowledge
- [ ] **SmolAgents**: HuggingFace's approach, automatic registration
- [ ] **LangChain/LangGraph**: Tools, agents, chains
- [ ] **From-Scratch vs Framework**: Trade-offs, when to use each

#### E. LLM Considerations
- [ ] **Which LLMs Support Tool-Calling**: API providers vs self-hosted
- [ ] **Why Self-Hosted Lacks Support**: Training, serving layer, API
- [ ] **Model-Agnostic Design**: Why bash-only matters for platforms

#### F. Production Concerns
- [ ] **Failure Modes**: Loops, timeouts, parse errors, context overflow
- [ ] **Cost Tracking**: Token counting, cost per call
- [ ] **Retry Logic**: Exponential backoff, which errors to retry
- [ ] **Output Limits**: Truncation, result limits, bounded responses

### Red Hat Specific Topics

#### G. Platform Thinking
- [ ] **Extensibility**: How teams add their own tools
- [ ] **MCP (Model Context Protocol)**: Open standard for tool integration
- [ ] **Multi-Model Support**: Why model-agnostic matters

#### H. Red Hat Products
- [ ] **Ansible Lightspeed**: Context from VS Code, code completion
- [ ] **OpenShift Lightspeed**: Cluster interaction, BYOK docs
- [ ] **InstructLab**: Fine-tuning with synthetic data
- [ ] **vLLM/KServe**: Model serving on Kubernetes

### Quick Reference: Key Talking Points

| Topic | Key Point |
|-------|-----------|
| Tool-calling | "Addition to bash-only, not replacement" |
| From-scratch | "Decorator attaches metadata, registry stores and executes" |
| Frameworks | "Handle registration, loop, parsing, errors automatically" |
| LLM support | "API providers support tool-calling, self-hosted often don't" |
| Platform | "Bash-only is model-agnostic, critical for diverse teams" |
| Trade-off | "Frameworks for speed, from-scratch for control and understanding" |

---

## Code Quick Reference

### From-Scratch Minimal Example

```python
from dataclasses import dataclass
from typing import Callable, Any

@dataclass
class Tool:
    name: str
    description: str
    func: Callable

def tool(name: str, description: str):
    def decorator(func):
        func._tool_metadata = Tool(name, description, func)
        return func
    return decorator

class ToolRegistry:
    def __init__(self):
        self.tools = {}

    def register(self, func):
        meta = func._tool_metadata
        self.tools[meta.name] = meta

    def execute(self, name: str, **kwargs) -> Any:
        return self.tools[name].func(**kwargs)

# Usage
@tool("greet", "Say hello")
def greet(name: str) -> str:
    return f"Hello, {name}!"

registry = ToolRegistry()
registry.register(greet)
print(registry.execute("greet", name="World"))  # "Hello, World!"
```

### SmolAgents Minimal Example

```python
from smolagents import tool, CodeAgent, LiteLLMModel

@tool
def greet(name: str) -> str:
    """Say hello to someone."""
    return f"Hello, {name}!"

agent = CodeAgent(
    tools=[greet],
    model=LiteLLMModel(model_id="anthropic/claude-3-5-sonnet")
)
agent.run("Greet the world")
```

---

## See Also

- `src/code_agent/tools/registry.py` - Our registry implementation
- `src/code_agent/tools/builtins.py` - Built-in tool implementations
- `docs/STUDY_NOTES.md` - General study notes
- `docs/LITELLM_EXPLAINED.md` - LLM wrapper notes
