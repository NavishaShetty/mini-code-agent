# Entry Points in Python Projects: Complete Guide

> Notes on CLI design, argument parsing, and programmatic usage patterns.

---

## Table of Contents

1. [What Are Entry Points?](#1-what-are-entry-points)
2. [CLI Entry Point (main.py)](#2-cli-entry-point-mainpy)
3. [Programmatic Entry Point (examples/)](#3-programmatic-entry-point-examples)
4. [How python -m Works](#4-how-python--m-works)
5. [Argument Parsing with argparse](#5-argument-parsing-with-argparse)
6. [Code Walkthrough](#6-code-walkthrough)
7. [Interview Preparation](#7-interview-preparation)

---

## 1. What Are Entry Points?

### Definition

Entry points are **how users start your application**. They define the boundary between your code and the outside world.

### Two Types of Entry Points

```
┌─────────────────────────────────────────────────────────────────────┐
│                      ENTRY POINT TYPES                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  CLI ENTRY POINT                    PROGRAMMATIC ENTRY POINT        │
│  ────────────────                   ─────────────────────────       │
│                                                                      │
│  For: End users, operators          For: Developers integrating     │
│                                                                      │
│  How: Command line flags            How: Import and call functions  │
│                                                                      │
│  Example:                           Example:                        │
│  $ python -m code_agent.main \       from code_agent import Agent     │
│      --task "List files" \          agent = Agent(model, env)       │
│      --mode confirm                 agent.run("List files")         │
│                                                                      │
│  Flexibility: Command-line args     Flexibility: Full Python        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Why Both Matter

| Use Case | Entry Point | Why |
|----------|-------------|-----|
| Quick testing | CLI | Fast, no code needed |
| CI/CD pipelines | CLI | Easy to script |
| Building a UI | Programmatic | Need full control |
| Integrating into another tool | Programmatic | Import as library |
| Learning/debugging | Either | Depends on preference |

### Real-World Examples

| Tool | CLI | Programmatic |
|------|-----|--------------|
| **pytest** | `pytest tests/` | `pytest.main(["tests/"])` |
| **black** | `black myfile.py` | `black.format_file_contents(...)` |
| **kubectl** | `kubectl get pods` | Go library imports |
| **git** | `git status` | libgit2 bindings |

---

## 2. CLI Entry Point (main.py)

### What It Does

Our CLI (`main.py`) provides a command-line interface to run the agent:

```bash
python -m code_agent.main --task "List Python files" --mode confirm
```

### The Structure

```
┌─────────────────────────────────────────────────────────────────────┐
│                      CLI STRUCTURE                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  main.py                                                            │
│  │                                                                   │
│  ├── Imports                                                        │
│  │   └── argparse, sys, agent components                           │
│  │                                                                   │
│  ├── main() function                                                │
│  │   │                                                               │
│  │   ├── 1. Define argument parser                                  │
│  │   │      └── --task, --mode, --model, --step-limit, etc.        │
│  │   │                                                               │
│  │   ├── 2. Parse arguments                                         │
│  │   │      └── args = parser.parse_args()                         │
│  │   │                                                               │
│  │   ├── 3. Initialize components                                   │
│  │   │      └── model, env, agent                                   │
│  │   │                                                               │
│  │   ├── 4. Run agent                                               │
│  │   │      └── status, message = agent.run(args.task)             │
│  │   │                                                               │
│  │   ├── 5. Report results                                          │
│  │   │      └── Print status, stats, cost                          │
│  │   │                                                               │
│  │   └── 6. Return exit code                                        │
│  │          └── 0 for success, non-zero for failure                │
│  │                                                                   │
│  └── if __name__ == "__main__":                                     │
│      └── sys.exit(main())                                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Available Arguments

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--task` | `-t` | Required | - | The task for the agent |
| `--mode` | `-m` | Choice | `confirm` | human/confirm/yolo |
| `--model` | - | String | `claude-sonnet-4-20250514` | LLM model |
| `--step-limit` | - | Int | 20 | Max iterations |
| `--cost-limit` | - | Float | 1.0 | Max cost in USD |
| `--no-auto-approve` | - | Flag | False | Disable safe auto-approve |
| `--verbose` | `-v` | Flag | False | Show stack traces |

### Example Usage

```bash
# Basic usage
python -m code_agent.main --task "List all Python files"

# With all options
python -m code_agent.main \
    --task "Refactor the main function" \
    --mode yolo \
    --model gpt-4o \
    --step-limit 50 \
    --cost-limit 2.0 \
    --verbose

# Short flags
python -m code_agent.main -t "Find bugs" -m confirm -v
```

### Exit Codes

```
┌─────────────────────────────────────────────────────────────────────┐
│                      EXIT CODES                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  0   = Success (task completed)                                     │
│  1   = Failure (error or task not completed)                        │
│  130 = Interrupted (Ctrl+C, SIGINT)                                 │
│                                                                      │
│  Why exit codes matter:                                             │
│  ─────────────────────                                              │
│                                                                      │
│  # In shell scripts:                                                │
│  python -m code_agent.main --task "..." && echo "Success!"          │
│                                                                      │
│  # In CI/CD:                                                        │
│  if agent fails, pipeline stops                                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Programmatic Entry Point (examples/)

### What It Does

The example script shows how to use the agent as a **library** in your own code:

```python
from code_agent import InteractiveAgent, ExecutionMode, LocalEnvironment, LiteLLMModel

model = LiteLLMModel(model_name="claude-sonnet-4-20250514")
env = LocalEnvironment()
agent = InteractiveAgent(model, env, mode=ExecutionMode.CONFIRM)

status, message = agent.run("List all Python files")
```

### Why This Matters

```
┌─────────────────────────────────────────────────────────────────────┐
│                 CLI vs PROGRAMMATIC                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  CLI:                                                               │
│  ────                                                               │
│  - Arguments as strings                                             │
│  - Output goes to stdout                                            │
│  - Can't easily process results                                     │
│  - Good for: humans, shell scripts                                  │
│                                                                      │
│  Programmatic:                                                      │
│  ─────────────                                                      │
│  - Full Python objects                                              │
│  - Return values you can use                                        │
│  - Can inspect agent state                                          │
│  - Good for: building UIs, integrations, testing                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Example: Building on Top

```python
# Use agent in a web API
from flask import Flask, request, jsonify
from code_agent import InteractiveAgent, ExecutionMode, LocalEnvironment, LiteLLMModel

app = Flask(__name__)

@app.route("/run", methods=["POST"])
def run_agent():
    task = request.json["task"]

    model = LiteLLMModel(model_name="claude-sonnet-4-20250514")
    env = LocalEnvironment()
    agent = InteractiveAgent(model, env, mode=ExecutionMode.YOLO)

    status, message = agent.run(task)

    return jsonify({
        "status": status,
        "message": message,
        "cost": model.get_stats()["total_cost"],
    })
```

### Example: Custom Workflow

```python
# Run multiple tasks sequentially
from code_agent import InteractiveAgent, ExecutionMode, LocalEnvironment, LiteLLMModel

def run_tasks(tasks: list[str]):
    model = LiteLLMModel(model_name="claude-sonnet-4-20250514")
    env = LocalEnvironment()
    agent = InteractiveAgent(model, env, mode=ExecutionMode.YOLO)

    results = []
    for task in tasks:
        status, message = agent.run(task)
        results.append({"task": task, "status": status})

        # Stop if any task fails
        if status != "Submitted":
            break

    return results

# Usage
results = run_tasks([
    "Find all TODO comments",
    "Create a summary of the codebase",
    "Suggest improvements",
])
```

---

## 4. How python -m Works

### The Command

```bash
python -m code_agent.main --task "..."
       ↑
       -m means "run as module"
```

### What Happens

```
┌─────────────────────────────────────────────────────────────────────┐
│                 python -m code_agent.main                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. Python looks for code_agent/main.py                              │
│                                                                      │
│  2. Python sets __name__ = "__main__"                               │
│                                                                      │
│  3. Python executes the file                                        │
│                                                                      │
│  4. The if __name__ == "__main__": block runs                       │
│                                                                      │
│  5. sys.exit(main()) is called                                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Why -m Instead of Direct Path?

```bash
# These are different!

python -m code_agent.main     # Runs as module (correct imports)
python src/code_agent/main.py # Runs as script (import errors!)
```

**The difference:**
- `-m` sets up Python's module system correctly
- Direct path doesn't know about the package structure
- `-m` allows relative imports to work

### The if __name__ == "__main__" Guard

```python
if __name__ == "__main__":
    sys.exit(main())
```

**What this does:**
- When run directly: `__name__` is `"__main__"`, so `main()` executes
- When imported: `__name__` is `"code_agent.main"`, so `main()` doesn't execute

**Why it matters:**
```python
# This is safe to import without running:
from code_agent.main import main  # main() won't execute

# But you can still run it:
if __name__ == "__main__":
    main()  # Only runs when executed directly
```

---

## 5. Argument Parsing with argparse

### Basic Pattern

```python
import argparse

def main():
    # 1. Create parser
    parser = argparse.ArgumentParser(description="My tool")

    # 2. Add arguments
    parser.add_argument("--name", required=True, help="Your name")
    parser.add_argument("--count", type=int, default=1, help="Count")

    # 3. Parse
    args = parser.parse_args()

    # 4. Use
    print(f"Hello {args.name}! " * args.count)
```

### Argument Types

```python
# Required argument
parser.add_argument("--task", "-t", required=True, help="...")

# Optional with default
parser.add_argument("--model", default="gpt-4", help="...")

# Integer
parser.add_argument("--limit", type=int, default=10, help="...")

# Float
parser.add_argument("--cost", type=float, default=1.0, help="...")

# Boolean flag (store_true = False by default, True if flag present)
parser.add_argument("--verbose", "-v", action="store_true", help="...")

# Choice from list
parser.add_argument("--mode", choices=["a", "b", "c"], default="a", help="...")
```

### Our Arguments Explained

```python
parser.add_argument(
    "--task", "-t",           # Long and short form
    required=True,            # Must be provided
    help="The task for the agent to complete",
)

parser.add_argument(
    "--mode", "-m",
    choices=["human", "confirm", "yolo"],  # Only these values allowed
    default="confirm",                      # If not provided
    help="Execution mode (default: confirm)",
)

parser.add_argument(
    "--no-auto-approve",
    action="store_true",      # Flag: present = True, absent = False
    help="Disable auto-approval of safe commands",
)
```

### Accessing Arguments

```python
args = parser.parse_args()

# Access by name (dashes become underscores)
print(args.task)           # --task value
print(args.mode)           # --mode value
print(args.no_auto_approve)  # --no-auto-approve flag (True/False)
print(args.step_limit)     # --step-limit value
```

---

## 6. Code Walkthrough

### main.py Complete Flow

```python
"""Main CLI entry point for the code agent."""

import argparse
import sys

from code_agent.agent.interactive import ExecutionMode, InteractiveAgent
from code_agent.environment.local import LocalEnvironment
from code_agent.model.litellm import LiteLLMModel


def main():
    # ┌─────────────────────────────────────────┐
    # │ STEP 1: Define argument parser          │
    # └─────────────────────────────────────────┘
    parser = argparse.ArgumentParser(
        description="Code Agent - ReAct pattern demonstration",
    )

    parser.add_argument("--task", "-t", required=True, help="...")
    parser.add_argument("--mode", "-m", choices=["human", "confirm", "yolo"], default="confirm")
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--step-limit", type=int, default=20)
    parser.add_argument("--cost-limit", type=float, default=1.0)
    parser.add_argument("--no-auto-approve", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")

    # ┌─────────────────────────────────────────┐
    # │ STEP 2: Parse arguments                 │
    # └─────────────────────────────────────────┘
    args = parser.parse_args()

    # ┌─────────────────────────────────────────┐
    # │ STEP 3: Show initialization info        │
    # └─────────────────────────────────────────┘
    print(f"🚀 Initializing agent...")
    print(f"   Model: {args.model}")
    print(f"   Mode: {args.mode}")
    print(f"   Limits: {args.step_limit} steps, ${args.cost_limit}")

    try:
        # ┌─────────────────────────────────────────┐
        # │ STEP 4: Initialize components           │
        # └─────────────────────────────────────────┘
        model = LiteLLMModel(model_name=args.model)
        env = LocalEnvironment()
        agent = InteractiveAgent(
            model,
            env,
            mode=ExecutionMode(args.mode),
            auto_approve_safe=not args.no_auto_approve,
            step_limit=args.step_limit,
            cost_limit=args.cost_limit,
        )

        # ┌─────────────────────────────────────────┐
        # │ STEP 5: Run the agent                   │
        # └─────────────────────────────────────────┘
        status, message = agent.run(args.task)

        # ┌─────────────────────────────────────────┐
        # │ STEP 6: Report results                  │
        # └─────────────────────────────────────────┘
        print("\n" + "=" * 50)
        print(f"Status: {status}")
        print(f"Message: {message[:500]}...")

        stats = model.get_stats()
        print(f"\nStats:")
        print(f"  API calls: {stats.get('n_calls', 0)}")
        print(f"  Total cost: ${stats.get('total_cost', 0):.4f}")
        print(f"  Tokens: {stats.get('total_tokens', 0)}")

        # ┌─────────────────────────────────────────┐
        # │ STEP 7: Return exit code                │
        # └─────────────────────────────────────────┘
        return 0 if status == "Submitted" else 1

    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
        return 130  # Standard Unix code for SIGINT

    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


# ┌─────────────────────────────────────────┐
# │ Only run main() when executed directly  │
# └─────────────────────────────────────────┘
if __name__ == "__main__":
    sys.exit(main())
```

### Example Script Flow

```python
"""Simple example of running the code agent."""

from code_agent import InteractiveAgent, ExecutionMode, LocalEnvironment, LiteLLMModel


def main():
    # ┌─────────────────────────────────────────┐
    # │ Create components (same as CLI)         │
    # └─────────────────────────────────────────┘
    print("🚀 Initializing agent...")

    model = LiteLLMModel(model_name="claude-sonnet-4-20250514")
    env = LocalEnvironment()
    agent = InteractiveAgent(
        model,
        env,
        mode=ExecutionMode.CONFIRM,
        step_limit=10,
        cost_limit=0.50,
    )

    # ┌─────────────────────────────────────────┐
    # │ Run with hardcoded task                 │
    # └─────────────────────────────────────────┘
    task = "List all Python files and count them"

    print(f"\n📋 Task: {task}\n")
    status, message = agent.run(task)

    # ┌─────────────────────────────────────────┐
    # │ Process results (you have full access)  │
    # └─────────────────────────────────────────┘
    print("\n" + "=" * 50)
    print(f"Status: {status}")
    print(f"Stats: {model.get_stats()}")

    # You can also access:
    # - agent.messages (full conversation history)
    # - agent.action_history (all commands run)
    # - model.get_stats() (cost, tokens, calls)


if __name__ == "__main__":
    main()
```

---

## 7. Interview Preparation

### Key Concepts

| Concept | Explanation |
|---------|-------------|
| **Entry point** | How users start your application |
| **CLI** | Command-line interface using argparse |
| **python -m** | Run as module (correct import paths) |
| **if __name__ == "__main__"** | Guard to prevent execution on import |
| **Exit codes** | 0 = success, non-zero = failure |
| **Programmatic usage** | Import and use as library |

### Why Both Entry Points?

> "We provide both CLI and programmatic entry points because they serve different users. Operators and CI/CD pipelines use the CLI - it's easy to script and doesn't require Python knowledge. Developers building on top of the agent use the programmatic API - they get full access to objects, can inspect state, and integrate into their own applications. This is the same pattern as kubectl (CLI) vs client-go (library)."

### Design Decisions

| Decision | Reasoning |
|----------|-----------|
| **CONFIRM as default mode** | Safest option, users can opt into more autonomy |
| **20 step limit default** | Prevents runaway costs |
| **$1.0 cost limit default** | Reasonable for testing |
| **--verbose for stack traces** | Clean output by default, debug when needed |
| **Exit code 130 for Ctrl+C** | Standard Unix convention |

### Production Considerations

```
┌─────────────────────────────────────────────────────────────────────┐
│                 PRODUCTION CLI PATTERNS                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. Always handle Ctrl+C gracefully                                 │
│     └── except KeyboardInterrupt: clean_up()                        │
│                                                                      │
│  2. Use proper exit codes                                           │
│     └── CI/CD depends on them                                       │
│                                                                      │
│  3. Provide --verbose for debugging                                 │
│     └── Don't show stack traces by default                          │
│                                                                      │
│  4. Show progress/stats at the end                                  │
│     └── Users want to know what happened                            │
│                                                                      │
│  5. Document with --help and examples                               │
│     └── epilog= in ArgumentParser                                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Red Hat Context

> "Enterprise tools need both interfaces. Ansible uses `ansible-playbook` CLI for operators and `ansible-runner` library for AWX/Tower. OpenShift has `oc` CLI and various Go libraries. The CLI is often a thin wrapper around the library - this ensures consistency and makes testing easier."

---

## Quick Reference

### Run CLI

```bash
# Basic
python -m code_agent.main --task "List files"

# All options
python -m code_agent.main \
    --task "Your task" \
    --mode confirm \
    --model claude-sonnet-4-20250514 \
    --step-limit 20 \
    --cost-limit 1.0 \
    --verbose
```

### Programmatic Usage

```python
from code_agent import InteractiveAgent, ExecutionMode, LocalEnvironment, LiteLLMModel

model = LiteLLMModel(model_name="claude-sonnet-4-20250514")
env = LocalEnvironment()
agent = InteractiveAgent(model, env, mode=ExecutionMode.CONFIRM)

status, message = agent.run("Your task")
print(f"Cost: ${model.get_stats()['total_cost']}")
```

### Get Help

```bash
python -m code_agent.main --help
```

---

## See Also

- `src/code_agent/main.py` - CLI implementation
- `examples/simple_task.py` - Example script
- `docs/INTERACTIVE_MODES_NOTES.md` - Execution modes
- `docs/STUDY_NOTES.md` - General agent concepts
