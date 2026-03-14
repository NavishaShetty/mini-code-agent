# Interactive Modes in AI Agents: Complete Guide

> Notes on human-in-the-loop execution patterns, trust boundaries, and safe command handling.

---

## Table of Contents

1. [The Problem: Autonomous Agents Need Guardrails](#1-the-problem-autonomous-agents-need-guardrails)
2. [The Three Execution Modes](#2-the-three-execution-modes)
3. [Safe vs Dangerous Commands](#3-safe-vs-dangerous-commands)
4. [How InteractiveAgent Works](#4-how-interactiveagent-works)
5. [Code Walkthrough](#5-code-walkthrough)
6. [Flow Diagrams](#6-flow-diagrams)
7. [Interview Preparation](#7-interview-preparation)

---

## 1. The Problem: Autonomous Agents Need Guardrails

### The Risk of Full Autonomy

A fully autonomous agent can:
- Delete important files (`rm -rf /`)
- Expose secrets (`cat ~/.ssh/id_rsa`)
- Make network calls (`curl malicious-site.com`)
- Install malware (`curl ... | bash`)
- Modify system files (`sudo ...`)

### Trust Boundaries

```
┌─────────────────────────────────────────────────────────────────┐
│                      TRUST SPECTRUM                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  FULLY TRUSTED          PARTIALLY TRUSTED       NOT TRUSTED     │
│  ─────────────          ─────────────────       ───────────     │
│                                                                  │
│  Read operations        Write operations        System commands │
│  • ls, cat, grep        • echo > file           • sudo          │
│  • find, tree           • mkdir, touch          • rm -rf        │
│  • git status           • git commit            • chmod 777     │
│                                                                  │
│  ✅ Auto-approve        🤔 Ask for approval     🚨 Warn + ask   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Real-World Patterns

| Product | Approach |
|---------|----------|
| **Claude Code** | Confirm mode by default, can switch to auto-accept |
| **GitHub Copilot** | Suggestions only, human always executes |
| **Cursor** | Shows diff, human approves before applying |
| **OpenCode** | Multiple modes like mini-swe-agent |

---

## 2. The Three Execution Modes

### Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     EXECUTION MODES                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  HUMAN MODE (👤)                                                     │
│  ─────────────────                                                  │
│  Agent: "I suggest running: ls -la"                                 │
│  You:   Type your own command → "ls -l src/"                        │
│  Agent: Executes YOUR command, observes result                      │
│                                                                      │
│  Use case: Learning, debugging, when you don't trust the agent      │
│                                                                      │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                      │
│  CONFIRM MODE (🤝) - DEFAULT                                        │
│  ─────────────────────────────                                      │
│  Agent: "I want to run: rm old_file.py"                             │
│  You:   [y]es / [n]o / [e]dit / [q]uit                              │
│                                                                      │
│  Special: Safe commands (ls, cat, grep) auto-approved               │
│                                                                      │
│  Use case: Production, when you want oversight but not tedium       │
│                                                                      │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                      │
│  YOLO MODE (🚀)                                                      │
│  ───────────────                                                    │
│  Agent: Executes everything without asking                          │
│  (Still warns on dangerous commands like rm -rf)                    │
│                                                                      │
│  Use case: CI/CD, automated pipelines, trusted environments         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### When to Use Each Mode

| Mode | When to Use | Risk Level |
|------|-------------|------------|
| **HUMAN** | Learning how agent thinks, debugging agent behavior, extremely sensitive operations | Lowest (you control everything) |
| **CONFIRM** | Normal development, code review, when mistakes are costly | Medium (you approve each action) |
| **YOLO** | CI/CD pipelines, batch processing, sandboxed environments, when speed matters | Highest (agent has full control) |

### Mode Selection Decision Tree

```
                    Is this a trusted environment?
                           /            \
                         NO              YES
                         /                \
              Is speed critical?      Is it CI/CD?
                /        \               /      \
              NO          YES          NO        YES
              /            \           /          \
         CONFIRM         HUMAN      CONFIRM      YOLO
         (review)       (control)   (safe)    (automated)
```

---

## 3. Safe vs Dangerous Commands

### Safe Command Patterns

These are **read-only** operations that can't cause harm:

```python
SAFE_PATTERNS = [
    r"^ls\b",           # List files
    r"^cat\b",          # Read file
    r"^head\b",         # Read file head
    r"^tail\b",         # Read file tail
    r"^grep\b",         # Search in files
    r"^find\b",         # Find files
    r"^pwd$",           # Print working directory
    r"^echo\b",         # Print text (mostly safe)
    r"^wc\b",           # Word count
    r"^tree\b",         # Directory tree
    r"^git status",     # Git status (read-only)
    r"^git log",        # Git log (read-only)
    r"^git diff",       # Git diff (read-only)
    r"^git branch",     # List branches (read-only)
]
```

### Why These Are Safe

| Command | Why Safe |
|---------|----------|
| `ls` | Only lists files, doesn't modify |
| `cat` | Only reads, doesn't write |
| `grep` | Only searches, doesn't modify |
| `git status` | Only shows state, doesn't change repo |
| `find` | Only locates files, doesn't touch them |

### Dangerous Command Patterns

These can cause **irreversible damage**:

```python
DANGEROUS_PATTERNS = [
    r"^rm\s+-rf",              # Recursive force delete
    r"^sudo\b",                # Superuser access
    r"curl.*\|\s*(sh|bash)",   # Download and execute (major red flag!)
]
```

### Why These Are Dangerous

| Pattern | Risk |
|---------|------|
| `rm -rf` | Can delete entire filesystem |
| `sudo` | Bypasses all permission checks |
| `curl \| bash` | Executes arbitrary code from internet |

### The Regex Explained

```python
r"^rm\s+-rf"
 │ │  │  │
 │ │  │  └── Matches "-rf" literally
 │ │  └───── \s+ matches one or more whitespace
 │ └──────── "rm" at the start
 └────────── ^ anchors to start of string

r"curl.*\|\s*(sh|bash)"
       │  │  │
       │  │  └── Matches "sh" or "bash"
       │  └───── Pipe character (escaped)
       └──────── .* matches anything between curl and pipe
```

### How Detection Works

```python
def is_safe_command(command: str) -> bool:
    """Check if command matches ANY safe pattern."""
    return any(re.match(p, command.strip()) for p in SAFE_PATTERNS)

def is_dangerous_command(command: str) -> bool:
    """Check if command matches ANY dangerous pattern."""
    return any(re.search(p, command.strip()) for p in DANGEROUS_PATTERNS)
```

**Key difference:**
- `re.match()` - Only matches at START of string
- `re.search()` - Matches ANYWHERE in string

Why? Safe commands must START with the pattern (`ls ...`), but dangerous patterns can appear anywhere (`something | sudo ...`).

---

## 4. How InteractiveAgent Works

### Class Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                     CLASS HIERARCHY                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Agent (base.py)                                                │
│    │                                                             │
│    ├── run()              - Main loop                           │
│    ├── step()             - One iteration                       │
│    ├── query()            - Call LLM                            │
│    ├── parse_action()     - Extract bash block                  │
│    ├── execute_action()   - Run command  ◄── OVERRIDDEN         │
│    └── get_observation()  - Format output                       │
│                                                                  │
│  InteractiveAgent (interactive.py) extends Agent                │
│    │                                                             │
│    ├── execute_action()   - Adds mode handling ◄── OVERRIDE     │
│    ├── _get_user_command() - HUMAN mode logic                   │
│    ├── _confirm_action()   - CONFIRM mode logic                 │
│    └── run()              - Adds mode announcement              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### The Key Insight: Only Override execute_action()

The `InteractiveAgent` only needs to intercept the moment before a command runs. Everything else (LLM calls, parsing, observation) stays the same.

```python
class InteractiveAgent(Agent):
    def execute_action(self, action: str) -> dict[str, Any]:
        # ┌─────────────────────────────────────────┐
        # │ INTERCEPT HERE - before actual execution │
        # └─────────────────────────────────────────┘

        if self.mode == ExecutionMode.HUMAN:
            action = self._get_user_command(action)   # Replace command
        elif self.mode == ExecutionMode.CONFIRM:
            action = self._confirm_action(action)     # Get approval
        elif self.mode == ExecutionMode.YOLO:
            if is_dangerous_command(action):
                print(f"⚠️  DANGEROUS: {action}")    # Just warn

        # Then call parent's execute_action
        return super().execute_action(action)
```

### Why This Design?

1. **Minimal code** - Only override what's needed
2. **Inheritance** - Reuse all base agent logic
3. **Single point of control** - All command execution goes through one method
4. **Easy to extend** - Add new modes by adding elif branches

---

## 5. Code Walkthrough

### The ExecutionMode Enum

```python
class ExecutionMode(str, Enum):
    HUMAN = "human"
    CONFIRM = "confirm"
    YOLO = "yolo"
```

Why inherit from both `str` and `Enum`?
- Can use as enum: `ExecutionMode.CONFIRM`
- Can use as string: `mode.value == "confirm"`
- Can construct from string: `ExecutionMode("confirm")`

### The InteractiveAgent Constructor

```python
def __init__(
    self,
    model: Model,
    env: Environment,
    *,
    mode: ExecutionMode = ExecutionMode.CONFIRM,  # Default is safest
    auto_approve_safe: bool = True,               # Auto-approve ls, cat, etc.
    **kwargs,
):
    super().__init__(model, env, **kwargs)        # Call parent constructor
    self.mode = mode
    self.auto_approve_safe = auto_approve_safe
```

### HUMAN Mode: _get_user_command()

```python
def _get_user_command(self, suggested: str) -> str:
    """In HUMAN mode, user provides the actual command."""

    # Show what agent suggested
    print(f"\n🤖 Suggests: {suggested}")
    print("Enter command ('quit' to abort, 'skip' to continue):")

    try:
        user_input = input(">>> ").strip()

        if user_input.lower() == "quit":
            raise UserAbort("User aborted")      # Terminates agent

        if user_input.lower() == "skip":
            return "echo '[skipped]'"            # No-op command

        return user_input or suggested           # User's command or default

    except (KeyboardInterrupt, EOFError):
        raise UserAbort("User interrupted")
```

### CONFIRM Mode: _confirm_action()

```python
def _confirm_action(self, action: str) -> str:
    """In CONFIRM mode, user approves/rejects agent's command."""

    # Auto-approve safe commands (if enabled)
    if self.auto_approve_safe and is_safe_command(action):
        print(f"\n✅ Auto-approved: {action}")
        return action                            # No user input needed

    # Warn on dangerous commands
    if is_dangerous_command(action):
        print(f"\n🚨 DANGEROUS!")

    # Ask for approval
    print(f"\n🤖 Run: {action}")
    print("[y]es / [n]o / [e]dit / [q]uit")

    try:
        choice = input(">>> ").strip().lower()

        if choice in ("y", "yes", ""):           # Yes or Enter
            return action

        if choice in ("n", "no"):                # Reject
            raise FormatError("Rejected. Try different approach.")
            # FormatError is NonTerminating - agent will retry

        if choice in ("e", "edit"):              # Edit command
            return input("New command: ").strip() or action

        if choice in ("q", "quit"):              # Quit
            raise UserAbort("User aborted")
            # UserAbort is Terminating - agent stops

        return action                            # Default: proceed

    except (KeyboardInterrupt, EOFError):
        raise UserAbort("User interrupted")
```

### The Exception Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                     EXCEPTION HANDLING                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  User rejects command (presses 'n')                             │
│       │                                                          │
│       └──► raise FormatError("Rejected...")                     │
│                 │                                                │
│                 └──► FormatError is NonTerminatingException     │
│                           │                                      │
│                           └──► Agent catches, adds to messages,  │
│                                and tries again                   │
│                                                                  │
│  ─────────────────────────────────────────────────────────────  │
│                                                                  │
│  User quits (presses 'q' or Ctrl+C)                             │
│       │                                                          │
│       └──► raise UserAbort("User aborted")                      │
│                 │                                                │
│                 └──► UserAbort is TerminatingException          │
│                           │                                      │
│                           └──► Agent catches, returns result,    │
│                                and stops                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Flow Diagrams

### Complete Flow: CONFIRM Mode

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CONFIRM MODE COMPLETE FLOW                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  LLM returns: "I'll list the files.\n```bash\nls -la\n```"          │
│       │                                                              │
│       ▼                                                              │
│  parse_action() extracts: "ls -la"                                  │
│       │                                                              │
│       ▼                                                              │
│  execute_action("ls -la") called                                    │
│       │                                                              │
│       ├── Mode is CONFIRM                                           │
│       │                                                              │
│       ▼                                                              │
│  _confirm_action("ls -la")                                          │
│       │                                                              │
│       ├── is_safe_command("ls -la") → True                         │
│       │       │                                                      │
│       │       └── ✅ Auto-approved: ls -la                          │
│       │           return "ls -la" (unchanged)                       │
│       │                                                              │
│       ▼                                                              │
│  super().execute_action("ls -la")                                   │
│       │                                                              │
│       └── Actually runs the command via subprocess                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Complete Flow: CONFIRM Mode (Dangerous Command)

```
┌─────────────────────────────────────────────────────────────────────┐
│                 CONFIRM MODE - DANGEROUS COMMAND                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  LLM returns: "```bash\nrm -rf old_folder/\n```"                    │
│       │                                                              │
│       ▼                                                              │
│  parse_action() extracts: "rm -rf old_folder/"                      │
│       │                                                              │
│       ▼                                                              │
│  execute_action("rm -rf old_folder/")                               │
│       │                                                              │
│       ├── Mode is CONFIRM                                           │
│       │                                                              │
│       ▼                                                              │
│  _confirm_action("rm -rf old_folder/")                              │
│       │                                                              │
│       ├── is_safe_command(...) → False                              │
│       ├── is_dangerous_command(...) → True                          │
│       │       │                                                      │
│       │       └── 🚨 DANGEROUS!                                     │
│       │           🤖 Run: rm -rf old_folder/                        │
│       │           [y]es / [n]o / [e]dit / [q]uit                   │
│       │                                                              │
│       ▼                                                              │
│  User types: "n"                                                    │
│       │                                                              │
│       └── raise FormatError("Rejected...")                          │
│                 │                                                    │
│                 └── Agent retries with different approach           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Mode Comparison Side-by-Side

```
Command: "rm old_file.py"

┌───────────────────┬───────────────────┬───────────────────┐
│    HUMAN MODE     │   CONFIRM MODE    │    YOLO MODE      │
├───────────────────┼───────────────────┼───────────────────┤
│                   │                   │                   │
│ 🤖 Suggests:      │ 🤖 Run:           │ (no prompt)       │
│ rm old_file.py    │ rm old_file.py    │                   │
│                   │ [y/n/e/q]         │                   │
│ >>> _             │ >>> _             │                   │
│                   │                   │                   │
│ You type your     │ You type y/n/e/q  │ Executes          │
│ own command       │                   │ immediately       │
│                   │                   │                   │
└───────────────────┴───────────────────┴───────────────────┘
```

---

## 7. Interview Preparation

### Key Concepts

| Concept | Explanation |
|---------|-------------|
| **Human-in-the-loop** | Human oversight at critical decision points |
| **Trust boundaries** | Defining what actions need approval |
| **Safe commands** | Read-only operations that can't cause harm |
| **Graceful degradation** | Warn on dangerous even in YOLO mode |

### Why Three Modes?

> "Different situations require different levels of autonomy. HUMAN mode is for learning and debugging - the human drives while the agent suggests. CONFIRM mode is the production default - the agent proposes, human approves. YOLO mode is for CI/CD pipelines where human oversight isn't practical. The key insight is that even in YOLO mode, we still warn on dangerous commands."

### Why Override Only execute_action()?

> "The InteractiveAgent follows the Open/Closed principle - open for extension, closed for modification. We only override the single method where human intervention makes sense. The LLM calls, parsing, and observation logic stay unchanged. This minimizes code and potential bugs."

### Production Considerations

| Consideration | How We Handle It |
|---------------|------------------|
| **Default to safe** | CONFIRM mode is default |
| **Auto-approve safe** | Read-only commands don't need approval |
| **Warn on dangerous** | Even YOLO mode warns on `rm -rf`, `sudo` |
| **User can always quit** | Ctrl+C or 'quit' always works |
| **Rejection = retry** | FormatError lets agent try different approach |

### Red Hat Context

> "In enterprise environments like Red Hat's Lightspeed products, trust boundaries are critical. Ansible Lightspeed operates in suggestion mode (like our HUMAN mode) - it suggests playbook code but humans always execute. OpenShift Lightspeed uses read-only cluster commands by default. The pattern is: start restrictive, allow users to opt into more autonomy."

---

## Quick Reference

### Mode Selection

```python
# Safest - user controls everything
agent = InteractiveAgent(model, env, mode=ExecutionMode.HUMAN)

# Balanced - agent proposes, user approves (DEFAULT)
agent = InteractiveAgent(model, env, mode=ExecutionMode.CONFIRM)

# Fastest - full autonomy
agent = InteractiveAgent(model, env, mode=ExecutionMode.YOLO)
```

### Disable Auto-Approve

```python
# Require approval for ALL commands, even safe ones
agent = InteractiveAgent(
    model, env,
    mode=ExecutionMode.CONFIRM,
    auto_approve_safe=False,  # Will ask for ls, cat, etc.
)
```

### User Options in CONFIRM Mode

| Key | Action |
|-----|--------|
| `y` or Enter | Approve and execute |
| `n` | Reject (agent will retry) |
| `e` | Edit the command |
| `q` | Quit the session |
| Ctrl+C | Interrupt immediately |

---

## See Also

- `src/code_agent/agent/interactive.py` - Implementation
- `src/code_agent/agent/base.py` - Base Agent class
- `docs/STUDY_NOTES.md` - General agent concepts
