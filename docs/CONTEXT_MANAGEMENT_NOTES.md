# Context Management in AI Agents: Complete Guide

> Notes on managing context windows, truncation strategies, and token budgeting.

---

## Table of Contents

1. [The Problem: Limited Context Windows](#1-the-problem-limited-context-windows)
2. [Where Context Management Fits in the Agent Loop](#2-where-context-management-fits-in-the-agent-loop)
3. [All Context Management Techniques](#3-all-context-management-techniques)
4. [Output Truncation Strategies](#4-output-truncation-strategies)
5. [Message History Strategies](#5-message-history-strategies)
6. [Token Budgeting](#6-token-budgeting)
7. [Context Injection (Ansible Lightspeed Pattern)](#7-context-injection-ansible-lightspeed-pattern)
8. [Our Implementation](#8-our-implementation)
9. [Interview Preparation](#9-interview-preparation)
10. [Wiring Context Management into the Agent](#10-wiring-context-management-into-the-agent)

---

## 1. The Problem: Limited Context Windows

### Every LLM Has a Token Limit

| Model | Context Window | Rough Character Limit |
|-------|---------------|----------------------|
| GPT-3.5 | 16,000 tokens | ~64,000 chars |
| GPT-4 | 128,000 tokens | ~512,000 chars |
| GPT-4o | 128,000 tokens | ~512,000 chars |
| Claude 3 Sonnet | 200,000 tokens | ~800,000 chars |
| Claude 3 Opus | 200,000 tokens | ~800,000 chars |
| Llama 3 (8B) | 8,000 tokens | ~32,000 chars |
| Llama 3 (70B) | 8,000 tokens | ~32,000 chars |
| Mistral 7B | 32,000 tokens | ~128,000 chars |

### What Happens When You Exceed It?

```
┌─────────────────────────────────────────────────────────────────┐
│  CONTEXT WINDOW EXCEEDED                                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Option 1: API Error                                            │
│  → "Context length exceeded. Maximum is 128000 tokens."         │
│                                                                  │
│  Option 2: Silent Truncation                                    │
│  → API silently drops old messages (dangerous!)                 │
│                                                                  │
│  Option 3: Increased Cost                                       │
│  → Some APIs charge more for long contexts                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### The Agent Problem

As an agent runs, context grows:

```
Iteration 1:   ~500 tokens    [system, task]
Iteration 5:   ~3,000 tokens  [system, task, 8 exchanges]
Iteration 20:  ~15,000 tokens [system, task, 38 exchanges]
Iteration 50:  ~50,000 tokens [system, task, 98 exchanges]
Iteration 100: ~100,000 tokens ⚠️ APPROACHING LIMIT
```

---

## 2. Where Context Management Fits in the Agent Loop

### The ReAct Loop Refresher

```
THINK → ACT → OBSERVE → THINK → ACT → OBSERVE → ...
```

The key insight: **The observation becomes part of the context for the next iteration.**

### Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    ONE ITERATION OF THE LOOP                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  STEP 1: THINK (query the LLM)                                  │
│  ─────────────────────────────                                  │
│                                                                  │
│  Messages sent to LLM:                                          │
│  [system, task, prev_assistant, prev_observation, ...]         │
│       │                                                          │
│       │  ◄── budget_context() called HERE                       │
│       │      (ensure messages fit in context window)            │
│       ▼                                                          │
│  LLM returns: "I'll list the files.\n```bash\nls -la\n```"     │
│       │                                                          │
│       │  Response added to messages as "assistant" role         │
│       ▼                                                          │
│                                                                  │
│  STEP 2: ACT (execute the command)                              │
│  ─────────────────────────────────                              │
│                                                                  │
│  Agent parses: "ls -la"                                         │
│  Agent runs: subprocess.run("ls -la")                           │
│       │                                                          │
│       ▼                                                          │
│  Raw output: "total 128\ndrwxr-xr-x  5 user...\n..."           │
│       │                                                          │
│       │  ⚠️ THIS COULD BE HUGE! (100,000+ lines)               │
│       ▼                                                          │
│                                                                  │
│  STEP 3: OBSERVE (add output to messages)                       │
│  ────────────────────────────────────────                       │
│                                                                  │
│       │  ◄── truncate_output() called HERE                      │
│       │      (before adding to messages!)                       │
│       ▼                                                          │
│  Truncated output added to messages as "user" role              │
│                                                                  │
│       │                                                          │
│       │  Messages now include the observation                   │
│       ▼                                                          │
│                                                                  │
│  STEP 4: LOOP BACK TO THINK                                     │
│  ──────────────────────────                                     │
│                                                                  │
│  All messages (including observation) sent to LLM               │
│       │                                                          │
│       │  If observation wasn't truncated, context explodes!     │
│       ▼                                                          │
│                                                                  │
│  THINK again with full context...                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Why Truncate at OBSERVE Step?

The observation (command output) **loops back** to become part of the context:

```
Iteration 1:
  messages = [system, task]

  THINK → LLM says "run ls"
  messages = [system, task, assistant("run ls")]

  ACT → command returns 50,000 lines

  OBSERVE → output added to messages
  messages = [system, task, assistant("run ls"), user(OUTPUT)]
                                                      ↑
                                            THIS GOES TO LLM NEXT!

Iteration 2:
  THINK → LLM receives ALL of messages including OUTPUT

  If OUTPUT is 50,000 lines = 💥 CONTEXT OVERFLOW
```

### The Problem Without Truncation

```python
# After command runs (NO truncation)
command_output = execute("find / -name '*.py'")  # Returns 50,000 files

messages = [
    {"role": "system", "content": "..."},           # 500 tokens
    {"role": "user", "content": "Find Python files"}, # 50 tokens
    {"role": "assistant", "content": "```bash\nfind / -name '*.py'\n```"}, # 30 tokens
    {"role": "user", "content": f"Output:\n{command_output}"},  # 200,000 tokens!
]
# Total: 200,580 tokens ❌ EXCEEDS MOST LIMITS!
```

### With Truncation

```python
# After command runs (WITH truncation)
command_output = execute("find / -name '*.py'")  # Returns 50,000 files
truncated = truncate_output(command_output, max_chars=10000)  # ~2,500 tokens

messages = [
    {"role": "system", "content": "..."},           # 500 tokens
    {"role": "user", "content": "Find Python files"}, # 50 tokens
    {"role": "assistant", "content": "```bash\nfind / -name '*.py'\n```"}, # 30 tokens
    {"role": "user", "content": f"Output:\n{truncated}"},  # 2,500 tokens
]
# Total: 3,080 tokens ✅ FITS!
```

### Where Each Function Is Called

| Function | Where Called | Why |
|----------|--------------|-----|
| `get_project_context()` | `run()` at start | Inject project info into system prompt |
| `budget_context()` | `query()` before LLM call | Ensure messages fit in context window |
| `truncate_output()` | `get_observation()` after command | Prevent huge output from bloating context |
| `truncate_messages()` | `query()` or periodically | Keep message count manageable |

### Code: Where Functions Should Be Called

```python
class Agent:
    def run(self, task: str) -> tuple[str, str]:
        # ┌─────────────────────────────────────────┐
        # │ INJECT PROJECT CONTEXT AT START         │
        # └─────────────────────────────────────────┘
        project_ctx = get_project_context()
        system_prompt = f"{base_system_prompt}\n\n{format_context_for_prompt(project_ctx)}"

        self.add_message("system", system_prompt)
        self.add_message("user", task)

        while True:
            self.step()

    def query(self) -> dict:
        # ┌─────────────────────────────────────────┐
        # │ BUDGET CONTEXT BEFORE SENDING TO LLM    │
        # └─────────────────────────────────────────┘
        messages_to_send = budget_context(
            self.messages,
            max_tokens=100000,
            reserve_for_response=4000
        )

        response = self.model.query(messages_to_send)
        self.add_message("assistant", **response)
        return response

    def get_observation(self, response: dict) -> dict:
        action = self.parse_action(response)
        output = self.execute_action(action)

        # ┌─────────────────────────────────────────┐
        # │ TRUNCATE OUTPUT BEFORE ADDING TO MSGS   │
        # └─────────────────────────────────────────┘
        truncated = truncate_output(output["output"], max_chars=10000)

        observation = f"Output:\n{truncated}"
        self.add_message("user", observation)  # This loops back to LLM!
        return output
```

### Why "user" Role for Observation?

```
From the LLM's perspective:

"user" role    = Information coming FROM the outside world
"assistant" role = What I (the LLM) said

So:
├── Task description    → "user"     (human gave it)
├── LLM's response      → "assistant" (I generated it)
└── Command output      → "user"     (world responded to my action)

The LLM sees observations as "the world giving me feedback on my action."
```

### The Complete Picture

```
┌────────────────────────────────────────────────────────────────────┐
│                     AGENT WITH CONTEXT MANAGEMENT                   │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  START                                                             │
│    │                                                                │
│    ├── get_project_context()  ──► Inject project info              │
│    │                                                                │
│    ▼                                                                │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                        LOOP                                   │  │
│  │                                                               │  │
│  │  THINK ─────────────────────────────────────────────────────│  │
│  │    │                                                         │  │
│  │    ├── budget_context()  ──► Fit messages in window         │  │
│  │    ├── Send to LLM                                           │  │
│  │    └── Add response as "assistant"                           │  │
│  │                                                               │  │
│  │  ACT ───────────────────────────────────────────────────────│  │
│  │    │                                                         │  │
│  │    ├── Parse bash command                                    │  │
│  │    └── Execute via subprocess                                │  │
│  │                                                               │  │
│  │  OBSERVE ───────────────────────────────────────────────────│  │
│  │    │                                                         │  │
│  │    ├── truncate_output()  ──► Prevent context explosion     │  │
│  │    └── Add output as "user" (loops back to THINK!)          │  │
│  │                                                               │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  END (when task complete or limits exceeded)                       │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

### Interview Talking Point

> "In a ReAct agent, the observation from each step becomes part of the context for the next iteration. This means command outputs loop back to the LLM. Without truncation, a single large output can blow up the context window. We truncate outputs immediately after execution, before adding them to the message history, using a head+tail pattern that preserves the most useful information - headers at the start and results/errors at the end."

---

## 3. All Context Management Techniques

### Complete List of Strategies

```
┌─────────────────────────────────────────────────────────────────┐
│              CONTEXT MANAGEMENT TECHNIQUES                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  OUTPUT TRUNCATION (single outputs)                             │
│  ──────────────────────────────────                             │
│  1. Head-only         - Keep first N chars                      │
│  2. Tail-only         - Keep last N chars                       │
│  3. Head + Tail       - Keep beginning and end                  │
│  4. Sampling          - Keep every Nth line                     │
│  5. Smart truncation  - Keep based on content type              │
│                                                                  │
│  MESSAGE HISTORY (conversation management)                      │
│  ─────────────────────────────────────────                      │
│  6. Sliding window    - Keep N most recent messages             │
│  7. First + Last      - Keep system + recent messages           │
│  8. Summarization     - Compress old messages into summary      │
│  9. Hierarchical      - Summarize in chunks                     │
│  10. Selective        - Keep important messages only            │
│                                                                  │
│  TOKEN BUDGETING (allocation)                                   │
│  ────────────────────────────                                   │
│  11. Fixed allocation  - Fixed tokens per section               │
│  12. Priority-based    - Allocate by importance                 │
│  13. Dynamic           - Adjust based on task                   │
│                                                                  │
│  CONTEXT INJECTION (adding context)                             │
│  ──────────────────────────────────                             │
│  14. Project context   - Files, git status, README              │
│  15. RAG retrieval     - Relevant documents                     │
│  16. Code context      - Open files, cursor position            │
│                                                                  │
│  ADVANCED TECHNIQUES                                            │
│  ───────────────────                                            │
│  17. Context caching   - Cache repeated prompts                 │
│  18. Prompt compression - Compress prompts                      │
│  19. Multi-turn summarization - Periodic summaries              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Output Truncation Strategies

### Strategy 1: Head-Only

Keep only the beginning of the output.

```python
def truncate_head(output: str, max_chars: int = 5000) -> str:
    if len(output) <= max_chars:
        return output
    return output[:max_chars] + "\n...[truncated]"
```

**When to use**: When the beginning has the most important info (headers, configs).

**Example**:
```
Original:  [100,000 chars of log file]
Truncated: [First 5,000 chars]...[truncated]
```

---

### Strategy 2: Tail-Only

Keep only the end of the output.

```python
def truncate_tail(output: str, max_chars: int = 5000) -> str:
    if len(output) <= max_chars:
        return output
    return "[truncated]...\n" + output[-max_chars:]
```

**When to use**: When the end has results/errors (build logs, test output).

**Example**:
```
Original:  [100,000 chars of npm install]
Truncated: [truncated]...[Last 5,000 chars with final status]
```

---

### Strategy 3: Head + Tail (Our Implementation)

Keep both beginning and end, remove middle.

```python
def truncate_head_tail(output: str, max_chars: int = 10000) -> str:
    if len(output) <= max_chars:
        return output

    head_size = max_chars // 3      # 1/3 for head
    tail_size = max_chars * 2 // 3  # 2/3 for tail

    head = output[:head_size]
    tail = output[-tail_size:]

    return f"{head}\n\n[...truncated...]\n\n{tail}"
```

**When to use**: General purpose - works for most command outputs.

**Visual**:
```
┌──────────────────────────────────────────────────────────────┐
│  ORIGINAL OUTPUT                                              │
├──────────────────────────────────────────────────────────────┤
│  [HEADER INFO]           ← Keep (useful)                     │
│  [Column names]          ← Keep (useful)                     │
│  [Data row 1]                                                │
│  [Data row 2]                                                │
│  [Data row 3]            ← Truncate (repetitive)             │
│  ...                                                          │
│  [Data row 9997]         ← Truncate (repetitive)             │
│  [Data row 9998]                                             │
│  [Data row 9999]         ← Keep (recent)                     │
│  [SUMMARY]               ← Keep (results)                    │
│  [ERRORS if any]         ← Keep (important)                  │
└──────────────────────────────────────────────────────────────┘
```

---

### Strategy 4: Sampling

Keep every Nth line.

```python
def truncate_sampling(output: str, keep_every: int = 10) -> str:
    lines = output.splitlines()
    if len(lines) <= 100:
        return output

    sampled = [lines[i] for i in range(0, len(lines), keep_every)]
    return f"[Sampled every {keep_every} lines]\n" + "\n".join(sampled)
```

**When to use**: When all data is equally important (logs, data dumps).

---

### Strategy 5: Smart Truncation

Truncate based on content type.

```python
def smart_truncate(output: str, content_type: str) -> str:
    if content_type == "error":
        return truncate_tail(output)  # Errors at end
    elif content_type == "json":
        return truncate_json(output)  # Parse and summarize
    elif content_type == "table":
        return truncate_head_tail(output)  # Headers + recent
    else:
        return truncate_head_tail(output)  # Default
```

**When to use**: When you know the output type.

---

## 5. Message History Strategies

### Strategy 6: Sliding Window (Our Implementation)

Keep N most recent messages.

```python
def sliding_window(messages: list, max_messages: int = 20) -> list:
    if len(messages) <= max_messages:
        return messages
    return messages[-max_messages:]
```

**Visual**:
```
BEFORE: [1] [2] [3] [4] [5] [6] [7] [8] [9] [10] [11] [12]

AFTER (max=5): [8] [9] [10] [11] [12]
               ↑
               Only keep 5 most recent
```

**Pros**: Simple, keeps recent context
**Cons**: Loses early context (task description, initial plan)

---

### Strategy 7: First + Last (Our Implementation)

Keep system message + recent messages.

```python
def first_plus_last(messages: list, max_messages: int = 20) -> list:
    if len(messages) <= max_messages:
        return messages

    system = [messages[0]]  # Always keep system/task
    recent = messages[-(max_messages - 1):]
    return system + recent
```

**Visual**:
```
BEFORE: [SYS] [1] [2] [3] [4] [5] [6] [7] [8] [9] [10]

AFTER (max=5): [SYS] [7] [8] [9] [10]
                ↑              ↑
                System kept    4 most recent
```

**Pros**: Preserves task context, keeps recent
**Cons**: Loses middle context (early attempts, learnings)

---

### Strategy 8: Summarization

Compress old messages into a summary.

```python
def summarize_old_messages(messages: list, max_messages: int = 20) -> list:
    if len(messages) <= max_messages:
        return messages

    # Keep system message
    system = messages[0]

    # Messages to summarize (old ones)
    to_summarize = messages[1:-10]

    # Ask LLM to summarize
    summary = llm.summarize(to_summarize)
    summary_msg = {"role": "user", "content": f"[Previous context summary]: {summary}"}

    # Recent messages to keep
    recent = messages[-10:]

    return [system, summary_msg] + recent
```

**Visual**:
```
BEFORE: [SYS] [1] [2] [3] [4] [5] [6] [7] [8] [9] [10] [11] [12]

AFTER:  [SYS] [SUMMARY of 1-7] [8] [9] [10] [11] [12]
                    ↑
                    LLM-generated summary
```

**Pros**: Preserves information in compressed form
**Cons**: Requires extra LLM call, summary may lose details

---

### Strategy 9: Hierarchical Summarization

Summarize in chunks as context grows.

```python
def hierarchical_summarize(messages: list, chunk_size: int = 10) -> list:
    """
    When we hit chunk_size messages, summarize them.
    Summaries get summarized too when they accumulate.
    """
    # Level 0: Raw messages
    # Level 1: Summaries of 10 messages each
    # Level 2: Summaries of 10 summaries each
    # ...
```

**Visual**:
```
Level 0: [1][2][3][4][5][6][7][8][9][10] [11][12]...
              ↓ summarize
Level 1: [Summary 1-10] [11][12][13]...
              ↓ when 10 summaries accumulate
Level 2: [Summary of summaries] [recent messages]
```

**Pros**: Scales to very long conversations
**Cons**: Complex, progressive information loss

---

### Strategy 10: Selective Retention

Keep only important messages (e.g., errors, decisions).

```python
def selective_retention(messages: list, max_messages: int = 20) -> list:
    important = []
    for msg in messages:
        content = msg.get("content", "")
        if is_important(content):  # Check for errors, decisions, etc.
            important.append(msg)

    # Always keep system + most recent
    system = [messages[0]]
    recent = messages[-5:]

    return system + important + recent
```

**When to use**: When some messages are clearly more important.

---

## 6. Token Budgeting

### Strategy 11: Fixed Allocation

Pre-allocate tokens to each section.

```python
TOKEN_BUDGET = {
    "system_prompt": 1000,
    "project_context": 2000,
    "rag_context": 3000,
    "message_history": 4000,
    "reserve_for_response": 4000,
}
# Total: 14,000 tokens
```

**Visual**:
```
┌─────────────────────────────────────────┐
│  CONTEXT WINDOW: 16,000 tokens          │
├─────────────────────────────────────────┤
│  System Prompt     │ 1,000  │ ██        │
│  Project Context   │ 2,000  │ ████      │
│  RAG Context       │ 3,000  │ ██████    │
│  Message History   │ 4,000  │ ████████  │
│  Reserved Response │ 4,000  │ ████████  │
│  Buffer            │ 2,000  │ ████      │
└─────────────────────────────────────────┘
```

---

### Strategy 12: Priority-Based Allocation

Allocate based on importance.

```python
def priority_budget(available_tokens: int) -> dict:
    priorities = [
        ("system_prompt", 1.0, 500, 2000),   # (name, priority, min, max)
        ("task", 0.9, 200, 1000),
        ("recent_messages", 0.8, 1000, 5000),
        ("rag_context", 0.6, 500, 3000),
        ("old_messages", 0.3, 0, 2000),
    ]
    # Higher priority items get their max first
    # Lower priority items get what's left
```

---

### Strategy 13: Dynamic Allocation

Adjust based on the task.

```python
def dynamic_budget(task_type: str, available: int) -> dict:
    if task_type == "code_review":
        # More tokens for code context
        return {"code": 0.6, "history": 0.2, "rag": 0.2}
    elif task_type == "question_answering":
        # More tokens for RAG
        return {"code": 0.2, "history": 0.2, "rag": 0.6}
    else:
        # Balanced
        return {"code": 0.33, "history": 0.33, "rag": 0.33}
```

---

## 7. Context Injection (Ansible Lightspeed Pattern)

### Strategy 14: Project Context

Gather information about the project.

```python
def get_project_context(cwd: str) -> dict:
    return {
        "project_name": Path(cwd).name,
        "files": list_files(cwd),
        "git_status": get_git_status(cwd),
        "readme": read_readme(cwd),
        "dependencies": read_package_json_or_requirements(cwd),
    }
```

---

### Strategy 15: RAG Retrieval

Retrieve relevant documents.

```python
def get_rag_context(query: str, index: RAGIndex, k: int = 5) -> str:
    results = index.search(query, k=k)
    context = "\n\n---\n\n".join([r["content"] for r in results])
    return f"Relevant documentation:\n{context}"
```

---

### Strategy 16: Code Context (IDE Integration)

Capture context from the IDE.

```python
def get_code_context(ide_state: dict) -> dict:
    return {
        "current_file": ide_state["active_file"],
        "cursor_position": ide_state["cursor"],
        "selected_text": ide_state["selection"],
        "open_files": ide_state["open_tabs"],
        "recent_edits": ide_state["recent_changes"],
    }
```

**This is exactly what Ansible Lightspeed does!**

```
┌─────────────────────────────────────────────────────────────────┐
│                 ANSIBLE LIGHTSPEED FLOW                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  VS Code Extension                                              │
│       │                                                          │
│       ├── Captures: current file, cursor, open files           │
│       │                                                          │
│       ▼                                                          │
│  Lightspeed Service                                             │
│       │                                                          │
│       ├── Adds: RAG context (Ansible docs)                      │
│       ├── Adds: Project context                                 │
│       │                                                          │
│       ▼                                                          │
│  LLM generates completion                                       │
│       │                                                          │
│       ▼                                                          │
│  Returns to VS Code                                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Our Implementation

### What We Built

```
src/code_agent/context/manager.py
│
├── truncate_output()      - Head + tail truncation
├── truncate_messages()    - Sliding window with system preserved
├── get_project_context()  - Gather project info
├── format_context_for_prompt() - Format for injection
├── estimate_tokens()      - Quick token estimation
└── budget_context()       - Priority-based token budgeting
```

### Usage Example

```python
from code_agent.context.manager import (
    truncate_output,
    truncate_messages,
    get_project_context,
    budget_context,
)

# 1. Truncate large command output
huge_output = run_command("find / -name '*.py'")
truncated = truncate_output(huge_output, max_chars=10000)

# 2. Get project context
context = get_project_context("/path/to/project")
context_str = format_context_for_prompt(context)

# 3. Budget messages to fit token limit
messages = [...long conversation...]
budgeted = budget_context(messages, max_tokens=100000)

# 4. Simple message truncation
truncated_msgs = truncate_messages(messages, max_messages=20)
```

---

## 9. Interview Preparation

### Key Concepts to Know

| Concept | Explanation |
|---------|-------------|
| **Context window** | Maximum tokens an LLM can process at once |
| **Token** | ~4 characters on average |
| **Head + tail** | Keep beginning and end, remove middle |
| **Sliding window** | Keep N most recent messages |
| **Summarization** | Compress old messages into summary |
| **Token budgeting** | Allocate tokens to different sections |
| **Context injection** | Add project/code context to prompts |

### Trade-offs to Discuss

| Strategy | Pros | Cons |
|----------|------|------|
| **Head + tail** | Keeps important parts | Loses middle context |
| **Sliding window** | Simple, keeps recent | Loses early context |
| **Summarization** | Preserves info | Extra LLM call, lossy |
| **Fixed budget** | Predictable | May waste tokens |
| **Dynamic budget** | Efficient | Complex to implement |

### Red Hat Specific

| Product | Context Management |
|---------|-------------------|
| **Ansible Lightspeed** | IDE context injection (cursor, open files) |
| **OpenShift Lightspeed** | Cluster context + BYOK docs (RAG) |
| **InstructLab** | Training data context for fine-tuning |

### Interview Talking Points

> "Context management is critical for production agents. We use multiple strategies: head+tail truncation for command outputs (preserving headers and results), sliding window for message history (keeping system prompt and recent context), and token budgeting to allocate context space efficiently. This mirrors Ansible Lightspeed's approach of capturing IDE context and enriching with RAG retrieval."

---

## Quick Reference

### Token Estimation

```python
# Rough estimate (fast)
tokens = len(text) // 4

# Accurate (slower, requires tiktoken)
import tiktoken
encoder = tiktoken.encoding_for_model("gpt-4")
tokens = len(encoder.encode(text))
```

### Common Context Limits

```python
CONTEXT_LIMITS = {
    "gpt-3.5-turbo": 16_000,
    "gpt-4": 128_000,
    "gpt-4o": 128_000,
    "claude-3-sonnet": 200_000,
    "claude-3-opus": 200_000,
    "llama-3-8b": 8_000,
    "llama-3-70b": 8_000,
}
```

### Typical Token Budget

```python
# For a 100k context window
BUDGET = {
    "system_prompt": 1_000,
    "project_context": 2_000,
    "rag_context": 5_000,
    "message_history": 80_000,
    "reserve_for_response": 8_000,
    "buffer": 4_000,
}
```

---

## 10. Wiring Context Management into the Agent

### What We Integrated

After building the context management utilities in `manager.py`, we wired them directly into `agent/base.py`:

```
┌──────────────────────────────────────────────────────────────────────┐
│                    CONTEXT MANAGEMENT INTEGRATION                     │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  run()                                                               │
│    │                                                                 │
│    ├──► get_project_context() ──► format_context_for_prompt()       │
│    │    Gathers: cwd, files, git status, README                     │
│    │    Injects into system prompt via Jinja2 template              │
│    │                                                                 │
│    └──► step() loop                                                  │
│          │                                                           │
│          ├──► query()                                                │
│          │      │                                                    │
│          │      └──► budget_context(messages, max_tokens=100k)      │
│          │           Keeps system + most recent messages            │
│          │           Drops old messages if over budget              │
│          │                                                           │
│          └──► get_observation()                                      │
│                 │                                                    │
│                 └──► truncate_output(output, max_chars=10k)         │
│                      Head + tail pattern for large outputs          │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### New Config Options Added to AgentConfig

```python
class AgentConfig(BaseModel):
    # ... existing config ...

    # Context management settings
    max_output_chars: int = 10000      # Truncate command output beyond this
    max_context_tokens: int = 100000   # Budget messages to fit this limit
    inject_project_context: bool = True # Add project info to system prompt
```

### Integration Point 1: Project Context Injection in `run()`

```python
def run(self, task: str, **kwargs) -> tuple[str, str]:
    # ... setup ...

    # Optionally inject project context (like Ansible Lightspeed pattern)
    if self.config.inject_project_context:
        project_ctx = get_project_context()
        self.extra_template_vars["project_context"] = format_context_for_prompt(project_ctx)

    # System prompt now includes {{ project_context }} via Jinja2
    self.add_message("system", self.render_template(self.config.system_template))
```

**Result in system prompt:**
```
You are a helpful coding assistant...

Current working directory: /Users/navisha/Desktop/PROJECTS/code-agent
Operating system: Darwin 25.2.0

Project Context:
Project: code-agent
Directory: /Users/navisha/Desktop/PROJECTS/code-agent
Files: src/, docs/, examples/, pyproject.toml, CLAUDE.md, README.md...
Git status: clean
README preview: # Code Agent + RAG Chatbot...
```

### Integration Point 2: Message Budgeting in `query()`

```python
def query(self) -> dict[str, Any]:
    # Check limits...

    # Budget messages to fit within context window
    budgeted_messages = budget_context(
        self.messages,
        max_tokens=self.config.max_context_tokens,
    )

    response = self.model.query(budgeted_messages)  # Send budgeted, not full
    self.add_message("assistant", **response)
    return response
```

**What happens:**
```
BEFORE (50 messages, ~150k tokens):
  [system] [task] [asst1] [obs1] [asst2] [obs2] ... [asst50] [obs50]

AFTER budget_context() (fits in 100k):
  [system] [asst45] [obs45] [asst46] [obs46] ... [asst50] [obs50]
     ↑
     System always kept, old messages dropped
```

### Integration Point 3: Output Truncation in `get_observation()`

```python
def get_observation(self, response: dict[str, Any]) -> dict[str, Any]:
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
```

**What happens:**
```
Command: find / -name "*.py"

BEFORE (50,000 lines):
  /usr/lib/python3/file1.py
  /usr/lib/python3/file2.py
  ... (49,998 more lines)

AFTER truncate_output() (~10k chars):
  /usr/lib/python3/file1.py
  /usr/lib/python3/file2.py
  ... (first ~3,300 chars)

  [...47,234 lines truncated...]

  /home/user/project/main.py
  /home/user/project/utils.py
  ... (last ~6,600 chars)
```

### Updated System Prompt Template

```python
DEFAULT_SYSTEM_TEMPLATE = """You are a helpful coding assistant...

Current working directory: {{ cwd }}
Operating system: {{ system }} {{ release }}
{% if project_context %}

Project Context:
{{ project_context }}
{% endif %}
"""
```

The `{% if project_context %}` block ensures the context only appears when `inject_project_context=True`.

### Complete Data Flow

```
┌────────────────────────────────────────────────────────────────────────┐
│                       FULL CONTEXT MANAGEMENT FLOW                      │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  User: "Find and count all Python files"                               │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ run()                                                            │   │
│  │   │                                                              │   │
│  │   ├─► get_project_context()                                      │   │
│  │   │     Returns: {cwd, files, git_status, readme}               │   │
│  │   │                                                              │   │
│  │   ├─► format_context_for_prompt()                                │   │
│  │   │     Returns: "Project: code-agent\nFiles: src/, docs/..."   │   │
│  │   │                                                              │   │
│  │   └─► Inject into system prompt template                         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ query() [THINK]                                                  │   │
│  │   │                                                              │   │
│  │   ├─► budget_context(messages, max_tokens=100k)                  │   │
│  │   │     Input:  50 messages (~150k tokens)                       │   │
│  │   │     Output: 15 messages (~95k tokens)                        │   │
│  │   │                                                              │   │
│  │   └─► Send budgeted messages to LLM                              │   │
│  │         LLM says: "I'll search for Python files\n```bash..."     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ get_observation() [ACT + OBSERVE]                                │   │
│  │   │                                                              │   │
│  │   ├─► Execute: find . -name "*.py"                               │   │
│  │   │     Returns: 50,000 lines of output                          │   │
│  │   │                                                              │   │
│  │   ├─► truncate_output(output, max_chars=10k)                     │   │
│  │   │     Input:  200,000 chars                                    │   │
│  │   │     Output: 10,000 chars (head + [...truncated...] + tail)  │   │
│  │   │                                                              │   │
│  │   └─► Add truncated output to messages                           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Loop back to query() with managed context...                          │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

### Interview Talking Point

> "We integrated context management at three key points in the agent loop:
> 1. **Project context injection** at startup - like Ansible Lightspeed capturing IDE state
> 2. **Token budgeting** before each LLM call - ensures we never exceed context limits
> 3. **Output truncation** after each command - prevents a single large output from blowing up the context
>
> This mirrors production patterns at Red Hat where agents need to handle real-world scenarios like `find /` returning millions of files or very long conversation histories."

---

## See Also

- `src/code_agent/context/manager.py` - Context management utilities
- `src/code_agent/agent/base.py` - Agent with context management wired in
- `docs/STUDY_NOTES.md` - General study notes
- `docs/TOOL_CALLING_NOTES.md` - Tool calling notes
