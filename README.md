# mini-code-agent

A minimal implementation of an AI coding agent in ~500 lines of Python, inspired by [Claude Code](https://claude.ai/claude-code), [mini-swe-agent](https://github.com/princeton-nlp/SWE-agent), and [OpenCode](https://github.com/opencode-ai/opencode).

Built to understand how tools like Claude Code actually work under the hood — by building one from scratch.

## What's Inside

### Code Agent
- **ReAct Loop**: Think → Act → Observe pattern — the same core loop powering production agents
- **Bash-Only**: No tool-calling API needed. The LLM writes bash, we parse and execute it. Works with any model.
- **Multi-Provider**: OpenAI, Anthropic, Google, Ollama, vLLM — all via LiteLLM
- **Three Execution Modes**: Human-in-the-loop, Confirm (approve/reject), YOLO (full autonomy)
- **Safety**: Auto-approves read-only commands, warns on dangerous ones (`rm -rf`, `sudo`)
- **Production Patterns**: Loop detection, cost tracking, context management, retry logic

### RAG Chatbot
- **Document Ingestion**: Token-based chunking (380 tokens, matching OpenShift Lightspeed's config)
- **Vector Search**: FAISS with inner product similarity
- **Context Assembly**: Source citations and relevance filtering
- **Multi-turn Chat**: Conversation history support

## Quick Start

```bash
# Install
pip install -e .

# Set an API key
export ANTHROPIC_API_KEY="your-key"   # or OPENAI_API_KEY, GEMINI_API_KEY

# Run the agent
python -m code_agent.main --task "List all Python files and count them"
```

## Usage

### CLI

```bash
# Confirm mode (default) — approve each command
python -m code_agent.main --task "Find the main function"

# YOLO mode — full autonomy
python -m code_agent.main --task "Count lines of code" --mode yolo

# Human mode — you provide the commands
python -m code_agent.main --task "Refactor this code" --mode human

# Different models
python -m code_agent.main --task "Find TODOs" --model gpt-4o
python -m code_agent.main --task "Find TODOs" --model gemini/gemini-2.0-flash
```

### Programmatic

```python
from code_agent import InteractiveAgent, ExecutionMode, LocalEnvironment, LiteLLMModel

model = LiteLLMModel(model_name="claude-sonnet-4-20250514")
env = LocalEnvironment()

agent = InteractiveAgent(
    model, env,
    mode=ExecutionMode.CONFIRM,
    step_limit=20,
    cost_limit=1.0,
)

status, message = agent.run("List all Python files and count them")
print(f"Status: {status}, Cost: ${model.get_stats()['total_cost']:.4f}")
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--task`, `-t` | Required | The task for the agent |
| `--mode`, `-m` | `confirm` | Execution mode: `human` / `confirm` / `yolo` |
| `--model` | `claude-sonnet-4-20250514` | LLM model to use |
| `--step-limit` | `20` | Maximum iterations |
| `--cost-limit` | `1.0` | Maximum cost in USD |
| `--no-auto-approve` | `False` | Disable auto-approval of safe commands |
| `--verbose`, `-v` | `False` | Show detailed error traces |

## How It Works

### The ReAct Loop

```
┌─────────────────────────────────────────────┐
│                AGENT LOOP                   │
├─────────────────────────────────────────────┤
│                                             │
│  1. THINK:   Query LLM with task + history  │
│      ↓                                      │
│  2. PARSE:   Extract ```bash``` block       │
│      ↓                                      │
│  3. ACT:     Execute via subprocess         │
│      ↓                                      │
│  4. OBSERVE: Add output to history          │
│      ↓                                      │
│  5. LOOP:    Back to THINK until done       │
│                                             │
└─────────────────────────────────────────────┘
```

The key insight: we don't need a tool-calling API. The LLM writes bash naturally (it's seen millions of bash examples in training), and we extract it with a single regex. This makes the agent **model-agnostic** — it works with any LLM, including self-hosted models on vLLM.

### What We Took From Each Project

| Pattern | Inspiration |
|---------|-------------|
| ReAct loop + bash parsing | [mini-swe-agent](https://github.com/princeton-nlp/SWE-agent) |
| Tool registry / extensibility | [OpenCode](https://github.com/opencode-ai/opencode) |
| Context management + safety | [Claude Code](https://claude.ai/claude-code) |
| RAG chunking + FAISS config | [OpenShift Lightspeed](https://www.redhat.com/en/technologies/cloud-computing/openshift/lightspeed) |

## Project Structure

```
src/code_agent/
├── agent/
│   ├── base.py          # Core ReAct loop (~300 lines)
│   └── interactive.py   # Execution modes (human/confirm/yolo)
├── model/
│   └── litellm.py       # Multi-provider LLM with retry & cost tracking
├── environment/
│   └── local.py         # Bash execution via subprocess
├── tools/
│   ├── registry.py      # Extensible tool registry
│   └── builtins.py      # read_file, write_file, glob, grep
├── rag/
│   ├── ingest.py        # Document chunking & FAISS indexing
│   ├── retrieve.py      # Vector search & context assembly
│   └── chat.py          # RAG chatbot interface
├── context/
│   └── manager.py       # Truncation & token budgeting
└── prompts/
    └── system.py        # System prompt templates
```

## Execution Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **HUMAN** | Agent suggests, you provide commands | Learning, sensitive operations |
| **CONFIRM** | Agent proposes, you approve (y/n/e/q) | Normal development (default) |
| **YOLO** | Full autonomy, no prompts | Batch processing, sandboxed environments |

Safe commands (`ls`, `cat`, `grep`, `git status`, etc.) are auto-approved in Confirm mode. Dangerous commands (`rm -rf`, `sudo`, `curl | bash`) always trigger warnings.

## RAG Chatbot

```python
from code_agent.rag import RAGIndex, create_rag_chat

# Index documents
index = RAGIndex(chunk_size=380)
index.add_documents("./docs")
index.save("./my_index")

# Chat
chat = create_rag_chat("./my_index")
response = chat.ask("What is the ReAct pattern?")
print(response["answer"])
```

```bash
# CLI chat
python -m code_agent.rag.chat ./my_index
```

## Dependencies

- **litellm** — Multi-provider LLM abstraction
- **tenacity** — Retry with exponential backoff
- **pydantic** — Config validation
- **faiss-cpu** — Vector similarity search
- **sentence-transformers** — Text embeddings
- **jinja2** — Prompt templating

## Further Reading

- [mini-swe-agent](https://github.com/princeton-nlp/SWE-agent) — the ReAct loop we based ours on
- [OpenCode](https://github.com/opencode-ai/opencode) — tool registry patterns
- [Claude Code](https://claude.ai/claude-code) — context management inspiration
- [OpenShift Lightspeed](https://github.com/openshift/lightspeed-service) — RAG production patterns (FAISS, chunking config)

## License

MIT
