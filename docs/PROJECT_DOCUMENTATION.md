# Code Agent Project Documentation

## Table of Contents

1. Introduction
2. Architecture Overview
3. Core Components
4. The ReAct Pattern
5. Agent Implementation
6. Interactive Execution Modes
7. Model Abstraction Layer
8. Context Management
9. RAG System
10. Tool Registry
11. Configuration Options
12. Usage Examples
13. Interview Preparation Topics

## 1. Introduction

### Project Purpose

This project demonstrates the implementation of an AI coding agent using the ReAct (Reasoning and Acting) pattern, combined with a Retrieval Augmented Generation (RAG) chatbot. It was built to understand and demonstrate production grade AI agent patterns used in systems like Claude Code, GitHub Copilot, and Red Hat Lightspeed.

### Key Design Decisions

The agent uses a bash only approach instead of tool calling APIs. This design choice provides several advantages:

1. Model Agnostic: Works with any LLM including self hosted models on vLLM
2. Simpler Debugging: Output is plain text, easy to read and debug
3. Natural Language: LLMs already know bash from training data
4. Fewer Failure Modes: No JSON schema validation issues

### Technology Stack

The project uses Python 3.10 or higher with the following key dependencies:

LiteLLM for multi provider LLM abstraction
Tenacity for retry logic with exponential backoff
Pydantic for configuration validation
FAISS for vector similarity search
Sentence Transformers for text embeddings
Jinja2 for prompt templating

## 2. Architecture Overview

### High Level Architecture

The system consists of two main components:

Code Agent: An interactive agent that executes tasks by running bash commands
RAG Chatbot: A document based question answering system

### Data Flow

The agent follows this flow for each task:

1. User provides a task description
2. System prompt and task are sent to the LLM
3. LLM responds with reasoning and a bash command
4. Agent extracts the command using regex
5. Command is executed via subprocess
6. Output is added to message history
7. Process repeats until task completion

### Component Interaction Diagram

```
User Task
    |
    v
Agent (base.py)
    |
    +--- Model (litellm.py) --- LLM API
    |
    +--- Environment (local.py) --- Subprocess
    |
    +--- Context Manager (manager.py)
    |
    v
Result
```

## 3. Core Components

### Agent Module

Located in agent/ directory:

base.py contains the main ReAct loop implementation with approximately 300 lines of code. It handles the core think, act, observe cycle.

interactive.py provides human in the loop execution modes with safe command detection and user approval workflows.

### Model Module

Located in model/ directory:

litellm.py wraps the LiteLLM library to provide retry logic with exponential backoff, cost tracking per API call, token counting, and simplified response format.

### Environment Module

Located in environment/ directory:

local.py executes bash commands via subprocess with timeout handling, output capture, and error handling.

### Context Module

Located in context/ directory:

manager.py manages the context window with output truncation using head plus tail pattern, message history budgeting, and project context injection.

### RAG Module

Located in rag/ directory:

ingest.py handles document processing and FAISS indexing
retrieve.py performs vector search and context assembly
chat.py provides the RAG chatbot interface

### Tools Module

Located in tools/ directory:

registry.py provides an extensible tool registration system
builtins.py contains built in tools like read file, write file, glob, and grep

## 4. The ReAct Pattern

### Definition

ReAct stands for Reasoning plus Acting. It is a pattern where an LLM alternates between thinking about what to do and taking actions in the real world.

### The Loop Structure

The pattern follows this cycle:

Think: LLM reasons about the next step
Act: LLM outputs a command to execute
Observe: System runs command and returns output
Repeat: Continue until task is complete

### Why ReAct Works

LLMs are trained on vast amounts of problem solving text, making them good at reasoning. They also know bash and code from training on GitHub and Stack Overflow. The observation step grounds the LLM in reality, preventing hallucination. Each step is independent, making debugging easy.

### State Management

The agent uses linear message history, which is the simplest approach. There is no complex state machine. Each step is independent and easy to replay or debug.

## 5. Agent Implementation

### Base Agent Class

The Agent class in base.py implements the core ReAct loop.

Key methods:

run(): Main entry point that initializes messages and runs the loop
step(): Executes one iteration consisting of query, parse, and execute
query(): Sends messages to LLM and checks limits
parse_action(): Extracts bash command using regex
execute_action(): Runs command and detects loops
get_observation(): Formats output for message history

### Action Parsing

Commands are extracted from LLM responses using regex. The pattern looks for content inside triple backtick bash blocks. The agent expects exactly one bash block per response. If zero or multiple blocks are found, a FormatError is raised and the agent retries.

### Exception Hierarchy

The agent uses a two tier exception system:

NonTerminatingException: Recoverable errors that trigger retry
  FormatError for invalid output format
  ExecutionTimeoutError for command timeout

TerminatingException: Errors that end the run
  Submitted when task completed successfully
  LimitsExceeded when step or cost limit reached
  LoopDetected when agent stuck in repetition

### Loop Detection

The agent tracks recent actions and detects when the same action is repeated multiple times in a row. By default, if the last three actions are identical, a LoopDetected exception is raised.

## 6. Interactive Execution Modes

### Three Modes Available

HUMAN Mode: The agent suggests commands but the user provides the actual command to run. Best for learning, debugging, or extremely sensitive operations.

CONFIRM Mode: The agent proposes commands and the user approves or rejects them. This is the default mode and best for normal development.

YOLO Mode: Full autonomy with no prompts. The agent executes everything automatically. Best for CI/CD pipelines, batch processing, or sandboxed environments.

### Safe Command Detection

Read only operations are automatically approved in CONFIRM mode:

ls, cat, head, tail, grep, find
pwd, echo, wc, tree
git status, git log, git diff, git branch

### Dangerous Command Detection

These commands trigger warnings even in YOLO mode:

rm with recursive force flag
sudo for superuser access
curl piped to bash for remote code execution

### User Interaction in CONFIRM Mode

When a command needs approval, the user sees options:

y or yes: Approve and execute
n or no: Reject and let agent try different approach
e or edit: Modify the command before executing
q or quit: Exit the session

## 7. Model Abstraction Layer

### Why LiteLLM

LiteLLM provides a unified interface to call over 100 LLM providers with the same code. Without it, you would need different code for each provider with different imports, methods, and response formats.

### Our Wrapper Features

The LiteLLMModel class adds production features:

Retry Logic: Automatic retry with exponential backoff on API failures. Attempts increase wait time from 2 seconds up to 30 seconds.

Cost Tracking: Tracks cost per API call and cumulative cost. Essential for budget management.

Token Counting: Tracks input and output tokens. Useful for debugging context window issues.

Simplified Response: Raw LiteLLM returns complex objects. Our wrapper returns a simple dictionary with content and extra fields.

### Configuration

```python
model = LiteLLMModel(
    model_name="claude-sonnet-4-20250514",
    temperature=0.0,
    max_tokens=4096,
    cost_tracking=True
)
```

### Statistics Access

```python
stats = model.get_stats()
# Returns: n_calls, total_cost, total_input_tokens, total_output_tokens
```

## 8. Context Management

### The Problem

Every LLM has a token limit (context window). As an agent runs, context grows with each iteration. Without management, you will hit API errors or silent truncation.

### Where Management Happens

Project context injection happens at the start of run()
Token budgeting happens before each LLM call in query()
Output truncation happens after each command in get_observation()

### Output Truncation Strategy

We use head plus tail truncation. This keeps the beginning (headers, column names) and end (results, errors) while removing the middle. This preserves the most useful information.

Configuration:

```python
agent = Agent(model, env, max_output_chars=10000)
```

### Message Budgeting

When messages exceed the token budget, old messages are dropped while keeping the system message and most recent exchanges.

Configuration:

```python
agent = Agent(model, env, max_context_tokens=100000)
```

### Project Context Injection

The agent can automatically gather and inject project information into the system prompt, similar to how Ansible Lightspeed captures IDE context.

Configuration:

```python
agent = Agent(model, env, inject_project_context=True)
```

## 9. RAG System

### What is RAG

RAG stands for Retrieval Augmented Generation. Instead of training a model on your data, you retrieve relevant documents at query time and inject them into the prompt.

### Pipeline Overview

Indexing Phase (offline, done once):
Documents are chunked into smaller pieces
Each chunk is embedded into a vector
Vectors are stored in FAISS index

Query Phase (online, every request):
User question is embedded
Similar chunks are found via vector search
Retrieved chunks are assembled into context
LLM generates answer based on context

### Chunking Strategy

We use recursive character splitting with a default chunk size of 380 tokens (same as OpenShift Lightspeed) and zero overlap.

Chunk size tradeoffs:
Too small loses context
Too large dilutes relevance
380 tokens provides good balance

### Vector Store

We use FAISS with inner product similarity for normalized vectors. FAISS is fast, free, and good for collections under one million documents.

### Retrieval Quality

search_with_threshold: Filters results below minimum similarity score
assemble_context: Combines chunks with source citations
format_rag_prompt: Creates prompt with grounding instructions

### RAG Chat Interface

```python
chat = RAGChat(index, model, top_k=5, min_score=0.3)
response = chat.ask("What is the ReAct pattern?")
```

Response includes answer, sources, number of results, and model statistics.

## 10. Tool Registry

### Purpose

The tool registry provides an extensible system for adding capabilities beyond bash commands. It demonstrates platform thinking where internal teams can add domain specific tools.

### The Decorator Pattern

Tools are defined using the @tool decorator which attaches metadata to functions:

```python
@tool(name="read_file", description="Read contents of a file")
def read_file(path: str) -> str:
    return open(path).read()
```

### The Registry Class

ToolRegistry stores tools and provides execution:

```python
registry = ToolRegistry()
registry.register(read_file)
result = registry.execute("read_file", path="config.yaml")
```

### Instance Based vs Global Registration

Our approach uses instance based registration where each registry is separate. This allows different agents to have different tool sets. This is important for platforms serving multiple teams with different needs.

### Built in Tools

read_file: Read file contents
write_file: Write or create files
glob_files: Find files by pattern
grep_search: Search file contents
list_directory: List directory contents

## 11. Configuration Options

### AgentConfig

```python
class AgentConfig:
    system_template: str          # Jinja2 template for system prompt
    instance_template: str        # Template for task description
    action_regex: str             # Regex to extract bash commands
    step_limit: int = 50          # Maximum iterations (0 for unlimited)
    cost_limit: float = 1.0       # Maximum cost in USD
    loop_threshold: int = 3       # Repeated actions before loop detection
    max_output_chars: int = 10000 # Truncation limit for outputs
    max_context_tokens: int = 100000  # Token budget for messages
    inject_project_context: bool = True  # Add project info to prompt
```

### CLI Arguments

task (required): The task for the agent
mode: human, confirm, or yolo (default: confirm)
model: LLM model name (default: claude-sonnet-4-20250514)
step_limit: Maximum iterations (default: 20)
cost_limit: Maximum cost in USD (default: 1.0)
no_auto_approve: Disable safe command auto approval
verbose: Show detailed error traces

## 12. Usage Examples

### CLI Usage

Basic usage with confirm mode:
```
python -m code_agent.main --task "List all Python files"
```

Full autonomy mode:
```
python -m code_agent.main --task "Find the main function" --mode yolo
```

With different model:
```
python -m code_agent.main --task "Count lines" --model gpt-4o
```

### Programmatic Usage

```python
from code_agent import InteractiveAgent, ExecutionMode
from code_agent import LocalEnvironment, LiteLLMModel

model = LiteLLMModel(model_name="claude-sonnet-4-20250514")
env = LocalEnvironment()

agent = InteractiveAgent(
    model, env,
    mode=ExecutionMode.CONFIRM,
    step_limit=20,
    cost_limit=1.0
)

status, message = agent.run("List Python files")
print(f"Status: {status}")
print(f"Cost: ${model.get_stats()['total_cost']:.4f}")
```

### RAG Usage

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

## 13. Interview Preparation Topics

### Agent Track Topics

ReAct Pattern: Explain the think, act, observe loop and why it works

Bash Only Approach: Discuss model agnostic design, simpler debugging, and fewer failure modes

Failure Modes: Cover infinite loops, parse errors, execution timeouts, and context overflow

State Management: Explain linear message history and why it is the simplest approach

Interactive Modes: Describe human, confirm, and yolo modes with use cases

### RAG Track Topics

Chunking Strategies: Discuss size tradeoffs, OpenShift Lightspeed uses 380 tokens

Vector Stores: Compare FAISS (fast, free, in memory) vs managed solutions

Retrieval Quality: Cover hybrid search, reranking, and threshold filtering

RAG vs Fine Tuning: RAG for dynamic knowledge with citations, fine tuning for consistent style

### Production Considerations

Cost Control: Step limits, cost limits, smaller models for simple tasks

Reliability: Retry logic with exponential backoff, graceful degradation

Safety: Safe command detection, dangerous command warnings, human in the loop

Context Management: Output truncation, message budgeting, project context injection

### Red Hat Specific Topics

Ansible Lightspeed: Context from VS Code, code completion for playbooks

OpenShift Lightspeed: Chat assistant for clusters, BYOK documentation

InstructLab: Fine tuning methodology using synthetic data

vLLM and KServe: High throughput model serving on Kubernetes

### Key Talking Points

The agent is just a loop that shuttles text between the LLM and the real world

Bash only approach is model agnostic which is critical for platforms serving diverse teams

Context management is essential because observations loop back and can explode the context window

Production systems need retry logic, cost tracking, and human in the loop options

RAG is better than fine tuning when knowledge changes frequently and you need citations
