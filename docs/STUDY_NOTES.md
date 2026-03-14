# AI Code Agents: Study Notes

> These notes cover the fundamentals of building AI coding agents, specifically for understanding systems like Claude Code, GitHub Copilot, and Red Hat Lightspeed.

---

## Table of Contents

1. [The ReAct Pattern](#1-the-react-pattern)
2. [How Agents Actually Work](#2-how-agents-actually-work)
3. [Tool-Calling API vs Bash-Only Agents](#3-tool-calling-api-vs-bash-only-agents)
4. [LLM Tool-Calling Support](#4-llm-tool-calling-support)
5. [Frameworks vs From-Scratch](#5-frameworks-vs-from-scratch)
6. [Failure Modes and Production Concerns](#6-failure-modes-and-production-concerns)
7. [Interview Quick Reference](#7-interview-quick-reference)

---

## 1. The ReAct Pattern

### 1.1 What is ReAct?

**ReAct** = **Re**asoning + **Act**ing

A pattern where an LLM alternates between thinking and taking actions in the real world.

```
Think → Act → Observe → Think → Act → Observe → ... → Done
```

### 1.2 The Three Phases

| Phase | Description | Example |
|-------|-------------|---------|
| **Think** | LLM reasons about what to do next | "I need to count Python files" |
| **Act** | LLM outputs a command to execute | `ls *.py \| wc -l` |
| **Observe** | System runs command, returns output | "5" |

### 1.3 Why ReAct Works

- LLMs are good at **reasoning** (trained on tons of problem-solving text)
- LLMs know **bash/code** (trained on GitHub, Stack Overflow)
- The **observation** grounds the LLM in reality (prevents hallucination)
- Each step is **independent** (easy to debug, replay)

### 1.4 Key Insight

> The "intelligence" is entirely in the LLM. The agent is just a loop that shuttles text back and forth between the LLM and the real world.

---

## 2. How Agents Actually Work

### 2.1 The Agent is Just a Loop

```python
while True:
    response = llm.query(messages)      # 1. Ask LLM
    command = parse(response)            # 2. Extract action
    output = execute(command)            # 3. Run it
    messages.append(output)              # 4. Record result
    if is_done(response):               # 5. Check completion
        break
```

### 2.2 Step-by-Step Mechanics

#### Step 1: Initialize Messages
```python
messages = [
    {"role": "system", "content": "You can run bash commands..."},
    {"role": "user", "content": "Count Python files"}
]
```

#### Step 2: Query the LLM
```python
response = llm.query(messages)
# Returns: "I'll count them.\n```bash\nls *.py | wc -l\n```"
```

#### Step 3: Parse the Action (Extract Command)
```python
import re
pattern = r"```bash\s*\n(.*?)\n```"
matches = re.findall(pattern, response, re.DOTALL)
command = matches[0]  # "ls *.py | wc -l"
```

#### Step 4: Execute the Command
```python
import subprocess
result = subprocess.run(command, shell=True, capture_output=True, text=True)
output = result.stdout  # "5\n"
```

#### Step 5: Update Message History
```python
messages.append({"role": "assistant", "content": response})
messages.append({"role": "user", "content": f"Output:\n{output}"})
```

#### Step 6: Loop or Terminate
```python
if "TASK_COMPLETE" in response:
    return "Done!"
# Otherwise, go back to Step 2
```

### 2.3 Message History Grows Each Iteration

```
Iteration 1:
  [system, user_task]

Iteration 2:
  [system, user_task, assistant_response_1, observation_1]

Iteration 3:
  [system, user_task, assistant_response_1, observation_1, assistant_response_2, observation_2]

... and so on
```

### 2.4 Why Message History Matters

- LLM sees **all previous steps** (context)
- Can learn from **failed attempts**
- Can **build on previous outputs**
- Eventually hits **context window limit** (need truncation strategies)

---

## 3. Tool-Calling API vs Bash-Only Agents

### 3.1 Two Approaches to Agent Actions

There are two fundamentally different ways to let an LLM "do things":

| Aspect | Tool-Calling API | Bash-Only |
|--------|------------------|-----------|
| **Output format** | Structured JSON | Free text with code blocks |
| **Parsing** | JSON parsing | Regex extraction |
| **Tool definition** | JSON Schema | Not needed |
| **Works with** | Models with tool support | Any LLM |

### 3.2 Tool-Calling API Approach

#### How It Works

1. You define tools as JSON schemas
2. Send schemas to LLM with your prompt
3. LLM returns structured JSON specifying which tool to call
4. You parse JSON and execute the tool

#### Example Code

```python
# Define tools as JSON schemas
tools = [{
    "name": "read_file",
    "description": "Read contents of a file",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string"}
        },
        "required": ["path"]
    }
}]

# Call API with tools
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Read config.yaml"}],
    tools=tools
)

# LLM returns structured JSON
tool_call = response.choices[0].message.tool_calls[0]
# tool_call = {"name": "read_file", "arguments": {"path": "config.yaml"}}

# Execute the tool
result = read_file(tool_call.arguments["path"])
```

#### Pros
- Structured, predictable output
- Type validation via JSON schema
- Clear separation of tools

#### Cons
- Only works with LLMs that support tool-calling
- More complex setup
- Schema validation can fail unexpectedly

### 3.3 Bash-Only Approach

#### How It Works

1. Tell LLM to output commands in ```bash``` blocks
2. LLM returns free text with embedded commands
3. Extract commands using regex
4. Execute via subprocess

#### Example Code

```python
# System prompt tells LLM the format
system = "Execute commands in ```bash``` blocks"

# LLM returns free text
response = "I'll read the file.\n```bash\ncat config.yaml\n```"

# Extract with regex
import re
pattern = r"```bash\s*\n(.*?)\n```"
command = re.findall(pattern, response, re.DOTALL)[0]
# command = "cat config.yaml"

# Execute via subprocess
import subprocess
result = subprocess.run(command, shell=True, capture_output=True, text=True)
```

#### Pros
- Works with ANY LLM (model-agnostic)
- Simple to debug (just read the text)
- LLM already knows bash
- Fewer failure modes

#### Cons
- Can be brittle if LLM formats incorrectly
- Limited to what bash can do
- Less structured

### 3.4 Visual Comparison

```
TOOL-CALLING:
User → LLM → {"tool": "read_file", "args": {...}} → Parse JSON → Execute

BASH-ONLY:
User → LLM → "```bash\ncat file\n```" → Regex Extract → subprocess.run()
```

### 3.5 When to Use Which

| Use Tool-Calling When | Use Bash-Only When |
|-----------------------|--------------------|
| Using OpenAI/Anthropic API | Need model-agnostic solution |
| Need strict input validation | Using self-hosted models |
| Building with frameworks | Want simple debugging |
| Complex multi-parameter tools | Bash can do what you need |

---

## 4. LLM Tool-Calling Support

### 4.1 API Providers (Tool-Calling Supported)

| Provider | Models | Tool Calling |
|----------|--------|--------------|
| OpenAI | GPT-4, GPT-4o, GPT-3.5-turbo | ✅ Yes |
| Anthropic | Claude 3, 3.5, 4 family | ✅ Yes |
| Google | Gemini Pro, Gemini Flash | ✅ Yes |
| Mistral AI | Mistral Large, Small (API) | ✅ Yes |
| Cohere | Command R, Command R+ | ✅ Yes |

### 4.2 Self-Hosted Models (Often No Tool-Calling)

| Model | Tool Calling |
|-------|--------------|
| Llama 2 (base) | ❌ No |
| Llama 3 / 3.1 | ⚠️ Partial (needs specific format) |
| Mistral 7B (self-hosted) | ❌ No |
| CodeLlama | ❌ No |
| Granite (IBM/Red Hat) | ⚠️ Depends on fine-tuning |
| Phi-3 | ❌ No |
| Mixtral (self-hosted) | ⚠️ Limited |

### 4.3 Why Self-Hosted Models Lack Tool-Calling

Tool-calling requires the entire stack to support it:

```
┌─────────────────────────────────────────┐
│  1. Model Training                      │  ← Was it fine-tuned for tools?
├─────────────────────────────────────────┤
│  2. Prompt Format                       │  ← Does it expect specific tags?
├─────────────────────────────────────────┤
│  3. Serving Layer (vLLM, TGI)          │  ← Does it parse tool calls?
├─────────────────────────────────────────┤
│  4. API Layer                          │  ← Does it expose tool endpoints?
└─────────────────────────────────────────┘

API Providers: Handle ALL 4 layers ✅
Self-Hosted:   Usually only layer 1-2 ⚠️
```

### 4.4 The Platform Problem (Red Hat Context)

When building a platform that serves multiple teams:

```
Team A uses: Claude 3.5 via API      → Tool-calling works ✅
Team B uses: Llama 3 on vLLM         → Tool-calling fails ❌
Team C uses: Custom Granite fine-tune → Tool-calling maybe works ⚠️
```

**Solution**: Use bash-only approach = works with ALL models

### 4.5 Key Insight for Interviews

> "Tool-calling works great with API providers, but Red Hat's platform serves internal teams who may use self-hosted models on vLLM. A bash-only approach is model-agnostic - it works with any model that can generate text."

---

## 5. Frameworks vs From-Scratch

### 5.1 Popular Agent Frameworks

| Framework | Creator | Tool Approach |
|-----------|---------|---------------|
| SmolAgents | HuggingFace | Tool-calling |
| LangChain | LangChain | Both |
| AutoGen | Microsoft | Tool-calling |
| CrewAI | CrewAI | Tool-calling |

### 5.2 What Frameworks Do For You

```python
# With SmolAgents (framework)
from smolagents import CodeAgent, tool, LiteLLMModel

@tool
def read_file(path: str) -> str:
    """Read a file."""
    return open(path).read()

agent = CodeAgent(tools=[read_file], model=LiteLLMModel(...))
agent.run("Read config.yaml")  # That's it!
```

Frameworks handle:
- The agent loop
- Message history management
- Tool schema generation
- Error handling and retries
- Parsing and execution

### 5.3 From-Scratch Approach

```python
# Without framework (~200 lines)
messages = []
while True:
    response = llm.query(messages)
    if "TASK_COMPLETE" in response:
        break
    command = re.findall(r"```bash\n(.*?)\n```", response)[0]
    output = subprocess.run(command, shell=True, capture_output=True)
    messages.append({"role": "assistant", "content": response})
    messages.append({"role": "user", "content": output.stdout})
```

You handle everything manually.

### 5.4 Trade-offs

| Aspect | Framework | From-Scratch |
|--------|-----------|--------------|
| Lines of code | ~10 | ~200 |
| Time to build | Hours | Days |
| Control | Limited | Complete |
| Understanding | Black box | Full visibility |
| Debugging | Harder | Easier |
| Customization | Constrained | Unlimited |

### 5.5 Why Build From-Scratch for Interviews?

| Interview Question | Framework Answer | From-Scratch Answer |
|--------------------|------------------|---------------------|
| "How does the loop work?" | "The framework handles it" ❌ | "It's a while loop that queries, parses, executes" ✅ |
| "How do you handle parse errors?" | "Framework retries" ❌ | "Raise FormatError, add to messages, LLM self-corrects" ✅ |
| "How detect infinite loops?" | "I think framework does something?" ❌ | "Track action_history, check for repetition" ✅ |

### 5.6 Best of Both Worlds

For your interview, you can say:

> "I've used SmolAgents which abstracts the loop and uses tool-calling. But I also built an agent from scratch to understand the fundamentals - the ReAct loop, failure handling, and state management."

---

## 6. Failure Modes and Production Concerns

### 6.1 Common Failure Modes

| Failure Mode | Description | Solution |
|--------------|-------------|----------|
| **Infinite Loop** | Agent repeats same action | Track action history, detect repetition |
| **Parse Error** | No bash block in response | Retry with format reminder |
| **Execution Timeout** | Command hangs | Kill after timeout, return partial output |
| **Context Overflow** | Too many messages | Truncate or summarize old messages |
| **Cost Runaway** | Agent keeps running | Set step/cost limits |

### 6.2 Loop Detection

```python
# Track recent actions
action_history = ["ls", "ls", "ls"]

# Check for repetition
recent = action_history[-3:]
if len(set(recent)) == 1:  # All same
    raise LoopDetected("Agent stuck!")
```

### 6.3 Parse Error Recovery

```python
try:
    command = parse_bash_block(response)
except FormatError:
    # Add error message to conversation
    messages.append({
        "role": "user",
        "content": "Error: Please provide exactly one ```bash``` block"
    })
    # LLM will see this and self-correct on next iteration
```

### 6.4 Timeout Handling

```python
try:
    result = subprocess.run(
        command,
        shell=True,
        timeout=30,  # Kill after 30 seconds
        capture_output=True
    )
except subprocess.TimeoutExpired as e:
    partial_output = e.output.decode() if e.output else ""
    # Return partial output, let LLM try different approach
```

### 6.5 Context Window Management

```python
def truncate_messages(messages, max_tokens=100000):
    """Keep system + recent messages within token limit."""
    # Always keep system message
    system = messages[0]

    # Keep most recent messages
    recent = messages[-20:]

    # Summarize middle if needed
    if token_count(messages) > max_tokens:
        middle_summary = summarize(messages[1:-20])
        return [system, {"role": "user", "content": middle_summary}] + recent

    return messages
```

### 6.6 Cost/Step Limits

```python
class Agent:
    def __init__(self, step_limit=50, cost_limit=1.0):
        self.step_limit = step_limit
        self.cost_limit = cost_limit

    def query(self):
        if self.model.n_calls >= self.step_limit:
            raise LimitsExceeded("Step limit reached")
        if self.model.cost >= self.cost_limit:
            raise LimitsExceeded("Cost limit reached")
        # ... continue with query
```

---

## 7. Interview Quick Reference

### 7.1 Key Concepts to Remember

| Concept | One-Liner |
|---------|-----------|
| **ReAct** | Think → Act → Observe loop |
| **Agent** | Just a while loop shuttling text to/from LLM |
| **Tool-calling** | LLM returns structured JSON for tool invocation |
| **Bash-only** | LLM returns free text, we regex out commands |
| **Model-agnostic** | Works with any LLM, not just tool-calling ones |

### 7.2 Why Bash-Only for Red Hat

> "Red Hat's platform serves teams using diverse models - API providers, self-hosted Llama, custom Granite. Bash-only is model-agnostic, simpler to debug, and has fewer failure modes than tool-calling."

### 7.3 Framework vs From-Scratch

> "Frameworks like SmolAgents are great for rapid prototyping. Building from scratch proves you understand fundamentals - the loop, parsing, failure handling, state management."

### 7.4 Production Concerns

1. **Failure modes**: loops, parse errors, timeouts, context overflow
2. **Cost control**: step limits, cost limits, smaller models for simple tasks
3. **Safety**: command allowlists, sandboxing, human-in-the-loop

### 7.5 Red Hat Specific Context

| System | What It Does |
|--------|--------------|
| **Ansible Lightspeed** | Code completion for Ansible playbooks |
| **OpenShift Lightspeed** | Chat assistant for OpenShift clusters |
| **InstructLab** | Fine-tuning methodology using synthetic data |
| **vLLM** | High-throughput LLM serving |
| **KServe** | Kubernetes-native model serving with autoscaling |

---

## Appendix: Code Reference

### A.1 Minimal Agent (~20 lines)

```python
import re
import subprocess

def run_agent(task, llm):
    messages = [
        {"role": "system", "content": "Use ```bash``` blocks. Say TASK_COMPLETE when done."},
        {"role": "user", "content": task}
    ]

    while True:
        response = llm.query(messages)
        messages.append({"role": "assistant", "content": response})

        if "TASK_COMPLETE" in response:
            return "Done!"

        command = re.findall(r"```bash\n(.*?)\n```", response, re.DOTALL)[0]
        output = subprocess.run(command, shell=True, capture_output=True, text=True)
        messages.append({"role": "user", "content": f"Output:\n{output.stdout}"})
```

### A.2 Key Files in Our Project

| File | Purpose |
|------|---------|
| `src/code_agent/agent/base.py` | Core agent loop (~240 lines) |
| `src/code_agent/model/litellm.py` | LLM wrapper with retry/cost tracking |
| `src/code_agent/environment/local.py` | Bash execution via subprocess |

---

*Notes compiled from interview prep session for Red Hat Lightspeed Core*
