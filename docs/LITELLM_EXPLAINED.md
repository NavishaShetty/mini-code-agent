# LiteLLM: Understanding the Model Abstraction Layer

> Quick reference for understanding LiteLLM and why we wrap it.

---

## What is LiteLLM?

**LiteLLM** is a Python library that provides a **unified interface** to call 100+ LLM providers with the same code.

### The Problem It Solves

```python
# WITHOUT LiteLLM - Different code for each provider:

# OpenAI
from openai import OpenAI
client = OpenAI(api_key="...")
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}]
)
text = response.choices[0].message.content

# Anthropic - DIFFERENT API!
from anthropic import Anthropic
client = Anthropic(api_key="...")
response = client.messages.create(
    model="claude-3-sonnet-20240229",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}]
)
text = response.content[0].text

# Each provider = different imports, different methods, different response formats 😫
```

```python
# WITH LiteLLM - Same code for ALL providers:

import litellm

# OpenAI
response = litellm.completion(model="gpt-4", messages=[...])

# Anthropic
response = litellm.completion(model="claude-3-sonnet-20240229", messages=[...])

# Ollama (local)
response = litellm.completion(model="ollama/llama3", messages=[...])

# vLLM (self-hosted)
response = litellm.completion(model="openai/my-model", api_base="http://localhost:8000", messages=[...])

# Same interface everywhere! 🎉
```

### Analogy

> LiteLLM is like a **universal power adapter**. Instead of carrying different chargers for US/EU/UK/Australia, you carry one adapter that works everywhere.

---

## Supported Providers

| Provider | Model Format Example |
|----------|---------------------|
| OpenAI | `gpt-4`, `gpt-3.5-turbo` |
| Anthropic | `claude-3-opus-20240229` |
| Google | `gemini/gemini-pro` |
| Mistral | `mistral/mistral-large` |
| Cohere | `cohere/command-r` |
| Ollama | `ollama/llama3` |
| vLLM | `openai/model-name` + `api_base` |
| Azure | `azure/deployment-name` |
| AWS Bedrock | `bedrock/anthropic.claude-3` |
| And 100+ more... | |

---

## Basic Usage

```python
import litellm

# Simple call
response = litellm.completion(
    model="claude-sonnet-4-20250514",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"}
    ],
    temperature=0.0,
    max_tokens=100
)

# Get the response text
text = response.choices[0].message.content
print(text)  # "4"

# Get usage info
print(response.usage.prompt_tokens)      # e.g., 25
print(response.usage.completion_tokens)  # e.g., 1
```

---

## Why We Wrap LiteLLM

LiteLLM handles provider abstraction, but production systems need more:

| Feature | Raw LiteLLM | Our Wrapper |
|---------|-------------|-------------|
| Multi-provider support | ✅ | ✅ |
| Retry on failure | ❌ Manual | ✅ Automatic with exponential backoff |
| Cost tracking | ❌ Manual | ✅ Automatic per-call |
| Token counting | ❌ One-off | ✅ Cumulative tracking |
| Simple response | ❌ Complex object | ✅ `{content, extra}` |
| Configuration | ❌ Per-call params | ✅ Configured once |

---

## Our Wrapper: Key Features

### 1. Retry with Exponential Backoff

```python
# If API fails, we retry with increasing delays:
# Attempt 1 → fail → wait 2s
# Attempt 2 → fail → wait 4s
# Attempt 3 → fail → wait 8s
# Attempt 4 → success!

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(min=2, max=30),
)
def _query(self, messages):
    return litellm.completion(...)
```

**Why?** APIs fail under load, rate limits hit, servers hiccup. Retry logic is essential for production.

### 2. Cost Tracking

```python
model = LiteLLMModel(model_name="gpt-4")
model.query([{"role": "user", "content": "Hello"}])
model.query([{"role": "user", "content": "World"}])

print(model.get_stats())
# {"n_calls": 2, "total_cost": 0.0045, ...}
```

**Why?** LLMs are expensive. Need visibility into spend.

### 3. Token Tracking

```python
print(model.get_stats())
# {
#   "n_calls": 5,
#   "total_cost": 0.0234,
#   "total_input_tokens": 1250,
#   "total_output_tokens": 890
# }
```

**Why?** Understand usage patterns, debug context window issues.

### 4. Simplified Response

```python
# Raw LiteLLM returns complex object:
ModelResponse(
    id='chatcmpl-abc',
    choices=[Choice(message=Message(content="Hi", role="assistant"))],
    usage=Usage(prompt_tokens=10, completion_tokens=2),
    ...many more fields...
)

# Our wrapper returns simple dict:
{
    "content": "Hi",
    "extra": {"model": "gpt-4", "usage": {...}, "cost": 0.001}
}
```

---

## Configuration Options

```python
from code_agent.model.litellm import LiteLLMModel

model = LiteLLMModel(
    model_name="claude-sonnet-4-20250514",  # Which model
    temperature=0.0,                          # 0=deterministic, 1=creative
    max_tokens=4096,                          # Max response length
    cost_tracking=True,                       # Track costs
    model_kwargs={                            # Extra params
        "top_p": 0.9,
    }
)
```

---

## Interview Relevance

### Why This Matters for Red Hat

1. **Platform Thinking**: LiteLLM lets teams choose their preferred model
   - Team A uses Claude API
   - Team B uses self-hosted Llama on vLLM
   - Same agent code works for both

2. **Cost Control**: Production systems need cost visibility
   - Track spend per team/project
   - Alert on unusual usage

3. **Reliability**: Retry logic is production-essential
   - APIs fail, especially under load
   - Exponential backoff is industry standard

### Talking Points

> "We use LiteLLM for provider abstraction - it lets us support OpenAI, Anthropic, and self-hosted models with the same interface. Our wrapper adds production features: retry logic with exponential backoff, cost tracking, and simplified response handling."

---

## Quick Reference

```python
# Initialize
from code_agent.model.litellm import LiteLLMModel
model = LiteLLMModel(model_name="claude-sonnet-4-20250514")

# Query
response = model.query([
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello"}
])
print(response["content"])  # The response text

# Check stats
print(model.get_stats())
# {"n_calls": 1, "total_cost": 0.001, "total_input_tokens": 15, "total_output_tokens": 5}
```

---

## See Also

- [LiteLLM Documentation](https://docs.litellm.ai/)
- [Tenacity (retry library)](https://tenacity.readthedocs.io/)
- Source: `src/code_agent/model/litellm.py`
