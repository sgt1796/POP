# Prompt Oriented Programming (POP)

```python
from pop import PromptFunction

pf = PromptFunction(
    prompt="Draw a simple ASCII art of <<<object>>>.",
    client="openai",
)

print(pf.execute(object="a cat"))
print(pf.execute(object="a rocket"))
```

```
 /\_/\  
( o.o )
 > ^ <  

   /\
  /  \
 /    \
 |    |
 |    |
```

---
Reusable, composable prompt functions for LLM workflows.

This 1.1.0 dev update restructures POP into small, focused modules and adds a provider registry inspired by pi-mono's `ai` package.

PyPI:
[https://pypi.org/project/pop-python/](https://pypi.org/project/pop-python/)

GitHub:
[https://github.com/sgt1796/POP](https://github.com/sgt1796/POP)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Update Note](#2-update-note)
3. [Major Updates](#3-major-updates)
4. [Features](#4-features)
5. [Installation](#5-installation)
6. [Setup](#6-setup)
7. [PromptFunction](#7-promptfunction)
8. [Provider Registry](#8-provider-registry)
9. [Tool Calling](#9-tool-calling)
10. [Function Schema Generation](#10-function-schema-generation)
11. [Embeddings](#11-embeddings)
12. [Web Snapshot Utility](#12-web-snapshot-utility)
13. [Examples](#13-examples)
14. [Contributing](#14-contributing)
---

# 1. Overview

Prompt Oriented Programming (POP) is a lightweight framework for building reusable, parameterized prompt functions.
Instead of scattering prompt strings across your codebase, POP lets you:

* encapsulate prompts as objects
* pass parameters cleanly via placeholders
* select a backend LLM client dynamically
* improve prompts using meta-prompting
* generate OpenAI-compatible function schemas
* use unified embedding tools
* work with multiple LLM providers through a centralized registry

POP is designed to be simple, extensible, and production-friendly.

---

# 2. Update Note

**1.1.0-dev (February 5, 2026)**

* **Breaking import path**: use `pop` (lowercase) for imports. Example: `from pop import PromptFunction`.
* **Provider registry**: clients live under `pop/providers/` and are instantiated via `pop.api_registry`.
* **LLMClient base class**: now in `pop.providers.llm_client` (kept as an abstract base class).

---

# 3. Major Updates

### 3.1. Modularized architecture

The project has been decomposed into small, focused modules:

* `pop/prompt_function.py`
* `pop/embedder.py`
* `pop/context.py`
* `pop/api_registry.py`
* `pop/providers/` (one provider per file)
* `pop/utils/`

This mirrors the structure in the pi-mono `ai` package for clarity and maintainability.

### 3.2. Provider registry + per-provider clients

Each provider has its own adaptor (OpenAI, Gemini, DeepSeek, Doubao, Local, Ollama). The registry gives you:

* `list_providers()`
* `list_default_model()`
* `list_models()`
* `get_client()`

---

# 4. Features

* **Reusable Prompt Functions**
  Use `<<<placeholder>>>` syntax to inject dynamic content.

* **Multi-LLM Backend**
  Choose between OpenAI, Gemini, DeepSeek, Doubao, Local, or Ollama.

* **Tool Calling**
  Pass a tool schema list to `execute()` and receive tool-call arguments.

* **Multimodal (Text + Image)**
  Pass `images=[...]` (URLs or base64) when the provider supports it.

* **Prompt Improvement**
  Improve or rewrite prompts using Fabric-style meta-prompts.

* **Function Schema Generation**
  Convert natural language descriptions into OpenAI-function schemas.

* **Unified Embedding Interface**
  Supports OpenAI, Jina AI embeddings, and local HuggingFace models.

* **Webpage Snapshot Utility**
  Convert any URL into structured text using r.jina.ai with optional image captioning.

---

# 5. Installation

Install from PyPI:

```bash
pip install pop-python
```

Or install in development mode from GitHub:

```bash
git clone https://github.com/sgt1796/POP.git
cd POP
pip install -e .
```

---

# 6. Setup

Create a `.env` file in your project root:

```ini
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gcp_gemini_key
DEEPSEEK_API_KEY=your_deepseek_key
DOUBAO_API_KEY=your_volcengine_key
JINAAI_API_KEY=your_jina_key
```

All clients automatically read keys from environment variables.

---

# 7. PromptFunction

The core abstraction of POP is the `PromptFunction` class.

```python
from pop import PromptFunction

pf = PromptFunction(
    sys_prompt="You are a helpful AI.",
    prompt="Give me a summary about <<<topic>>>.",
)

print(pf.execute(topic="quantum biology"))
```

---

## 7.1. Placeholder Syntax

Use angle-triple-brackets inside your prompt:

```
<<<placeholder>>>
```

These are replaced at execution time.

Example:

```python
prompt = "Translate <<<sentence>>> to French."
```

---

## 7.2. Reserved Keywords

Within `.execute()`, the following keyword arguments are **reserved** and should not be used as placeholder names:

* `model`
* `sys`
* `fmt`
* `tools`
* `tool_choice`
* `temp`
* `images`
* `ADD_BEFORE`
* `ADD_AFTER`

Most keywords are used for parameters. `ADD_BEFORE` and `ADD_AFTER` will attach input string to head/tail of the prompt.

---

## 7.3. Executing prompts

```python
result = pf.execute(
    topic="photosynthesis",
    model="gpt-5-mini",
    temp=0.3,
)
```

---

## 7.4. Improving Prompts

You can ask POP to rewrite or enhance your system prompt:

```python
better = pf.improve_prompt()
print(better)
```

This uses a Fabric-inspired meta-prompt bundled in the `pop/prompts/` directory.

---

# 8. Provider Registry

Use the registry to list providers/models or instantiate clients.

```python
from pop import list_providers, list_models, list_default_model, get_client

print(list_providers())
print(list_default_model())
print(list_models())

client = get_client("openai")
```

Non-default model example:

```python
from pop import PromptFunction, get_client

client = get_client("gemini", "gemini-2.5-pro")

pf = PromptFunction(prompt="Draw a rocket.", client=client)
print(pf.execute())
```

Direct provider class example:

```python
from pop import PromptFunction
from pop.providers.gemini_client import GeminiClient

pf = PromptFunction(prompt="Draw a rocket.", client=GeminiClient(model="gemini-2.5-pro"))
print(pf.execute())
```

---

# 9. Tool Calling

```python
from pop import PromptFunction

tools = [
    {
        "type": "function",
        "function": {
            "name": "create_reminder",
            "description": "Create a reminder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "when": {"type": "string"},
                },
                "required": ["description"],
            },
        },
    }
]

pf = PromptFunction(
    sys_prompt="You are a helpful assistant.",
    prompt="<<<input>>>",
    client="openai",
)

result = pf.execute(input="Remind me to walk at 9am.", tools=tools)
print(result)
```

---

# 10. Function Schema Generation

POP supports generating **OpenAI function-calling schemas** from natural language descriptions.

```python
schema = pf.generate_schema(
    description="Return the square and cube of a given integer."
)

print(schema)
```

What this does:

* Applies a standard meta-prompt
* Uses the selected LLM backend
* Produces a valid JSON Schema for OpenAI function calling
* Optionally saves it under `schemas/`

---

# 11. Embeddings

POP includes a unified embedding interface:

```python
from pop import Embedder

embedder = Embedder(use_api="openai")
vecs = embedder.get_embedding(["hello world"])
```

Supported modes:

* OpenAI embeddings
* JinaAI embeddings
* Local HuggingFace model embeddings (cpu/gpu)

Large inputs are chunked automatically when needed.

---

# 12. Web Snapshot Utility

```python
from pop.utils.web_snapshot import get_text_snapshot

text = get_text_snapshot("https://example.com", image_caption=True)
print(text[:500])
```

Supports:

* optional image removal
* optional image captioning
* DOM selector filtering
* returning JSON or plain text

---

# 13. Examples

```python
from pop import PromptFunction

pf = PromptFunction(prompt="Give me 3 creative names for a <<<thing>>>.")

print(pf.execute(thing="robot"))
print(pf.execute(thing="new language"))
```

Multimodal example (provider must support images):

```python
from pop import PromptFunction

image_b64 = "..."  # base64-encoded image

pf = PromptFunction(prompt="Describe the image.", client="openai")
print(pf.execute(images=[image_b64]))
```

---

# 14. Contributing

Steps:

1. Fork the GitHub repo
2. Create a feature branch
3. Add tests or examples
4. Submit a PR with a clear explanation
