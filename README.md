# Prompt Oriented Programming (POP)

```python
from POP import PromptFunction

pf = PromptFunction(
    prompt="Draw a simple ASCII art of <<<object>>>.",
    client = "openai"
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

This release cleans the architecture, moves all LLM client logic to a separate `LLMClient` module, and extends multi-LLM backend support.

PyPI:
[https://pypi.org/project/pypop/](https://pypi.org/project/pypop/)

GitHub:
[https://github.com/sgt1796/POP](https://github.com/sgt1796/POP)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Major Updates](#2-major-updates)
3. [Features](#3-features)
4. [Installation](#4-installation)
5. [Setup](#5-setup)
6. [PromptFunction](#6-promptfunction)

   * Placeholders
   * Reserved Keywords
   * Executing prompts
   * Improving prompts
7. [Function Schema Generation](#7-function-schema-generation)
8. [Embeddings](#8-embeddings)
9. [Web Snapshot Utility](#9-web-snapshot-utility)
10. [Examples](#10-examples)
11. [Contributing](#11-contributing)
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
* work with multiple LLM providers through `LLMClient` subclasses

POP is designed to be simple, extensible, and production-friendly.

---

# 2. Major Updates

This version introduces structural and functional improvements:

### 2.1. LLMClient moved into its own module

`LLMClient.py` now holds all LLM backends:

* OpenAI
* Gemini
* Deepseek
* Doubao
* Local PyTorch stub
* Extensible architecture for adding new backends

### 2.2. Expanded multi-LLM support

Each backend now has consistent interface behavior and multimodal (text + image) support where applicable.

---

# 3. Features

* **Reusable Prompt Functions**
  Use `<<<placeholder>>>` syntax to inject dynamic content.

* **Multi-LLM Backend**
  Choose between OpenAI, Gemini, Deepseek, Doubao, or local models.

* **Prompt Improvement**
  Improve or rewrite prompts using Fabric-style metaprompts.

* **Function Schema Generation**
  Convert natural language descriptions into OpenAI-function schemas.

* **Unified Embedding Interface**
  Supports OpenAI, Jina AI embeddings, and local HuggingFace models.

* **Webpage Snapshot Utility**
  Convert any URL into structured text using r.jina.ai with optional image captioning.

---

# 4. Installation

Install from PyPI:

```bash
pip install pypop
```

Or install in development mode from GitHub:

```bash
git clone https://github.com/sgt1796/POP.git
cd POP
pip install -e .
```

---

# 5. Setup

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

# 6. PromptFunction

The core abstraction of POP is the `PromptFunction` class.

```python
from pypop import PromptFunction

pf = PromptFunction(
    sys_prompt="You are a helpful AI.",
    prompt="Give me a summary about <<<topic>>>."
)

print(pf.execute(topic="quantum biology"))
```

---

## 6.1. Placeholder Syntax

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

## 6.2. Reserved Keywords

Within `.execute()`, the following keyword arguments are **reserved** and should not be used as placeholder names:

* `model`
* `sys`
* `fmt`
* `tools`
* `temp`
* `images`
* `ADD_BEFORE`
* `ADD_AFTER`

Most keywords are used for parameters. `ADD_BEFORE` and `ADD_AFTER` will attach input string to head/tail of the prompt.

---

## 6.3. Executing prompts

```python
result = pf.execute(
    topic="photosynthesis",
    model="gpt-4o-mini",
    temp=0.3
)
```

---

## 6.4. Improving Prompts

You can ask POP to rewrite or enhance your system prompt:

```python
better = pf._improve_prompt()
print(better)
```

This uses a Fabric-inspired meta-prompt bundled in the `prompts/` directory.

---

# 7. Function Schema Generation

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
* Optionally saves it under `functions/`

---

# 8. Embeddings

POP includes a unified embedding interface:

```python
from pypop.Embedder import Embedder

embedder = Embedder(use_api="openai")
vecs = embedder.get_embedding(["hello world"])
```

Supported modes:

* OpenAI embeddings
* JinaAI embeddings
* Local HuggingFace model embeddings (cpu/gpu)

Large inputs are chunked automatically when needed.

---

# 9. Web Snapshot Utility

```python
from pypop import get_text_snapshot

text = get_text_snapshot("https://example.com", image_caption=True)
print(text[:500])
```

Supports:

* optional image removal
* optional image captioning
* DOM selector filtering
* returning JSON or plain text

---

# 10. Examples

```python
from pypop import PromptFunction

pf = PromptFunction(prompt="Give me 3 creative names for a <<<thing>>>.")

print(pf.execute(thing="robot"))
print(pf.execute(thing="new language"))
```

---

# 11. Contributing

Steps:

1. Fork the GitHub repo
2. Create a feature branch
3. Add tests or examples
4. Submit a PR with a clear explanation
