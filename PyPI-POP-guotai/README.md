# Prompt Oriented Programming (POP)

**Reusable, composable prompt functions for LLMs.**
POP treats prompts like first-class functions: reusable, mutable, and structured for programmatic execution. It supports prompt enhancement, function/code generation, multiple LLM backends, and embeddings.

**PyPI Link:** [https://pypi.org/project/POP-guotai/](https://pypi.org/project/POP-guotai/)

---

## Table of Contents

0. [Updates](#updates)
1. [Features](#features)
2. [Installation](#installation)
3. [Setup & Configuration](#setup--configuration)
4. [Usage](#usage)

   * [PromptFunction Class](#promptfunction-class)
   * [Improving Prompts](#improving-prompts)
   * [Function Schema & Code Generation](#function-schema--code-generation)
   * [Embeddings](#embeddings)
   * [Web Snapshot Utility](#web-snapshot-utility)
5. [Example](#example)
6. [Future Plans](#future-plans)
7. [Contributing](#contributing)

---

## Updates

* **0.3.1**: add image support to gemini and openai client

---

## Features

* **Prompt as a Function**: Define reusable prompts with `<<<placeholders>>>` for flexible execution.
* **Multi-LLM Support**: Use OpenAI (default), GCP Gemini, local PyTorch, or Deepseek stubs.
* **Function Schema & Code Generation**:

  * Turn natural language descriptions into OpenAI function schemas.
  * Optionally generate full Python function code with docstrings and type hints.
* **Prompt Improvement**: Enhance base prompts using Fabric-inspired meta-prompts.
* **Embeddings**:

  * OpenAI and Jina API embeddings
  * Local Hugging Face model support
* **Utility Functions**:

  * `get_text_snapshot(url)` for webpage-to-text extraction with optional image captioning.

---

## Installation

Install from PyPI:

```bash
pip install POP-guotai
```

Or from source:

```bash
git clone https://github.com/sgt1796/POP.git
cd POP
pip install -e .
```

---

## Setup & Configuration

1. Create a `.env` file in your project root:

```ini
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gcp_gemini_key
JINAAI_API_KEY=your_jina_api_key
```

2. Dependencies are automatically handled via `setup.py`:

* `openai`, `requests`, `python-dotenv`, `pydantic`, `transformers`, `numpy`, `backoff`

---

## Usage

### PromptFunction Class

```python
from POP import PromptFunction

pf = PromptFunction(
    sys_prompt="You are a helpful AI assistant.",
    prompt="Write a short poem about <<<topic>>>."
)

result = pf.execute(topic="space travel")
print(result)
```

---

### Improving Prompts

```python
improved_prompt = pf._improve_prompt()
print(improved_prompt)
```

---

### Function Schema & Code Generation

```python
# 1. Generate a JSON function schema
schema = pf.generate_schema(description="Multiply two integers and return the product.")

# 2. Generate actual Python code
code = pf.generate_code(schema)
print(code)
```

---

### Embeddings

```python
from POP.Embedder import Embedder

embedder = Embedder(use_api="openai")
vectors = embedder.get_embedding(["Hello world", "POP is awesome!"])
print(vectors.shape)  # (2, embedding_dim)
```

---

### Web Snapshot Utility

```python
from POP import get_text_snapshot

content = get_text_snapshot("https://example.com", image_caption=True)
print(content[:500])
```

---

## Example

```python
from POP import PromptFunction

pf = PromptFunction(
    prompt="Draw a simple ASCII art of <<<object>>>."
)

print(pf.execute(object="a cat"))
print(pf.execute(object="a rocket"))
```

Sample Output:

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

## Future Plans

* Complete support for local PyTorch and Deepseek clients
* Prompt chaining and workflow composition
* Automated prompt testing framework
* Extended multimodal support (image + text prompts)

---

## Contributing

1. Fork the repo
2. Create a feature branch
3. Submit a PR with clear commit messages and examples/tests

---
