# Prompt Oriented Programming (POP)

**Reusable, mutable prompt functions for LLMs.** Inspired by object-oriented programming (OOP), POP treats prompts as first-class citizens—defining them like functions with inputs, outputs, and reusable logic. Execution is delegated to a language model (LLM), such as OpenAI’s GPT, enabling you to call, modify, and reuse prompts in a structured, programmatic way.

---

## Table of Contents
1. [Features](#features)  
2. [Installation](#installation)  
3. [Setup & Configuration](#setup--configuration)  
4. [Usage](#usage)  
   - [PromptFunction Class](#promptfunction-class)  
   - [Improving Prompts](#improving-prompts)  
5. [Example](#example)  
6. [Future Plans](#future-plans)  
7. [Contributing](#contributing)  

---

## Features
- **Prompt as a Function**: Declare a prompt once, then execute it any number of times with different inputs.  
- **Dynamic Placeholders**: Insert variables via placeholders (`<<<variable>>>`), allowing flexible, user-defined parameters.  
- **System Role Defaults**: POP stores the main prompt in the “system” role by default, letting you easily switch between roles or contexts.  
- **Prompt Enhancement**: Leverages patterns from [Fabric](https://github.com/fabric/fabric) (MIT licensed) to expand basic prompts into more detailed instructions via `pf._improve_prompt()`.

---

## Installation

1. **Clone or Download** this repository (or add it as a submodule in your project).
2. **Install via `pip`** (recommended):
   ```bash
   pip install -e .
   ```
   This uses the `setup.py` (and optionally `pyproject.toml`) so that `pop` can be globally recognized as a Python package.

For more details on packaging and distribution, see the [Installation Steps in the conversation](#).

---

## Setup & Configuration

### Environment Variables
Place a `.env` file in the same directory as `POP.py` or `Embedder.py`, containing API keys if you use OpenAI or other APIs. For example:

```ini
OPENAI_API_KEY=your_openai_api_key
JINAAI_API_KEY=your_jina_api_key  # Required only if you use Jina features
```

> **Note**: POP uses OpenAI by default. Make sure your `OPENAI_API_KEY` is valid, or configure a different LLM in your code.

---

## Usage

### PromptFunction Class
```python
from pop import PromptFunction

# Create a prompt function
pf = PromptFunction(
    sys_prompt="You are a helpful AI assistant.",
    prompt="""
    Please draw a simple ASCII art of <<<object>>>.
    """
)

# Execute with user input
art = pf.execute(object="a cat")
print(art)
```
- **Placeholders**: `<<<object>>>` is replaced by your chosen argument (`object="a cat"`).
- **Multiple Calls**: You can call `pf.execute(...)` repeatedly with different inputs.

### Improving Prompts
```python
expanded_prompt = pf._improve_prompt()
print(expanded_prompt)
```
- **`_improve_prompt()`**: Expands or refines your base prompt with more detailed instructions (inspired by Fabric’s approach).

---

## Example

```python
from pop import PromptFunction

# Define a prompt function
pf = PromptFunction(
    prompt="""
    Draw a simple ASCII art of an object described by: <<<user_input>>>
    """
)

# First Execution
result = pf.execute(user_input="A cat")
print("ASCII art of a cat:\n", result)

# Second Execution
result = pf.execute(user_input="An apple")
print("ASCII art of an apple:\n", result)
```

**Sample Output**:
```
ASCII art of a cat:
 /\_/\  
( o.o )
 > ^ <  

ASCII art of an apple:
    ,--./,-.
   / #      \
  |          |
   \        /
    `._,._,'
```

---

## Future Plans
- **Additional LLM Integrations**: Support local models or alternative APIs beyond OpenAI and Jina.
- **Enhanced Prompt Composition**: Build chains of PromptFunctions for complex workflows.
- **Prompt Testing Framework**: Automate checks to validate prompt outputs across multiple inputs.

---

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests:
1. **Fork** the repo
2. **Create a feature branch** for your changes
3. **Open a PR** once ready

Please include clear commit messages and relevant tests or examples.

---

Enjoy **Prompt Oriented Programming**! If you have questions or suggestions, don’t hesitate to reach out or file an issue.