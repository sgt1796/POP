from abc import ABC, abstractmethod
from xml.parsers.expat import model  # unused but kept for consistency
from pydantic import BaseModel
from os import getenv
import requests

# Note: The real POP repository uses the ``openai`` and ``google.genai``
# libraries to access external LLM services.  Those packages are not
# available in this environment, so the client implementations here
# serve as placeholders.  They preserve the API surface but will
# raise at runtime if invoked without the required third‑party
# dependencies.  If you wish to use remote LLMs, ensure the
# corresponding packages are installed and API keys are set in your
# environment.

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore
try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None  # type: ignore
    types = None  # type: ignore


##############################################
# LLM Client Interface and Implementations
##############################################

class LLMClient(ABC):
    """
    Abstract Base Class for LLM Clients.
    """
    @abstractmethod
    def chat_completion(self, messages: list, model: str, temperature: float, **kwargs):
        pass


class OpenAIClient(LLMClient):
    """
    OpenAI API Client implementation.  Requires the ``openai`` package.
    """
    def __init__(self):
        if OpenAI is None:
            raise ImportError(
                "openai package is not installed. Install it to use OpenAIClient."
            )
        # Instantiate a new OpenAI client with the API key.   
        self.client = OpenAI(api_key=getenv("OPENAI_API_KEY"))

    def chat_completion(self, messages: list, model: str, temperature: float = 0.7, **kwargs):
        request_payload = {
            "model": model,
            "messages": [],
            "temperature": temperature
        }

        # Optional images
        images = kwargs.pop("images", None)

        # Build OpenAI-style messages
        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "user")

            # If images are provided, attach them to the last user message
            if images and role == "user":
                multi_content = [{"type": "text", "text": content}]
                for img in images:
                    if isinstance(img, str) and img.startswith("http"):
                        multi_content.append({"type": "image_url", "image_url": {"url": img}})
                    else:
                        multi_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})
                content = multi_content
            request_payload["messages"].append({"role": role, "content": content})

        # Handle response_format (JSON schema)
        fmt = kwargs.get("response_format", None)
        if fmt:
            if isinstance(fmt, BaseModel):
                request_payload["response_format"] = fmt
            else:
                request_payload["response_format"] = {"type": "json_schema", "json_schema": fmt}

        # Handle function tools
        tools = kwargs.get("tools", None)
        if tools:
            request_payload["tools"] = [{"type": "function", "function": tool} for tool in tools]
            request_payload["tool_choice"] = "auto"

        # Temporary patch for models not supporting system roles
        if model == "o1-mini" and request_payload["messages"] and request_payload["messages"][0]["role"] == "system":
            request_payload["messages"][0]["role"] = "user"

        # Execute request
        try:
            response = self.client.chat.completions.create(**request_payload)
        except Exception as e:
            raise RuntimeError(f"OpenAI chat_completion error: {e}")
        return response


class LocalPyTorchClient(LLMClient):
    """
    Local PyTorch-based LLM client.
    (Placeholder implementation)
    """
    def chat_completion(self, messages: list, model: str, temperature: float, **kwargs):
        return "Local PyTorch LLM response (stub)"


class DeepseekClient(LLMClient):
    """
    Deepseek API client.  Requires ``openai`` with a Deepseek API base URL.
    """
    def __init__(self):
        if OpenAI is None:
            raise ImportError(
                "openai package is not installed. Install it to use DeepseekClient."
            )
        self.client = OpenAI(api_key=getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

    def chat_completion(self, messages: list, model: str, temperature: float, **kwargs):
        request_payload = {
            "model": model,
            "messages": [],
            "temperature": temperature
        }

        # Optional images
        images = kwargs.pop("images", None)
        if images:
            raise NotImplementedError("DeepseekClient does not support images yet.")

        # Build OpenAI-style messages
        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "user")
            request_payload["messages"].append({"role": role, "content": content})

        # Execute request
        try:
            response = self.client.chat.completions.create(**request_payload)
        except Exception as e:
            raise RuntimeError(f"Deepseek chat_completion error: {e}")
        return response


class GeminiClient(LLMClient):
    """
    GCP Gemini API client.  Requires the ``google.generativeai`` library.
    """
    def __init__(self, model="gemini-2.5-flash"):
        if genai is None or types is None:
            raise ImportError(
                "google generativeai package is not installed. Install it to use GeminiClient."
            )
        self.client = genai.Client(api_key=getenv("GEMINI_API_KEY"))
        self.model_name = model

    def chat_completion(self, messages: list, model: str = None, temperature: float = 0.7, **kwargs):
        model_name = model or self.model_name

        # Extract system instruction and user content
        system_instruction = None
        user_contents = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system" and system_instruction is None:
                system_instruction = content
            else:
                user_contents.append(content)

        # Prepare multimodal contents
        contents = []
        images = kwargs.pop("images", None)

        if images:
            try:
                from PIL import Image
                import base64
                from io import BytesIO
            except Exception:
                raise ImportError("PIL and base64 are required for image support in GeminiClient.")

            for img in images:
                # Accept base64 or PIL
                if isinstance(img, Image.Image):
                    contents.append(img)
                elif isinstance(img, str):
                    try:
                        # Base64 -> PIL Image
                        img_data = base64.b64decode(img)
                        image = Image.open(BytesIO(img_data))
                        contents.append(image)
                    except Exception:
                        # Assume URL string
                        contents.append(img)
        # Add text content last
        if user_contents:
            contents.append("\n".join(user_contents))

        # Config with system instruction and temperature
        gen_config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction
        )

        try:
            response = self.client.models.generate_content(
                model=model_name,
                contents=contents,
                config=gen_config
            )
        except Exception as e:
            raise RuntimeError(f"Gemini chat_completion error: {e}")

        # Wrap response in OpenAI-like structure for PromptFunction compatibility
        class FakeMessage:
            def __init__(self, content):
                self.content = content
                self.tool_calls = None

        class FakeChoice:
            def __init__(self, message):
                self.message = message

        class FakeResponse:
            def __init__(self, text):
                self.choices = [FakeChoice(FakeMessage(text))]

        return FakeResponse(response.text or "")


class DoubaoClient(LLMClient):
    """
    Doubao (Volcengine Ark) API client.
    This is a stub implementation because the required API library is
    not available in this environment.  If you need to use Doubao
    models, install the relevant client from Volcengine and update
    this class accordingly.
    """
    def __init__(self):
        if OpenAI is None:
            raise ImportError(
                "openai package is not installed. Install it to use DoubaoClient."
            )
        # The base URL for Doubao's API and API key must be provided via environment
        self.client = OpenAI(base_url="https://ark.cn-beijing.volces.com/api/v3",
                             api_key=getenv('DOUBAO_API_KEY'))

    def chat_completion(self, messages: list, model: str, temperature: float = 0.7, **kwargs):
        payload = {
            "model": model,
            "messages": [],
            "temperature": temperature,
        }
        images = kwargs.pop("images", None)
        
        ## If the images are passed as string URLs or base64, wrap them in a list
        if images and not isinstance(images, list):
            images = [images]

        # Pass through common knobs if present
        passthrough = [
            "top_p","max_tokens","stop","frequency_penalty","presence_penalty",
            "logprobs","top_logprobs","logit_bias","service_tier","thinking",
            "stream","stream_options",
        ]
        for k in passthrough:
            if k in kwargs and kwargs[k] is not None:
                payload[k] = kwargs[k]

        # Messages (attach images on user turns, same structure used for OpenAI)
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if images and role == "user":
                multi = [{"type": "text", "text": content}]
                for img in images:
                    if isinstance(img, str) and img.startswith("http"):
                        multi.append({"type": "image_url", "image_url": {"url": img}})
                    else:
                        multi.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})
                content = multi
            payload["messages"].append({"role": role, "content": content})

        # Tools (function calling)
        tools = kwargs.get("tools")
        if tools:
            raise NotImplementedError("DoubaoClient does not support tools yet.")
        try:
            response = self.client.chat.completions.create(**payload)
        except Exception as e:
            raise RuntimeError(f"Doubao chat_completion error: {e}")
        return response

class OllamaClient(LLMClient):
    """
    Ollama-compatible LLM client using the /api/generate endpoint.
    """

    def __init__(self, model="llama3:latest", base_url="http://localhost:11434", default_options=None, timeout=300):
        self.model = model
        self.base_url = base_url
        # sensible defaults for extraction tasks
        self.default_options = default_options or {
            "num_ctx": 8192,        # ↑ context window so long docs don't truncate
            "temperature": 0.02,     # low variance, more literal
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.05,
            "mirostat": 0           # disable mirostat for predictable outputs
        }
        self.timeout = timeout

    def chat_completion(self, messages: list, model: str = None, temperature: float = 0.7, **kwargs):
        # Extract a proper system string for /api/generate
        system_parts, user_assistant_lines = [], []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_parts.append(content)
            elif role == "assistant":
                user_assistant_lines.append(f"[Assistant]: {content}")
            else:
                user_assistant_lines.append(f"[User]: {content}")

        system = "\n".join(system_parts) if system_parts else None
        prompt = "\n".join(user_assistant_lines)

        # Merge caller-provided options with our defaults
        # Allow both a dict under 'ollama_options' and top-level knobs (max_tokens -> num_predict)
        caller_opts = kwargs.pop("ollama_options", {}) or {}
        options = {**self.default_options, **caller_opts}

        # keep legacy temperature kw in sync with options
        if temperature is not None:
            options["temperature"] = temperature

        payload = {
            "model": model or self.model,
            "prompt": prompt,
            "stream": False,
            # num_predict must be top level for /api/generate
            "num_predict": kwargs.get("max_tokens", 1024),
            "options": options,
        }

        # pass system separately (clearer than prepending to prompt)
        if system:
            payload["system"] = system

        # JSON mode / schema
        fmt = kwargs.get("response_format")
        if fmt:
            fmt = self._normalize_schema(fmt)
            payload["format"] = fmt  # raw JSON schema or "json"

        # optional stops and keep-alive
        if "stop" in kwargs and kwargs["stop"]:
            payload["stop"] = kwargs["stop"]
        if "keep_alive" in kwargs and kwargs["keep_alive"]:
            payload["keep_alive"] = kwargs["keep_alive"]

        try:
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout)
            response.raise_for_status()
            content = response.json().get("response", "")
            return self._wrap_response(content)
        except Exception as e:
            raise RuntimeError(f"OllamaClient error: {e}")
    # normalize: accept raw dict, {"schema": {...}}, JSON string, or path
    def _normalize_schema(self, fmt):
        import json, os
        if fmt is None:
            return None
        if isinstance(fmt, str):
            # try file path, else JSON string
            if os.path.exists(fmt):
                return json.load(open(fmt, "r", encoding="utf-8"))
            return json.loads(fmt)
        if isinstance(fmt, dict) and "schema" in fmt and isinstance(fmt["schema"], dict):
            return fmt["schema"]              # <-- unwrap OpenAI-style wrapper
        if isinstance(fmt, dict):
            return fmt                         # already a schema object
        raise TypeError("response_format must be a JSON schema dict, a JSON string, or a file path")

    def _wrap_response(self, content: str):
        class Message:
            def __init__(self, content): self.content = content; self.tool_calls = None

        class Choice:
            def __init__(self, message): self.message = message

        class Response:
            def __init__(self, content): self.choices = [Choice(Message(content))]

        return Response(content)