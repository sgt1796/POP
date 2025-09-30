from abc import ABC, abstractmethod
from xml.parsers.expat import model
from pydantic import BaseModel
from openai import OpenAI
from google import genai
from google.genai import types
from os import getenv
import requests



##############################################
# LLM Client Interface and Implementations
##############################################

class LLMClient(ABC):
    """
    Abstract Base Class for LLM Clients.
    """
    @abstractmethod
    def chat_completion(self, messages: list, model: str, temperature: float, **kwargs) -> str:
        pass

 
class OpenAIClient(LLMClient):
    """
    OpenAI API Client implementation using the new openai.OpenAI client.
    """
    # This fix the issue of httpx 0.28 causing by removal of 'proxies' parameter
    # %pip install openai==1.55.3 httpx==0.27.2 --force-reinstall --quiet
    def __init__(self):
        # Instantiate a new OpenAI client with the API key.   
        self.client = OpenAI(api_key=getenv("OPENAI_API_KEY"))

    def chat_completion(self, messages: list, model: str, temperature: float = 0.7, **kwargs):
        """
        Supports text and optional image inputs for multimodal completion.
        Pass images as `images=[...]` in kwargs; can be base64 or URLs.
        """
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
    def chat_completion(self, messages: list, model: str, temperature: float, **kwargs) -> str:
        # Implement your local PyTorch LLM call here.
        return "Local PyTorch LLM response (stub)"

class DeepseekClient(LLMClient):
    """
    Deepseek API client.
    """
    def __init__(self):
        self.client = OpenAI(api_key=getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

    def chat_completion(self, messages: list, model: str, temperature: float, **kwargs) -> str:
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
            raise RuntimeError(f"OpenAI chat_completion error: {e}")

        return response

class GeminiClient(LLMClient):
    """
    GCP Gemini API client.
    """
    def __init__(self, model="gemini-2.5-flash"):
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
            from PIL import Image
            import base64
            from io import BytesIO

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

    """
    def __init__(self):
        self.client = OpenAI(base_url="https://ark.cn-beijing.volces.com/api/v3",
                             api_key=getenv('DOUBAO_API_KEY'))

    def chat_completion(self, messages: list, model: str, temperature: float = 0.7, **kwargs):
        # Doubao payload (OpenAI-like)
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