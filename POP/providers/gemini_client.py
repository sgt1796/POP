"""Gemini API client implementation.

This module provides a :class:`LLMClient` implementation for Google
Gemini models.  It wraps the ``google.genai`` client so that
responses conform to the expected OpenAI‑like structure used by
:class:`pop.prompt_function.PromptFunction`.

If the ``google-genai`` library is unavailable, importing
this module will not error, but instantiating :class:`GeminiClient`
will raise an :class:`ImportError`.
"""

from typing import List, Dict, Any, Optional
from os import getenv

from .llm_client import LLMClient


try:
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore
except Exception:
    genai = None  # type: ignore
    types = None  # type: ignore


class GeminiClient(LLMClient):
    """Client for Google's Gemini models."""

    def __init__(self, model: str = "gemini-2.5-flash") -> None:
        if genai is None or types is None:
            raise ImportError(
                "google-genai package is not installed. Install it to use GeminiClient."
            )
        # Authenticate with the API key
        self.client = genai.Client(api_key=getenv("GEMINI_API_KEY"))
        self.model_name = model

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> Any:
        model_name = model or self.model_name

        # Extract system instruction and collate user/assistant content
        system_instruction: Optional[str] = None
        user_contents: List[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system" and system_instruction is None:
                system_instruction = content
            else:
                user_contents.append(content)

        # Prepare multimodal contents.  Gemini accepts a list of
        # strings/images; embed images if passed via kwargs.
        contents: List[Any] = []
        images = kwargs.pop("images", None)
        if images:
            try:
                from PIL import Image  # type: ignore
                import base64
                from io import BytesIO
            except Exception:
                raise ImportError("PIL and base64 are required for image support in GeminiClient.")
            for img in images:
                # Accept PIL.Image, base64 string or URL
                if isinstance(img, Image.Image):
                    contents.append(img)
                elif isinstance(img, str):
                    try:
                        # Assume base64 encoded image
                        img_data = base64.b64decode(img)
                        image = Image.open(BytesIO(img_data))  # type: ignore
                        contents.append(image)
                    except Exception:
                        # Fallback to URL
                        contents.append(img)
        # Add concatenated text as the last element
        if user_contents:
            contents.append("\n".join(user_contents))

        # Configure generation
        gen_config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction,
        )

        try:
            response = self.client.models.generate_content(
                model=model_name,
                contents=contents,
                config=gen_config,
            )
        except Exception as exc:
            raise RuntimeError(f"Gemini chat_completion error: {exc}") from exc

        # Wrap the Gemini response into an OpenAI‑like structure
        class FakeMessage:
            def __init__(self, content: str):
                self.content = content
                self.tool_calls = None

        class FakeChoice:
            def __init__(self, message: FakeMessage):
                self.message = message

        class FakeResponse:
            def __init__(self, text: str):
                self.choices = [FakeChoice(FakeMessage(text))]

        return FakeResponse(response.text or "")


__all__ = ["GeminiClient"]
