"""Local PyTorch LLM client.

This client provides a minimal placeholder implementation of an
LLM client that runs locally using PyTorch.  Because actual model
weights are not included in this distribution, the client returns a
static response indicating that it is a stub.  You can extend this
class to load and run a local model (for example via HuggingFace
transformers) if desired.
"""

from typing import List, Dict, Any

from .llm_client import LLMClient


class LocalPyTorchClient(LLMClient):
    """Placeholder client for local PyTorch models."""

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> Any:
        # This stub simply returns a canned response.  Real
        # implementations should load a local model and generate a
        # response based on the messages and parameters.
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

        return FakeResponse("Local PyTorch LLM response (stub)")


__all__ = ["LocalPyTorchClient"]
