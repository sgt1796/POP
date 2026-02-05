"""PromptFunction class for reusable prompts.

The :class:`PromptFunction` encapsulates a system prompt, a base
prompt template and the logic to execute that prompt against any
registered LLM provider.  It provides features such as dynamic
placeholder substitution, prompt improvement via a meta prompt,
function schema generation and prompt saving.

This implementation mirrors the original POP ``PromptFunction`` but
delegates provider instantiation to the central registry defined in
``pop.api_registry`` and stores meta prompts in ``pop/prompts/``.
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass
from os import getenv, path
from typing import List, Dict, Any, Optional, Union

from .providers.llm_client import LLMClient
from .api_registry import get_client
from .models import DEFAULT_MODEL


class PromptFunction:
    """Represent a reusable prompt function."""

    def __init__(
        self,
        sys_prompt: str = "",
        prompt: str = "",
        client: Union[LLMClient, str, None] = None,
    ) -> None:
        """Initialize a new prompt function.

        Parameters
        ----------
        sys_prompt : str
            The system prompt that provides high‑level instructions to the LLM.
        prompt : str
            The base prompt template.  Placeholders of the form
            ``<<<name>>>`` will be replaced with values passed to
            :meth:`execute`.
        client : LLMClient | str | None
            An instance of an LLM client or a provider identifier.  If
            omitted, the default provider is ``openai``.
        """
        self.prompt: str = prompt
        self.sys_prompt: str = sys_prompt
        self.placeholders: List[str] = self._get_place_holder()
        self.client: LLMClient

        # Instantiate the client based on the type of ``client``
        if isinstance(client, LLMClient):
            self.client = client
        else:
            provider_name = client or "openai"
            self.client = get_client(provider_name)  # type: ignore[assignment]
            if self.client is None:
                raise ValueError(f"Unknown provider: {provider_name}")

        # Choose a default model based on the client class name
        self.default_model_name: str = DEFAULT_MODEL.get(self.client.__class__.__name__, "")
        # gpt-5/mini/nano only supports temperature 1 (legacy from POP)
        if (
            self.client.__class__.__name__ == "OpenAIClient"
            and self.default_model_name in ["gpt-5-nano", "gpt-5-mini", "gpt-5"]
        ):
            self.temperature: float = 1.0
        else:
            self.temperature = 0.0
        self.last_response: Any = None
        # Provide some debug output so users know what client and model are in use
        print(
            f"[PromptFunction] Using client: {self.client.__class__.__name__}, using model: {self.client.model_name}"
        )

    def execute(self, *args: str, **kwargs: Any) -> str:
        """Execute the prompt with dynamic argument injection.

        Parameters
        ----------
        *args : str
            Positional strings appended to the prompt.
        **kwargs : Any
            Keyword arguments for placeholder replacement or extra
            context.  Special keys include:

            * ``model`` – override the default model name.
            * ``sys`` – additional system instructions.
            * ``fmt`` – response format (pydantic model or JSON schema).
            * ``tools`` – list of function tools for tool calling.
            * ``tool_choice`` – specific tool to call (default ``"auto"`` if
              ``tools`` is provided).
            * ``temp`` – override the temperature.
            * ``images`` – list of image URLs or base64 strings.
            * ``ADD_BEFORE`` – text prepended to the prompt.
            * ``ADD_AFTER`` – text appended to the prompt.

        Returns
        -------
        str
            The LLM‑generated response.  If the provider returns a
            function call, its arguments are returned instead.
        """
        # Pop recognised special keys
        model = kwargs.pop("model", self.default_model_name)
        system_extra = kwargs.pop("sys", "")
        fmt = kwargs.pop("fmt", None)
        tools = kwargs.pop("tools", None)
        temp = kwargs.pop("temp", self.temperature)
        images = kwargs.pop("images", None)
        tool_choice = kwargs.pop("tool_choice", None)
        if tools and not tool_choice:
            tool_choice = "auto"

        # Prepare the prompt with dynamic injections
        formatted_prompt = self._prepare_prompt(*args, **kwargs)

        # Build the message payload
        system_message = {
            "role": "system",
            "content": (
                "You are a general‑purpose helpful assistant that is responsible for executing Prompt Functions, you will receive instructions and return as ordered. "
                "Since your return is expected to be read by code most of the time, DO NOT wrap your returns in '```' tags unless user explicitly asks for markdown or similar format. "
                f"Base system prompt:\n{self.sys_prompt}\n\n"
                f"Additional instructions:\n{system_extra}"
            ),
        }
        user_message = {"role": "user", "content": f"<if no user message, check system prompt.> {formatted_prompt}"}
        messages = [system_message, user_message]

        # Assemble call parameters
        call_kwargs: Dict[str, Any] = {
            "messages": messages,
            "model": model,
            "temperature": temp,
        }
        if fmt is not None:
            call_kwargs["response_format"] = fmt
        if tools is not None:
            call_kwargs["tools"] = tools
            call_kwargs["tool_choice"] = tool_choice
        if images is not None:
            call_kwargs["images"] = images

        # Execute the call
        try:
            raw_response = self.client.chat_completion(**call_kwargs)
        except Exception as exc:
            # Print verbose diagnostics
            print(
                f"Error occurred while executing prompt function: {exc}\nparameters:\n"
                f"model: {model}\ntemperature: {temp}\nprompt: {formatted_prompt}\nsys: {system_extra}\n"
                f"format: {fmt}\ntools: {tools}\nimages: {images}"
            )
            return ""
        # Save entire response for later inspection
        self.last_response = raw_response

        # Extract the reply content.  If it's a function call, extract the
        # arguments instead of the raw content
        reply_content = ""
        try:
            first_choice = raw_response.choices[0]
            message = first_choice.message
            if getattr(message, "tool_calls", None):
                tool_call = message.tool_calls[0]
                reply_content = tool_call.function.arguments  # type: ignore[attr-defined]
            else:
                reply_content = message.content
        except Exception:
            # Fallback: attempt to coerce response to string
            reply_content = str(raw_response)
        return reply_content

    def _prepare_prompt(self, *args: str, **kwargs: Any) -> str:
        """Prepare the prompt by injecting dynamic arguments.

        Replacement occurs in three passes: 1) base prompt or system
        prompt; 2) positional arguments appended; 3) replace
        placeholders and remaining keyword arguments; 4) prepend or
        append additional text.
        """
        before = kwargs.pop("ADD_BEFORE", "")
        after = kwargs.pop("ADD_AFTER", "")
        # Determine starting prompt
        prompt = self.prompt
        if not prompt:
            if self.sys_prompt:
                prompt = "User instruction:"
                # When building from system prompt, encode kwargs into lines
                if kwargs:
                    prompt += "\n" + "\n".join(f"{k}: {v}" for k, v in kwargs.items())
            else:
                raise ValueError("No prompt or system prompt provided.")
        # Append positional arguments
        if args:
            prompt = prompt + "\n" + "\n".join(args)
        # First pass: replace placeholders defined in the original template
        for placeholder in self.placeholders:
            if placeholder in kwargs:
                prompt = prompt.replace(f"<<<{placeholder}>>>", str(kwargs.pop(placeholder)))
        # Second pass: replace any remaining kwargs by exact key match
        for key, value in list(kwargs.items()):
            prompt = prompt.replace(f"<<<{key}>>>", str(value))
        # Prepend and append additional text
        if before:
            prompt = before + "\n" + prompt
        if after:
            prompt = prompt + "\n" + after
        return prompt

    def _get_place_holder(self) -> List[str]:
        """Extract placeholders from the prompt or system prompt."""
        target_text = self.prompt if self.prompt else self.sys_prompt
        if not target_text:
            return []
        placeholders = re.findall(r"<<<(.*?)>>>", target_text)
        if placeholders:
            print("Placeholders found:", placeholders)
        return placeholders

    def improve_prompt(
        self,
        replace: bool = False,
        use_prompt: str = "fabric",
        instruction: Optional[str] = None,
        user_instruction: Optional[str] = None,
    ) -> str:
        """Improve the prompt using a meta prompt.

        Parameters
        ----------
        replace : bool, optional
            If True, replace the existing system prompt with the improved
            version (default False).
        use_prompt : str, optional
            Identifier for which meta prompt to use (currently only
            ``"fabric"`` is supported).
        instruction : str, optional
            Override the meta prompt instructions by providing a full
            meta prompt directly.
        user_instruction : str, optional
            Additional instructions from the user.
        Returns
        -------
        str
            The improved prompt.
        """
        if use_prompt == "fabric":
            # Determine path to the meta prompt file relative to this file
            current_dir = path.dirname(path.abspath(__file__))
            file_path = path.join(current_dir, "prompts", "fabric-improve_prompt.md")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    instruction = f.read()
            except FileNotFoundError:
                raise FileNotFoundError(f"File not found: {file_path}")
        # Compose meta instruction that preserves placeholders
        meta_instruction = (
            f"\nAdditional instruction:\n{user_instruction}\n"
            "Ensure that original placeholders (<<<placeholder>>>) are preserved in the improved prompt and placed in a clear position."
            "do not use any '<<<'  or '>>>' in the improved prompt other than the original placeholder, and you have to show the placehold in the exact same order and amount of times as in the original prompt."
        )
        # Execute the meta prompt via the model
        improved_prompt = self.execute(
            ADD_BEFORE=meta_instruction,
            model="gpt-5",
            sys=(
                "You are asked to improve the above 'Base system prompt' using the following instruction:\n"
                + (instruction or "")
            ),
        )
        if use_prompt == "fabric":
            # Extract only the part after the '# OUTPUT' marker
            if "# OUTPUT" in improved_prompt:
                improved_prompt = improved_prompt.split("# OUTPUT", 1)[-1].lstrip("\n")
        if replace:
            self.sys_prompt = improved_prompt
        return improved_prompt

    def generate_schema(
        self,
        description: Optional[str] = None,
        meta_prompt: Optional[str] = None,
        meta_schema: Optional[dict] = None,
        model: str = "gpt-5-mini",
        save: bool = True,
    ) -> dict:
        """Generate a function schema from a natural language description.

        The schema is generated by calling the underlying model with a
        meta prompt that instructs it to produce a JSON schema.  This
        mirrors the behaviour of the original POP implementation.
        """
        # Fallback to the instance prompt if no description provided
        if not description:
            if self.prompt:
                description = self.prompt
            else:
                raise ValueError(
                    "Description or instance prompt must be provided to generate a function schema."
                )
        # Load meta prompt from file if necessary
        if meta_prompt is None:
            try:
                meta_prompt = PromptFunction.load_prompt("prompts/openai-json_schema_generator.md")
            except FileNotFoundError:
                raise FileNotFoundError(
                    "Meta prompt file 'prompts/openai-json_schema_generator.md' not found. "
                    "Either place it there or pass meta_prompt manually."
                )
        else:
            meta_prompt = PromptFunction.load_prompt(meta_prompt)
        # Load meta schema from file if provided
        if meta_schema is not None and not isinstance(meta_schema, dict):
            # assume string path to JSON file
            with open(meta_schema, "r", encoding="utf-8") as f:
                meta_schema = json.load(f)
        # Prepare messages
        messages = [
            {"role": "system", "content": meta_prompt},
            {"role": "user", "content": "Description:\n" + description},
        ]
        # Execute call using the chosen model; response_format holds the meta_schema
        response = self.client.chat_completion(
            messages=messages,
            model=model,
            temperature=self.temperature,
            response_format=meta_schema,
        )
        # Parse JSON from the model's response
        try:
            content = response.choices[0].message.content
        except Exception:
            content = str(response)
        parsed_schema = json.loads(content)
        # Optionally save to a file under schemas/
        if save:
            import os
            os.makedirs("schemas", exist_ok=True)
            prompts_name = parsed_schema.get("name", "generated_schema")
            safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", prompts_name)
            file_path = os.path.join("schemas", f"{safe_name}.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(parsed_schema, f, indent=2)
            print(f"[generate_schema] Function schema saved to {file_path}")
        return parsed_schema

    @staticmethod
    def load_prompt(file: str) -> str:
        """Load a prompt from a file.

        The ``file`` parameter may be an absolute or relative path.  If
        it is relative, it is resolved relative to this module's
        directory to ensure that files under ``pop/prompts`` are found.
        """
        # Determine absolute path
        if not path.isabs(file):
            current_dir = path.dirname(path.abspath(__file__))
            file = path.join(current_dir, file)
        with open(file, "r", encoding="utf-8") as f:
            return f.read()

    def set_temperature(self, temperature: float) -> None:
        """Set the sampling temperature for the next execution."""
        self.temperature = temperature

    def save(self, file_path: str) -> None:
        """Save the base prompt to a file."""
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.prompt)


__all__ = ["PromptFunction"]
