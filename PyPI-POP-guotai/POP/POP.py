import re
import json
import requests
from dotenv import load_dotenv
from os import getenv, path
from abc import ABC, abstractmethod
from pydantic import BaseModel
from openai import OpenAI
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()
default_model = {
    "OpenAIClient": "gpt-4o-mini",
    "GeminiClient": "gemini-2.5-flash",
}

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
    (Placeholder implementation)
    """
    def chat_completion(self, messages: list, model: str, temperature: float, **kwargs) -> str:
        # Implement your Deepseek API call here.
        return "Deepseek LLM response (stub)"

class GeminiClient(LLMClient):
    """
    GCP Gemini API client.
    (Placeholder implementation)
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
    
##############################################
# PromptFunction Class
##############################################

class PromptFunction:
    """
    A class representing a reusable prompt function.
    """
    def __init__(self, 
                 sys_prompt: str = "", 
                 prompt: str = "",
                 client: LLMClient = None):
        """
        Initializes a new prompt function.
        
        Args:
            prompt (str): The base prompt template.
            sys_prompt (str): The system prompt for additional context.
            client (LLMClient): An instance of an LLM client. Defaults to OpenAIClient.
        """
        self.prompt = prompt
        self.temperature = 0.7
        self.sys_prompt = sys_prompt
        self.placeholders = self._get_place_holder()
        self.client = client if client is not None else OpenAIClient()
        self.last_response = None
        self.default_model_name = default_model[self.client.__class__.__name__]

    def execute(self, *args, **kwargs) -> str:
        """
        Executes the prompt function with dynamic argument injection.
        
        Special keys in kwargs:
            - model: Model name (default "gpt-4o-mini")
            - sys: Additional system instructions.
            - fmt: Response format/schema.
            - tools: List of function tools to use (for function calling).
            - temp: Temperature.
            - ADD_BEFORE: Text to prepend.
            - ADD_AFTER: Text to append.
        
        Args:
            *args: Positional arguments to add to the prompt.
            **kwargs: Keyword arguments for placeholder replacement or extra context.
        
        Returns:
            str: The LLM-generated response.
        """
        model = kwargs.pop("model", self.default_model_name)
        system_extra = kwargs.pop("sys", "")
        fmt = kwargs.pop("fmt", None)
        tools = kwargs.pop("tools", None)
        temp = kwargs.pop("temp", self.temperature)
        images = kwargs.pop("images", None)

        # Prepare the prompt with dynamic injections.
        formatted_prompt = self._prepare_prompt(*args, **kwargs)

        # Build the message payload.
        system_message = {
            "role": "system",
            "content": (
                f"You are a general-purpose helpful assistant that is responsible for executing Prompt Functions, you will receive instructions and return as ordered. "
                f"Since your return is expected to be read by code most of the time, DO NOT wrap your returns in '```' tags unless user explicitly asks for markdown or similar format. "
                f"Base system prompt:\n{self.sys_prompt}\n\n"
                f"Additional instructions:\n{system_extra}"
            )
        }
        user_message = {"role": "user", "content": f"<if no user message, check system prompt.> {formatted_prompt}"}
        messages = [system_message, user_message]

        # Call the LLM client.
        raw_response = self.client.chat_completion(
            messages=messages, 
            model=model, 
            temperature=temp, 
            response_format=fmt,
            tools=tools,
            images=images
        )

        # Save entire response
        self.last_response = raw_response

        # default to message content
        reply_content = raw_response.choices[0].message.content

        # if it is a function call, extract the arguments
        if raw_response.choices[0].message.tool_calls:
            tool_call = raw_response.choices[0].message.tool_calls[0]
            reply_content = tool_call.function.arguments

        return reply_content

    def _prepare_prompt(self, *args, **kwargs) -> str:
        """
        Prepares the prompt by injecting dynamic arguments into placeholders.
        Special keys: "ADD_BEFORE" and "ADD_AFTER".
        """
        before = kwargs.pop("ADD_BEFORE", "")
        after = kwargs.pop("ADD_AFTER", "")

        # Start with the base prompt. If empty, fallback to sys_prompt or join args.
        prompt = self.prompt
        if not prompt:
            if self.sys_prompt: # In the case of sys_prompt provided, construct prompt from user input
                prompt = "User instruction:"
                if args:
                    prompt += "\n" + "\n".join(args)
                if kwargs:
                    prompt += "\n" + "\n".join(f"{k}: {v}" for k, v in kwargs.items())
            else:
                raise ValueError("No prompt or system prompt provided.")

        # First pass: Replace placeholders that we detected in the prompt (not system prompt).
        for placeholder in self.placeholders:
            if placeholder in kwargs:
                prompt = prompt.replace(f"<<<{placeholder}>>>", str(kwargs.pop(placeholder)))
        
        # Second pass: Replace any remaining kwargs.
        for key, value in kwargs.items():
            prompt = prompt.replace(f"<<<{key}>>>", str(value))
        
        # Prepend or append additional text if provided.
        if before:
            prompt = before + "\n" + prompt
        if after:
            prompt = prompt + "\n" + after
        
        # Append any positional arguments at the end.
        if args:
            prompt = prompt + "\n" + "\n".join(args)

        return prompt

    def _improve_prompt(self, replace: bool = False, use_prompt: str = "fabric", instruction: str = None, user_instruction: str = None) -> str:
        """
        Improves the prompt using a metaprompt.
        
        Args:
            replace (bool): Whether to replace the existing prompt.
            use_prompt (str): Identifier for which metaprompt to use.
            instruction (str): Custom instruction if provided.
            user_instruction (str): Additional user instruction.
        
        Returns:
            str: The improved prompt.
        """
        if use_prompt == "fabric":
            # Dynamically build an absolute path based on where POP.py is located
            current_dir = path.dirname(path.abspath(__file__)) ## __file__ refers to the current file
            file = path.join(current_dir, "prompts", "fabric-improve_prompt.md")

            try:
                with open(file, 'r', encoding='utf-8') as f:
                    instruction = f.read()
            except FileNotFoundError:
                raise FileNotFoundError(f"File not found: {file}")

        meta_instruction = (
            f"\nAdditional instruction:\n{user_instruction}\n"
            "Ensure that original placeholders (<<<placeholder>>>) are preserved in the improved prompt and placed in a clear position."
            "do not use any '<<<'  or '>>>' in the improved prompt other than the original placeholder, and you have to show the placehold in the exact same order and amount of times as in the original prompt."
        )
        
        improved_prompt = self.execute(ADD_BEFORE=meta_instruction, 
                                       model="gpt-4o", 
                                       sys=f"You are asked to improve the above 'Base system prompt' using the following instruction:\n{instruction}")
        
        if use_prompt == "fabric":
            improved_prompt = improved_prompt.split("# OUTPUT\n\n")[-1]
        
        if replace:
            self.sys_prompt = improved_prompt
        return improved_prompt

    def set_temperature(self, temperature: float) -> None:
        """
        Sets the temperature for the LLM.
        """
        self.temperature = temperature

    def save(self, file_path: str) -> None:
        """
        Saves the prompt to a file.
        """
        with open(file_path, "w", encoding='utf-8') as file:
            file.write(self.prompt)

    def _get_place_holder(self) -> list:
        """
        Extracts placeholders (of the format <<<placeholder>>>) from the prompt or system prompt.
        """
        target_text = self.prompt if self.prompt else self.sys_prompt
        if not target_text:
            print("No prompt or system prompt provided.")
            return []
        placeholders = re.findall(r"<<<(.*?)>>>", target_text)
        if placeholders:
            print("Placeholders found:", placeholders)
        else:
            print("No placeholders found.")
        return placeholders

    def generate_schema(self, 
                        description: str = None,
                        meta_prompt: str = None,
                        meta_schema: dict = None,
                        model: str = "gpt-4o-mini",
                        save: bool = True
                       ) -> dict:
        """
        Instance method to generate a function schema from a natural language description
        using the PromptFunction's prompt if no explicit description is given.
        Also stores the generated schema to functions/ by default.

        Args:
            description (str): What the function should do.
            meta_prompt (str): Optionally override the meta prompt.
            meta_schema (dict): Optionally override the meta schema.
            model (str): Which model to use.
            save (bool): whether to store to functions/ directory

        Returns:
            dict: A valid OpenAI function tool schema.
        """

        # fallback to self.prompt if no description
        if not description:
            if self.prompt:
                description = self.prompt
            else:
                raise ValueError(
                    "Description or instance prompt must be provided to generate a function schema."
                )
        print(description)
        # fallback meta prompt from file
        if meta_prompt is None:
            try:
                meta_prompt = PromptFunction.load_prompt(
                    "prompts/openai-function_description_generator.md"
                )
            except FileNotFoundError:
                raise FileNotFoundError(
                    "Meta prompt file 'prompts/openai-function_description_generator.md' not found. "
                    "Either place it there or pass meta_prompt manually."
                )

        # fallback meta schema from file
        if meta_schema is None:
            try:
                meta_schema = json.loads(
                    PromptFunction.load_prompt(
                        "prompts/openai-function_description_generator.fmt"
                    )
                )
            except FileNotFoundError:
                raise FileNotFoundError(
                    "Meta schema file 'prompts/openai-function_description_generator.fmt' not found. "
                    "Either place it there or pass meta_schema manually."
                )

        completion = self.client.chat_completion(
            model=model,
            temperature=0.02,
            response_format=meta_schema,
            messages=[
                {
                    "role": "system",
                    "content": PromptFunction.load_prompt("prompts/openai-function_description_generator.md"),
                },
                {
                    "role": "user",
                    "content": "Description:\n" + description,
                },
            ],
        )

        parsed_schema = json.loads(completion.choices[0].message.content)

        # store to disk if requested
        if save:
            import os
            os.makedirs("functions", exist_ok=True)
            
            function_name = parsed_schema["name"]
            safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", function_name)
            file_path = os.path.join("functions", f"{safe_name}.json")
            
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(parsed_schema, f, indent=2)
            print(f"[generate_schema] Function schema saved to {file_path}")

        return parsed_schema

    def generate_code(self, 
                    schema: dict | str = None,
                    model: str = "gpt-4o",
                    save: bool = True) -> str:
        """
        Generate actual Python code for a function, given its meta schema.

        Args:
            schema (dict): The function schema (from generate_schema).
            model (str): LLM model name.
            save (bool): Whether to save the code to a .py file.

        Returns:
            str: The generated Python code as string.
        """
        if not schema:
            raise ValueError("You must provide a function schema to generate code.")
        # if schema is a string, try to parse it as JSON
        if isinstance(schema, str):
            try:
                schema = json.loads(schema)
            except json.JSONDecodeError:
                raise ValueError("Provided schema string is not valid JSON.")
        
        description = schema.get("description", "")
        function_name = schema.get("name", "generated_function")
        parameters = schema.get("parameters", {}).get("properties", {})

        # create a user instruction
        param_list = "\n".join(
            f"{param}: {info['description']}" 
            for param, info in parameters.items()
        )

        code_prompt = (
            f"Please write a Python function named `{function_name}`. "
            f"The function should do the following: {description}\n"
            f"Parameters:\n{param_list}\n"
            "Provide the full working code, including type annotations and docstring."
        )

        messages = [
            {"role": "system", "content": "You are a senior Python developer who writes clean and robust code."},
            {"role": "user", "content": code_prompt}
        ]

        completion = self.client.chat_completion(
            messages=messages,
            model=model,
            temperature=0.1
        )
        
        code = completion.choices[0].message.content

        # optionally store it
        if save:
            import os
            os.makedirs("functions", exist_ok=True)
            safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", function_name)
            file_path = os.path.join("functions", f"{safe_name}.py")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)
            print(f"[generate_code] Python code saved to {file_path}")

        return code


    @staticmethod
    def load_prompt(file: str) -> str:
        """
        Loads a prompt from a file.
        """
        with open(file, 'r', encoding='utf-8') as f:
            return f.read()
        

##############################################
# Utility Function
##############################################

def get_text_snapshot(web_url: str, 
                      use_api_key: bool = True, 
                      return_format: str = "default", 
                      timeout: int = 0, 
                      target_selector: list = None, 
                      wait_for_selector: list = None, 
                      exclude_selector: list = None, 
                      remove_image: bool = False, 
                      links_at_end: bool = False, 
                      images_at_end: bool = False, 
                      json_response: bool = False, 
                      image_caption: bool = False,
                      cookie: str = None) -> str:
    """
    Fetch a text snapshot of the webpage using r.jina.ai.
    """
    target_selector = target_selector or []
    wait_for_selector = wait_for_selector or []
    exclude_selector = exclude_selector or []

    headers = {}
    api_key = 'Bearer ' + getenv("JINAAI_API_KEY") if use_api_key else None

    header_values = {
        "Authorization": api_key,
        "X-Return-Format": None if return_format == "default" else return_format,
        "X-Timeout": timeout if timeout > 0 else None,
        "X-Target-Selector": ",".join(target_selector) if target_selector else None,
        "X-Wait-For-Selector": ",".join(wait_for_selector) if wait_for_selector else None,
        "X-Remove-Selector": ",".join(exclude_selector) if exclude_selector else None,
        "X-Retain-Images": "none" if remove_image else None,
        "X-With-Links-Summary": "true" if links_at_end else None,
        "X-With-Images-Summary": "true" if images_at_end else None,
        "Accept": "application/json" if json_response else None,
        "X-With-Generated-Alt": "true" if image_caption else None,
        "X-Set-Cookie": cookie if cookie else None
    }

    for key, value in header_values.items():
        if value is not None:
            headers[key] = value

    try:
        api_url = f"https://r.jina.ai/{web_url}"
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        return f"Error fetching text snapshot: {e}"
