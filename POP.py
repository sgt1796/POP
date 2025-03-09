import re
import requests
from dotenv import load_dotenv
from os import getenv, path
from abc import ABC, abstractmethod
from pydantic import BaseModel
from openai import OpenAI

# Load environment variables
load_dotenv()

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

    def chat_completion(self, messages: list, model: str, temperature: float, **kwargs) -> str:
        request_payload = {
            "model": model,
            "messages": messages, 
            "temperature": temperature
        }

        ## process the response format
        fmt = kwargs.get("response_format", None)
        if fmt:
            if isinstance(fmt, BaseModel):
                request_payload["response_format"] = fmt
            else:
                request_payload["response_format"] = {"type": "json_schema", "json_schema": fmt}
         
        # Temporary workaround for models not supporting system messages.
        if model == 'o1-mini' and messages and messages[0].get("role") == "system":
            messages[0]["role"] = "user"
        
        response = self.client.chat.completions.create(**request_payload)
        return response.choices[0].message.content


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

class GCPGeminiClient(LLMClient):
    """
    GCP Gemini API client.
    (Placeholder implementation)
    """
    def chat_completion(self, messages: list, model: str, temperature: float, **kwargs) -> str:
        # Implement your GCP Gemini API call here.
        return "GCP Gemini LLM response (stub)"

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

    def execute(self, *args, **kwargs) -> str:
        """
        Executes the prompt function with dynamic argument injection.
        
        Special keys in kwargs:
            - model: Model name (default "gpt-4o-mini")
            - sys: Additional system instructions.
            - fmt: Response format/schema.
            - temp: Temperature.
            - ADD_BEFORE: Text to prepend.
            - ADD_AFTER: Text to append.
        
        Args:
            *args: Positional arguments to add to the prompt.
            **kwargs: Keyword arguments for placeholder replacement or extra context.
        
        Returns:
            str: The LLM-generated response.
        """
        model = kwargs.pop("model", "gpt-4o-mini")
        system_extra = kwargs.pop("sys", "")
        fmt = kwargs.pop("fmt", None)
        temp = kwargs.pop("temp", self.temperature)

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
        response = self.client.chat_completion(
            messages=messages, 
            model=model, 
            temperature=temp, 
            response_format=fmt
        )
        return response

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
                      image_caption: bool = False) -> str:
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
        "X-With-Generated-Alt": "true" if image_caption else None
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
