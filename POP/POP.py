import re
import json
import requests
from dotenv import load_dotenv
from os import getenv, path
from .LLMClient import LLMClient, OpenAIClient, GeminiClient, DeepseekClient, LocalPyTorchClient, DoubaoClient, OllamaClient

# Load environment variables
load_dotenv()
default_model = {
    "OpenAIClient": "gpt-5-nano",
    "GeminiClient": "gemini-2.5-flash",
    "DeepseekClient": "deepseek-chat",
    "DoubaoClient": "doubao-seed-1-6-flash-250715",
    "OllamaClient": "mistral:7b",
}
client_map = {
            "openai": OpenAIClient,
            "gemini": GeminiClient,
            "local": LocalPyTorchClient,
            "deepseek": DeepseekClient,
            "doubao": DoubaoClient,
            "ollama": OllamaClient
        }

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
                 client: LLMClient | str = None):
        """
        Initializes a new prompt function.
        
        Args:
            prompt (str): The base prompt template.
            sys_prompt (str): The system prompt for additional context.
            client (LLMClient | str): An instance of an LLM client or a string identifier. Defaults to OpenAIClient. ("openai", "gemini", "local", "deepseek")
        """
        self.prompt = prompt
        self.sys_prompt = sys_prompt
        self.placeholders = self._get_place_holder()
        self.client = None
        
        if isinstance(client, LLMClient):
            self.client = client
        elif isinstance(client, str):
            self.client = client_map.get(client, None)
            if self.client:
                self.client = self.client() # instantiate LLMClient if user passed a string

        # gpt-5/mini/nano only supports temperature 1
        self.temperature = 1 if self.client.__class__.__name__ == "OpenAIClient" and default_model[self.client.__class__.__name__] in ["gpt-5-nano", "gpt-5-mini", "gpt-5"] else 0.0
        self.last_response = None
        self.default_model_name = default_model[self.client.__class__.__name__]
        print(f"[PromptFunction] Using client: {self.client.__class__.__name__}, default model: {self.default_model_name}")

    def execute(self, *args, **kwargs) -> str:
        """
        Executes the prompt function with dynamic argument injection.
        
        Special keys in kwargs:
            - model: Model name (defaults to self.default_model_name).
            - sys: Additional system instructions.
            - fmt: Response format/schema.
            - tools: List of function tools to use (for function calling).
            - tool_choice
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
        tool_choice = kwargs.pop("tool_choice", None)
        if tools and not tool_choice:
            tool_choice = "auto"

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

        try:
            # Call the LLM client. Always include tool_choice when tools are provided.
            call_kwargs = {
                "messages": messages,
                "model": model,
                "temperature": temp,
                "response_format": fmt,
                "tools": tools,
                "images": images,
                # Always include tool_choice key so callers can assert on it
                "tool_choice": tool_choice,
            }

            raw_response = self.client.chat_completion(**call_kwargs)
        except Exception as e:
            verbose = True
            if verbose:
                print(f"Error occurred while executing prompt function: {e}\nparameters:\nmodel: {model}\ntemperature: {temp}\nprompt: {formatted_prompt}\nsys: {system_extra}\nformat: {fmt}\ntools: {tools}\nimages: {images}")
            # raise RuntimeError(f"Error occurred while executing prompt function: {e}\n")
            print(f"Error occurred while executing prompt function: {e}\n")
            return ""

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
                if kwargs:
                    prompt += "\n" + "\n".join(f"{k}: {v}" for k, v in kwargs.items())
            else:
                raise ValueError("No prompt or system prompt provided.")
                
        # Append any positional arguments
        if args:
            prompt = prompt + "\n" + "\n".join(args)

        # Replace placeholders detected in the prompt (not system prompt).
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
                                       model="gpt-5", 
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
                        model: str = "gpt-5-mini",
                        save: bool = True
                       ) -> dict:
        """
        Instance method to generate a function schema from a natural language description
        using the PromptFunction's prompt if no explicit description is given.
        Also stores the generated schema to functions/ by default.

        Args:
            description (str): What the function should do.
            meta_prompt (str): Optionally override the meta prompt, path to file.
            meta_schema (dict): Optionally override the meta schema, path to file.
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
                    "prompts/openai-json_schema_generator.md"
                )
            except FileNotFoundError:
                raise FileNotFoundError(
                    "Meta prompt file 'prompts/openai-json_schema_generator.md' not found. "
                    "Either place it there or pass meta_prompt manually."
                )
        else:
            meta_prompt = PromptFunction.load_prompt(meta_prompt)

        # fallback meta schema from file
        if meta_schema is None:
            pass
        else:   
            with open(meta_schema, "r", encoding="utf-8") as f:
                meta_schema = json.load(f)

        completion = self.client.chat_completion(
            model=model,
            response_format=meta_schema,
            temperature=self.temperature,
            messages=[
                {
                    "role": "system",
                    "content": meta_prompt,
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
            os.makedirs("schemas", exist_ok=True)
            
            prompts_name = parsed_schema["name"] if "name" in parsed_schema else "generated_schema"
            safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", prompts_name)
            file_path = os.path.join("schemas", f"{safe_name}.json")

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(parsed_schema, f, indent=2)
            print(f"[generate_schema] Function schema saved to {file_path}")

        return parsed_schema

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
