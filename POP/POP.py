import openai
import requests
from dotenv import load_dotenv
from os import getenv
import re
from pydantic import BaseModel


## A class representing a reusable prompt function executed via GPT API (except jina reader API for get_text_snapshot).

load_dotenv()
client = openai.Client()
class PromptFunction:
    """
    A class representing a reusable prompt function in POP.
    """
    def __init__(self, sys_prompt: str = "", prompt = None):
        """
        Initializes a new prompt function with a prompt template.
        **Updates: 
        1. Role of the default (without kwargs) prompt given while the pf creation will be set to "system" instead of "user".
        2. pf.execute("somthing to say") --> "somthing to say" will go to user's prompt.

        Args:
            prompt (str): The base prompt template for the function.
            sys_prompt (str): The system prompt for the function.
        """
        self.prompt = prompt if prompt else ""
        self.temperature = 0.7
        self.sys_prompt = sys_prompt
        self.placeholders = self._get_place_holder()

    def execute(self, *args, **kwargs) -> str:
        """
        Executes the prompt function with dynamic argument injection.

        Args:
            *args: Positional arguments to add before, after, or inject into the prompt.
            **kwargs: Key-value arguments to replace placeholders in the prompt or provide additional context.

        Returns:
            str: The AI-generated response.
        """
        model = kwargs.pop("model", "gpt-4o-mini")
        sys = kwargs.pop("sys", None)
        fmt = kwargs.pop("fmt", None)
        temp = kwargs.pop("temp", self.temperature)

        # Step 1: Inject user arguments into the prompt
        prompt = self._prepare_prompt(*args, **kwargs)

        # Step 2: Call AI with the prepared prompt
        return self._call_ai(prompt, model=model, sys=sys, fmt=fmt, temp=temp)

    def _prepare_prompt(self, *args, **kwargs) -> str:
        """
        Prepares the prompt by dynamically injecting arguments into the existing placeholders.
        This version explicitly uses the self.placeholders list.

        Args:
            *args: Positional arguments to be appended to the prompt.
            **kwargs: Keyword arguments for replacing placeholders or adding context. 
                    Special keys include "ADD_BEFORE" and "ADD_AFTER" for prepending or appending text.

        Returns:
            str: The fully formatted prompt.
        """
        # Extract special injection modes (if provided)
        before = kwargs.pop("ADD_BEFORE", None)
        after = kwargs.pop("ADD_AFTER", None)
        
        # Start with the base prompt
        prompt = self.prompt
        # If the prompt is empty, no kwargs provided, but the system prompt is not, use the positional argument as the prompt
        if not prompt and not kwargs and self.sys_prompt:
            # join them with a newline character
            prompt = f"{prompt}\n" + "\n".join(args)
        # If the prompt is empty, with kwargs and sys_prompt both provided, will print the kwargs dict as the user's prompt
        elif not prompt and kwargs and self.sys_prompt:
            prompt = f"{prompt}\n" + "\n".join([f"{k}: {v}" for k, v in kwargs.items()])

        # First pass: Replace only placeholders that exist in self.placeholders
        for placeholder in self.placeholders:
            if placeholder in kwargs:
                prompt = prompt.replace(f"<<<{placeholder}>>>", str(kwargs.pop(placeholder)))
            # Optionally: else warn or leave the placeholder as is

        # Second pass: For any remaining kwargs, attempt replacement 
        # (useful if there are dynamic replacements not originally detected)
        for key, value in kwargs.items():
            prompt = prompt.replace(f"<<<{key}>>>", str(value))
        
        # Prepend and append any additional text if provided
        if before:
            prompt = f"{before}\n{prompt}"
        if after:
            prompt = f"{prompt}\n{after}"
        
        # Append any positional arguments at the end of the prompt
        if args:
            prompt = f"{prompt}\n" + "\n".join(args)

        return prompt

    
    def _call_ai(self, formatted_prompt: str, model: str, sys: str, fmt: dict, temp: float) -> str:
        """
        Internal method to call OpenAI API using the chat completion API.

        Args:
            formatted_prompt (str): The prepared prompt for execution.
            model (str): The model name to use.
            sys (str): System instructions.
            fmt (dict): Response format/schema.
            temp (float): Sampling temperature.

        Returns:
            str: The raw response content.
        """
        sys = "" if sys is None else sys
        request_payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": f"You are a General-purposed helpful assistant unless otherwise instructed. User's prompt for system:\n{self.sys_prompt}\n\nAdditional instruction:\n{sys}\n"},
                {"role": "user", "content": formatted_prompt}
            ],
            "temperature": temp
        }

        if fmt:
            if isinstance(fmt, BaseModel):
                request_payload["response_format"] = fmt
            else:
                request_payload["response_format"] = {"type": "json_schema", "json_schema": fmt}
        
        ## Currently the o1-mini model not support the sys role, So if the model is o1-mini, we change the sys prompt to user prompt.
        # this is a temporary solution -----
        if model == 'o1-mini':
            request_payload["messages"][0]["role"] = "user"

            
        response = client.chat.completions.create(**request_payload)
        return response.choices[0].message.content

    
    def _improve_prompt(self, replace = False, use_prompt = "fabric", instruction = None, system_instruction = None) -> str:
        """
        Improving prompt using the improve_prompt pattern provided by fabric or an user provided metaprompt.

        Args:
            replace (bool): Whether to replace the existing prompt in addition to returning it.
            use_prompt (str): The metaprompt to use for improving the prompt.

        Returns:
            str: The improved prompt.
        """
        if use_prompt == "fabric":
            file = "prompts/fabric-improve_prompt.md"
        
        try:
            with open(file, 'r', encoding='utf-8') as f:
                instruction = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file}")
        
        # use instruction to improve prompt
        instruction = "" if instruction is None else instruction
        improved_prompt = self.execute(ADD_BEFORE=f"\naddition instruction:\n{instruction} \nyou must preserve the original placeholder (the placeholder is of the format <<<placeholder>>>) in the improved prompt if there is one.\n\n"+instruction,
                                       model="gpt-4o", 
                                       sys = system_instruction)
        
        improved_prompt = improved_prompt.split("# OUTPUT\n\n")[-1] if use_prompt == "fabric" else improved_prompt

        if replace:
            self.prompt = improved_prompt
        else:
            return improved_prompt
        
    def set_temperature(self, temperature: float) -> None:
        """
        Sets the temperature for the AI model.

        Args:
            temperature (float): The temperature value to set.
        """
        self.temperature = temperature

    def save(self, file_path: str) -> None:
        """
        Saves the prompt function to a file.

        Args:
            file_path (str): The file path to save the prompt function to.
        """
        with open(file_path, "w") as file:
            file.write(self.prompt)

    def _get_place_holder(self):
        """
        Get all the placeholders in the prompt. 

        Returns:
            list: A list of all the placeholders in the prompt.
        """
        if not self.prompt:
            print("No prompt found. Checking sys_prompt...")
            if not self.sys_prompt:
                print("No system prompt found. Please provide a prompt.")
                return []
            else:
                result = re.findall(r"<<<(.*?)>>>", self.sys_prompt)
                if not result:
                    print("No placeholders found in the system prompt.")
                    return []
              
                print("Placeholders:", '\n'.join(result))
                    
                return result
        result = re.findall(r"<<<(.*?)>>>", self.prompt)
        if not result:
            print("No placeholders found in the prompt.")
            return []
        print("Placeholders:", '\n'.join(result))
        return result
    
    def _load_prompt(file):
        with open(file, 'r') as f:
            return f.read()

