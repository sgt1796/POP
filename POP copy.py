import openai
import requests
from dotenv import load_dotenv
from os import getenv
from typing import Any 

## A class representing a reusable prompt function executed via GPT API (except jina reader API for get_text_snapshot).

load_dotenv()
client = openai.Client()
class PromptFunction:
    """
    A class representing a reusable prompt function in POP.
    """
    def __init__(self, prompt: str):
        """
        Initializes a new prompt function with a prompt template.
        
        Args:
            prompt (str): The base prompt template for the function.
        """
        self.prompt = prompt
        self.temperature = 0.7

    def execute(self, *args, **kwargs) -> str:
        """
        Executes the prompt function with dynamic argument injection.

        Args:
            *args: Positional arguments to add before, after, or inject into the prompt.
            **kwargs: Key-value arguments to replace placeholders in the prompt or provide additional context.

        Returns:
            str: The AI-generated response.
        """
        use_model = kwargs.pop("USE_MODEL", "gpt-4o-mini")
        system_instruction = kwargs.pop("SYSTEM_INSTRUCTION", None)
        response_format = kwargs.pop("RESPONSE_FORMAT", None)
        # Step 1: Inject user arguments into the prompt
        prompt = self._prepare_prompt(*args, **kwargs)

        # Step 2: Call AI with the prepared prompt
        return self._call_ai(prompt, use_model=use_model, system_instruction=system_instruction, response_format=response_format)

    def _prepare_prompt(self, *args, **kwargs) -> str:
        """
        Prepares the prompt by dynamically injecting arguments.

        Args:
            *args: Positional arguments to add to the prompt.
            **kwargs: Key-value arguments for placeholder replacement or context.

        Returns:
            str: The modified prompt ready for execution.
        """
        # Default injection modes: Before, After, or Inject placeholders
        before = kwargs.pop("ADD_BEFORE", None)
        after = kwargs.pop("ADD_AFTER", None)
        

        prompt = self.prompt
        ## If a placeholder is provided, replace it with the value
        for key, value in kwargs.items():
            prompt = prompt.replace(f"{{{key}}}", str(value))

        # Add "before" text if provided
        prompt = f"{before}\n{prompt}" if before else prompt
        # Add "after" text if provided
        prompt = f"{prompt}\n{after}" if after else prompt
        # Append positional arguments to the end
        if args:
            prompt = f"{prompt}\n" + "\n".join(args)

        return prompt
    
    def _call_ai(self, formatted_prompt: str = None, use_model: str = "gpt-4o-mini", response_format: dict = None, system_instruction = None) -> Any:
        """
        Internal method to call OpenAI API using the chat completion API.

        Args:
            formatted_prompt (str): The prepared prompt for execution.
        
        Returns:
            Any: The API response.
        """
        if not formatted_prompt:
            formatted_prompt = self.prompt
        system_instruction = "" if system_instruction is None else system_instruction
        # Define request payload
        request_payload = {
            "model": use_model,
            "messages": [
                {"role": "system", "content": f"You are a helpful assistant that perform tasks user asked to. Here's addition instruction if any: \n{system_instruction}"},
                {"role": "user", "content": formatted_prompt}
            ],
            "temperature": self.temperature
        }

        if response_format:
            request_payload["response_format"] = {
                "type": "json_schema",
                "json_schema": response_format
            }

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
        improved_prompt = self.execute(ADD_BEFORE=f"\naddition instruction:\n{instruction} \nyou must preserve the original placeholder in the improved prompt if there is one.\n\n"+instruction,
                                       USE_MODEL="gpt-4o", 
                                       SYSTEM_INSTRUCTION = system_instruction)
        
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

def load_prompt(file):
        with open(file, 'r') as f:
            return f.read()
        
def get_text_snapshot(web_url, use_api_key=True, return_format="default", timeout=0, target_selector=[], wait_for_selector=[], exclude_selector=[], remove_image=False, links_at_end=False, images_at_end=False, json_response=False, image_caption=False):
        """
        Fetch a text snapshot of the webpage using r.jina.ai.
        
        Args:
            web_url (str): The URL of the webpage to process.
            use_api_key (bool): Whether to use the API key for authorization. Default is True.
            return_format (str): The format in which to return the snapshot. Options are "default", "markdown", "html", "text", "screenshot", "pageshot". Default is "default".
            timeout (int): The number of seconds to wait for the server to send data before giving up. Default is 0 (no timeout).
            target_selector (list): A list of CSS selector to focus on more specific parts of the page. Useful when your desired content doesn't show under the default settings.
            wait_for_selector (list): A list of CSS selector to wait for specific elements to appear before returning. Useful when your desired content doesn't show under the default settings.
            exclude_selector (list): A list of CSS selector to remove the specified elements of the page. Useful when you want to exclude specific parts of the page like headers, footers, etc.
            remove_image (bool): Remove all images from the response.
            links_at_end (bool): A "Buttons & Links" section will be created at the end. This helps the downstream LLMs or web agents navigating the page or take further actions.
            images_at_end (bool): An "Images" section will be created at the end. This gives the downstream LLMs an overview of all visuals on the page, which may improve reasoning.
            json_response (bool): The response will be in JSON format, containing the URL, title, content, and timestamp (if available).
            image_caption (bool): Captions all images at the specified URL, adding 'Image [idx]: [caption]' as an alt tag for those without one. This allows downstream LLMs to interact with the images in activities such as reasoning and summarizing.
            
        Returns:
            str: The cleaned text content from the webpage, or an error message.
        """
        headers = {}
        if use_api_key:
            api_key = 'Bearer ' + getenv("JINAAI_API_KEY")
        else:
            print("No API key found, proceeding without it.")
            api_key = None
        
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
            if value is not None:  # Add header only if the value is not None
                headers[key] = value
        
        try:
            # Construct the API URL
            api_url = f"https://r.jina.ai/{web_url}"
            
            # Make a GET request to fetch the cleaned content
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)

            # Return the text content of the response
            return response.text
        except requests.exceptions.RequestException as e:
            return f"Error fetching text snapshot: {e}"