from openai import OpenAI, RateLimitError
from dotenv import load_dotenv
from os import getenv
import json
import requests
import backoff

load_dotenv()
client = OpenAI()

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

@backoff.on_exception(backoff.expo, RateLimitError, max_time=60)
def find_available_content(query_str):
    if _is_url(query_str):
        content = get_text_snapshot(query_str)
    else:
        content = query_str

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": load_prompt('./2024-11-19-content_finder.md')
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": content
                    }
                ]
            }
        ],
        temperature=0.0,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "categories_dict",
                "schema": {
                    "type": "object",
                    "properties": {
                        "categories": {
                            "description": "The categories of contents to be collected, in a dictionary format. The key is the category name, and the value is the url to contents.",
                            "type": "object"
                        }
                    },
                    "additionalProperties": False
                }
            }
        }
    )
    return completion

@backoff.on_exception(backoff.expo, RateLimitError, max_time=60)
def GPT_boolean(query_str, query):
    """
    Checks if the given URL's content matches the query description.

    Args:
        url (str): The URL to evaluate.
        query (str): The query description to validate against the URL.

    Returns:
        bool: True if the URL content matches the query description, False otherwise.
    """
    if _is_url(query_str):
        content = get_text_snapshot(query_str)
    else:
        content = query_str

    try:
        # OpenAI completion API call
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
            {"role": "system", "content": "Some times AI returns incorrect urls, You are a helpful assistant that evaluates that checks the webpage snap shot and determine if there actually stories (with title and urls) in that page. Return 'True' if the content is relevant to the query and 'False' otherwise. return false if page not found or story not found."},
            {"role": "user", "content": content},
            {"role": "user", "content": query}
            ],
            temperature=0.0,
            response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "GPT_boolean",
                "schema": {
                "type": "object",
                "properties": {
                    "decision": {
                    "description": "Whether or not the URL is related to the query.",
                    "type": "boolean"
                    }
                },
                "additionalProperties": False
                }
            }
            }
        )

        # Extract the model's response
        response = json.loads(completion.choices[0].message.content)
        decision = response.get("decision", None)

        # Return True or False based on the response
        if isinstance(decision, bool):
            return decision
        else:
            raise ValueError("Unexpected response from GPT model.")
    
    except Exception as e:
        print(f"Error occurred: {e}")
        return False

@backoff.on_exception(backoff.expo, RateLimitError, max_time=60)
def get_titles_and_urls(query_str):
    if _is_url(query_str):
        content = get_text_snapshot(query_str)
    else:
        content = query_str
        
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": load_prompt('./2024-11-19-get_title_and_url.md')
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": content
                    }
                ]
            }
        ],
        temperature=0.0,
        response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "titles_dict",
            "schema": {
                "type": "object",
                "properties": {
                    "titles_and_urls": {
                        "description": "A dictionary where keys are content titles and values are the corresponding URLs.",
                        "type": "object",
                        "additionalProperties": {
                            "type": "string"
                        }
                    },
                    "next_page": {
                        "description": "The URL of the next page to fetch titles and URLs from. Empty means no more pages.",
                        "type": "string"
                    }
                },
                "required": ["titles_and_urls", "next_page"],  # Ensure both keys are present
                "additionalProperties": False  # Disallow extra fields
            }
        }
    }
    )
    
    return completion

@backoff.on_exception(backoff.expo, RateLimitError, max_time=60)
def get_content(query_str):
    if _is_url(query_str):
        content = get_text_snapshot(query_str)
    else:
        content = query_str
        
    completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": load_prompt('./2024-11-19-get_content.md')
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": content
                        }
                    ]
                }
            ],
            temperature=0.0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "story_dict",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "description": "A dictionary containing the extracted story information.",
                                "type": "object",
                                "properties": {
                                    "title": {
                                        "description": "The title of the story.",
                                        "type": "string"
                                    },
                                    "author": {
                                        "description": "The author of the story, if available. Empty string if unknown.",
                                        "type": "string"
                                    },
                                    "content": {
                                        "description": "The main content of the story.",
                                        "type": "string"
                                    }
                                },
                                "required": ["title", "author", "content"],
                                "additionalProperties": False  # No extra fields allowed
                            },
                            "next_page": {
                                "description": "The URL of the next page to fetch more content from. Empty string means no more pages.",
                                "type": "string"
                            }
                        },
                        "required": ["content", "next_page"],  # Ensure these keys are always present
                        "additionalProperties": False  # Disallow extra fields in the top-level object
                    }
                }
    }
    )
    return completion

def _is_url(url):
    try:
        response = requests.get(url)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False