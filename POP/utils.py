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