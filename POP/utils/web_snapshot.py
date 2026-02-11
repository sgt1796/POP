"""
Web snapshot utility.

This module wraps the ``get_text_snapshot`` function from the original
POP project.  It uses the r.jina.ai service to fetch text snapshots
of arbitrary web pages and supports various flags to control the
formatting of the returned content.
"""

import requests
from os import getenv
from typing import List, Optional

def get_text_snapshot(
    web_url: str,
    use_api_key: bool = True,
    return_format: str = "default",
    timeout: int = 0,
    target_selector: Optional[List[str]] = None,
    wait_for_selector: Optional[List[str]] = None,
    exclude_selector: Optional[List[str]] = None,
    remove_image: bool = False,
    links_at_end: bool = False,
    images_at_end: bool = False,
    json_response: bool = False,
    image_caption: bool = False,
    cookie: str = None,
) -> str:
    """Fetch a text snapshot of the webpage using r.jina.ai.

    Parameters
    ----------
    web_url:
        The URL of the page to snapshot.  This should include the
        protocol (http:// or https://) and any path or query parameters.
    use_api_key:
        Whether to send the JINAAI_API_KEY from the environment as an
        authorization header.
    return_format:
        The return format accepted by r.jina.ai.  Defaults to ``default``.
    timeout:
        Request timeout in seconds.  0 means no timeout header is sent.
    target_selector:
        A list of CSS selectors to target specific content within the page.
    wait_for_selector:
        A list of CSS selectors to wait for before capturing the snapshot.
    exclude_selector:
        A list of CSS selectors to exclude from the snapshot.
    remove_image:
        If ``True``, remove all images from the snapshot.
    links_at_end:
        If ``True``, append a links summary to the end of the snapshot.
    images_at_end:
        If ``True``, append an images summary to the end of the snapshot.
    json_response:
        If ``True``, request the snapshot as JSON rather than plain text.
    image_caption:
        If ``True``, include generated alt text for images.
    cookie:
        An optional cookie string to include in the request.

    Returns
    -------
    str
        The snapshot text, or an error message if the request fails.
    """
    target_selector = target_selector or []
    wait_for_selector = wait_for_selector or []
    exclude_selector = exclude_selector or []
    
    api_key = None
    if use_api_key and getenv("JINAAI_API_KEY"):
        api_key = 'Bearer ' + getenv("JINAAI_API_KEY", "")
    headers = {
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
        "X-Set-Cookie": cookie if cookie else None,
    }

    try:
        api_url = f"https://r.jina.ai/{web_url}"
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        return f"Error fetching text snapshot: {e}"
    

if __name__ == "__main__":
    # support url passed via command line for quick testing
    import sys
    if len(sys.argv) > 1:
        url = sys.argv[1]
        snapshot = get_text_snapshot(url, use_api_key=False)
        print(snapshot) 
