"""
HTTP proxy utilities.

This module acts as a placeholder for HTTP proxy configuration.  In
pi-ai the proxy module centralises configuration for upstream proxies
and TLS settings.  Here we expose a simple helper that returns a
requests session configured with environment proxy variables.
"""

import requests

def get_session_with_proxy() -> requests.Session:
    """Create a requests Session that respects HTTP(S)_PROXY environment variables."""
    session = requests.Session()
    session.trust_env = True
    return session
