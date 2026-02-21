"""
Embedding utilities for POP.

This module implements a unified embedding interface capable of
fetching embeddings via third‑party APIs (JinaAI, OpenAI, Gemini) or via
a local PyTorch model.  It is largely derived from the original
POP project’s ``Embedder.py`` and can be used independently of
``PromptFunction``.

Example usage:

>>> from POP.embedder import Embedder
>>> embedder = Embedder(use_api='openai')
>>> vectors = embedder.get_embedding(["Hello, world!"])

The return value is a numpy array of shape (n_texts, embedding_dim).
"""

import numpy as np
import openai
import requests as HTTPRequests
from os import getenv
from backoff import on_exception, expo
from typing import List

# Maximum number of tokens permitted by the Jina segmenter
MAX_TOKENS = 8194
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

class Embedder:
    """
    A class supporting multiple embedding methods, including Jina API,
    OpenAI API, Gemini API (OpenAI-compatible), and local model embeddings via PyTorch.

    Parameters
    ----------
    model_name:
        Name of the model to use for embedding.  If ``None`` the default
        model for the selected API will be chosen.
    use_api:
        Which API to use for embedding.  Supported values are
        ``'jina'``, ``'openai'``, ``'gemini'`` and ``None`` (for local embedding).
    to_cuda:
        If ``True``, use GPU; otherwise use CPU for local embeddings.
    attn_implementation:
        Optional attention implementation to pass to the transformer
        when loading the local model.
    """

    def __init__(self, model_name: str = None, use_api: str = None,
                 to_cuda: bool = False, attn_implementation: str = None):
        self.use_api = use_api
        self.model_name = model_name
        self.to_cuda = to_cuda

        # API‑based embedding initialisation
        if self.use_api is not None:
            supported_apis = ['', 'jina', 'openai', 'gemini']
            if self.use_api not in supported_apis:
                raise ValueError(f"API type '{self.use_api}' not supported. Supported APIs: {supported_apis}")

            if self.use_api == '':
                # empty string falls back to OpenAI
                self.use_api = 'openai'

            if self.use_api == 'jina':
                # The Jina client requires an API key; nothing to initialise
                self.client = None
            elif self.use_api == 'openai':
                # Initialise OpenAI client
                self.client = openai.Client(api_key=getenv("OPENAI_API_KEY"))
            elif self.use_api == 'gemini':
                # Initialise OpenAI-compatible Gemini embeddings client
                self.client = openai.Client(
                    api_key=getenv("GEMINI_API_KEY"),
                    base_url=GEMINI_OPENAI_BASE_URL,
                )
        else:
            # Load PyTorch model for local embedding generation
            if not model_name:
                raise ValueError("Model name must be provided when using a local model.")
            self.attn_implementation = attn_implementation
            self._initialize_local_model()

    def _initialize_local_model(self) -> None:
        """Initialise the PyTorch model and tokenizer for local embedding generation."""
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "Local embedding requires optional dependency 'torch'. "
                "Install torch or use use_api='openai'/'jina'."
            ) from exc

        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError as exc:
            raise ImportError(
                "Local embedding requires optional dependency 'transformers'. "
                "Install transformers or use use_api='openai'/'jina'."
            ) from exc

        if self.attn_implementation:
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                attn_implementation=self.attn_implementation,
                torch_dtype=torch.float16,
            ).to('cuda' if self.to_cuda else 'cpu')
        else:
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            ).to('cuda' if self.to_cuda else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model.eval()

    def get_embedding(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Parameters
        ----------
        texts:
            A list of strings to embed.

        Returns
        -------
        numpy.ndarray
            Embeddings as a 2‑D array of shape (len(texts), embedding_dim).
        """
        if not isinstance(texts, list):
            raise ValueError("Input must be a list of strings.")

        if self.use_api:
            if self.use_api == 'jina':
                # set default model if not provided
                if not self.model_name:
                    self.model_name = "jina-embeddings-v3"
                return self._get_jina_embedding(texts)
            elif self.use_api == 'openai':
                if not self.model_name:
                    self.model_name = "text-embedding-3-small"
                return self._get_openai_embedding(texts)
            elif self.use_api == 'gemini':
                if not self.model_name:
                    self.model_name = "gemini-embedding-001"
                return self._get_gemini_embedding(texts)
            else:
                raise ValueError(f"API type '{self.use_api}' is not supported.")
        else:
            return self._get_torch_embedding(texts)

    @on_exception(expo, HTTPRequests.exceptions.RequestException, max_time=30)
    def _get_jina_embedding(self, texts: List[str]) -> np.ndarray:
        """Fetch embeddings from the Jina API.  Requires Jina API key in .env."""
        url = 'https://api.jina.ai/v1/embeddings'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f"Bearer {getenv('JINAAI_API_KEY')}"
        }
        data = {
            "model": self.model_name or "jina-embeddings-v3",
            "task": "text-matching",
            "dimensions": 1024,
            "late_chunking": False,
            "embedding_type": "float",
            "input": [text for text in texts],
        }
        response = HTTPRequests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            embeddings = response.json().get('data', [])
            embeddings_np = np.array([e['embedding'] for e in embeddings], dtype='f')
            return embeddings_np
        elif response.status_code == 429:
            raise HTTPRequests.exceptions.RequestException(
                f"Rate limit exceeded: {response.status_code}, {response.text}"
            )
        elif response.status_code == 400:
            # input too long; segment and average
            ebd = []
            for text in texts:
                chunks = self._Jina_segmenter(text, max_token=MAX_TOKENS)
                token_counts = [len(chunk) for chunk in chunks]
                chunk_embedding = self.get_embedding(chunks)
                weighted_avg = np.average(chunk_embedding, weights=token_counts, axis=0)
                ebd.append(weighted_avg)
            return np.array(ebd, dtype='f')
        else:
            raise Exception(f"Failed to get embedding from Jina API: {response.status_code}, {response.text}")

    @on_exception(expo, HTTPRequests.exceptions.RequestException, max_time=30)
    def _get_openai_embedding(self, texts: List[str]) -> np.ndarray:
        """Fetch embeddings from the OpenAI API and return them as a NumPy array."""
        batch_size = 2048
        if len(texts) > batch_size:
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self._get_openai_embedding(batch_texts)
                all_embeddings.append(batch_embeddings)
            return np.vstack(all_embeddings)
        texts = [text.replace("\n", " ") for text in texts]
        response = self.client.embeddings.create(input=texts, model=self.model_name)
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings, dtype='f')

    @on_exception(expo, HTTPRequests.exceptions.RequestException, max_time=30)
    def _get_gemini_embedding(self, texts: List[str]) -> np.ndarray:
        """Fetch embeddings from the Gemini API via OpenAI-compatible endpoint."""
        batch_size = 2048
        if len(texts) > batch_size:
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self._get_gemini_embedding(batch_texts)
                all_embeddings.append(batch_embeddings)
            return np.vstack(all_embeddings)
        texts = [text.replace("\n", " ") for text in texts]
        response = self.client.embeddings.create(input=texts, model=self.model_name)
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings, dtype='f')

    def _get_torch_embedding(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using a local PyTorch model."""
        try:
            import torch
            import torch.nn.functional as F
        except ImportError as exc:
            raise ImportError(
                "Local embedding requires optional dependency 'torch'. "
                "Install torch or use use_api='openai'/'jina'."
            ) from exc

        @torch.no_grad()
        def _encode(instance: 'Embedder', input_texts: List[str]) -> np.ndarray:
            batch_dict = instance.tokenizer(
                input_texts,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors='pt',
                return_attention_mask=True,
            ).to('cuda' if instance.to_cuda else 'cpu')
            outputs = instance.model(**batch_dict)
            attention_mask = batch_dict['attention_mask']
            hidden = outputs.last_hidden_state
            def _weighted_mean_pooling(hidden_states, mask):
                # compute weighted mean over tokens
                mask_ = mask * mask.cumsum(dim=1)
                s = (hidden_states * mask_.unsqueeze(-1).float()).sum(dim=1)
                d = mask_.sum(dim=1, keepdim=True).float()
                return s / d
            reps = _weighted_mean_pooling(hidden, attention_mask)
            embeddings = F.normalize(reps, p=2, dim=1).detach().cpu().numpy()
            return embeddings
        return _encode(self, texts)

    @on_exception(expo, HTTPRequests.exceptions.RequestException, max_time=30)
    def _Jina_segmenter(self, text: str, max_token: int) -> List[str]:
        """Segments text into chunks using Jina API.  (free but needs API key)"""
        url = 'https://segment.jina.ai/'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f"Bearer {getenv('JINAAI_API_KEY')}"
        }
        data = {
            "content": text,
            "return_tokens": True,
            "return_chunks": True,
            "max_chunk_length": max_token,
        }
        response = HTTPRequests.post(url, headers=headers, json=data)
        return response.json().get('chunks', [])
