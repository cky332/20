"""Gemini Embedding 2 API client wrapper."""

import os
import time
import numpy as np
import httpx
from google import genai
from google.genai import types

import sys
sys.path.insert(0, ".")
from config import GOOGLE_API_KEY, MODEL_NAME, RATE_LIMIT_DELAY, MAX_RETRIES


class GeminiEmbeddingClient:
    """Wrapper around Gemini Embedding 2 API with rate limiting and retries."""

    def __init__(self, api_key=None, model_name=None):
        self.api_key = api_key or GOOGLE_API_KEY
        self.model_name = model_name or MODEL_NAME
        if not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY not set. Export it as an environment variable: "
                "export GOOGLE_API_KEY='your-key'"
            )
        proxy_url = os.environ.get("https_proxy") or os.environ.get("http_proxy")
        if proxy_url:
            http_client = httpx.Client(proxy=proxy_url)
            self.client = genai.Client(api_key=self.api_key, http_options={"client": http_client})
        else:
            self.client = genai.Client(api_key=self.api_key)
        self._last_call_time = 0

    def _rate_limit(self):
        """Enforce rate limiting between API calls."""
        elapsed = time.time() - self._last_call_time
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)
        self._last_call_time = time.time()

    def _call_with_retry(self, contents):
        """Call the embedding API with retry logic."""
        for attempt in range(MAX_RETRIES):
            try:
                self._rate_limit()
                result = self.client.models.embed_content(
                    model=self.model_name,
                    contents=contents,
                )
                return np.array(result.embeddings[0].values)
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                wait_time = 2 ** (attempt + 1)
                print(f"  API call failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                print(f"  Retrying in {wait_time}s...")
                time.sleep(wait_time)

    def embed_text(self, text: str) -> np.ndarray:
        """Get embedding for a text string.

        Args:
            text: Input text to embed.

        Returns:
            Numpy array of embedding values.
        """
        return self._call_with_retry([text])

    def embed_image(self, image_bytes: bytes, mime_type: str = "image/png") -> np.ndarray:
        """Get embedding for an image.

        Args:
            image_bytes: Raw image bytes.
            mime_type: MIME type of the image.

        Returns:
            Numpy array of embedding values.
        """
        contents = [
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        ]
        return self._call_with_retry(contents)

    def embed_multimodal(self, text: str, image_bytes: bytes,
                         mime_type: str = "image/png") -> np.ndarray:
        """Get a single aggregated embedding for text + image.

        Args:
            text: Text input.
            image_bytes: Raw image bytes.
            mime_type: MIME type of the image.

        Returns:
            Numpy array of embedding values.
        """
        contents = [
            types.Content(
                parts=[
                    types.Part(text=text),
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                ]
            )
        ]
        return self._call_with_retry(contents)

    def embed_texts_batch(self, texts: list) -> list:
        """Get embeddings for multiple texts.

        Args:
            texts: List of text strings.

        Returns:
            List of numpy arrays.
        """
        embeddings = []
        for text in texts:
            emb = self.embed_text(text)
            embeddings.append(emb)
        return embeddings

    def embed_images_batch(self, images: list, mime_type: str = "image/png") -> list:
        """Get embeddings for multiple images.

        Args:
            images: List of (image_bytes,) or (image_bytes, mime_type) tuples.
            mime_type: Default MIME type.

        Returns:
            List of numpy arrays.
        """
        embeddings = []
        for img in images:
            if isinstance(img, tuple):
                emb = self.embed_image(img[0], img[1] if len(img) > 1 else mime_type)
            else:
                emb = self.embed_image(img, mime_type)
            embeddings.append(emb)
        return embeddings
