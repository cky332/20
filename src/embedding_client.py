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

# Monkey-patch httpx to force proxy usage, since google-genai SDK ignores proxy env vars.
# This patches BOTH sync and async httpx clients.
_httpx_sync_original_init = httpx.Client.__init__
_httpx_async_original_init = httpx.AsyncClient.__init__


def _get_proxy():
    """Get proxy URL from environment variables."""
    return (os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
            or os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY"))


def _httpx_sync_patched_init(self, *args, **kwargs):
    if "proxy" not in kwargs:
        proxy = _get_proxy()
        if proxy:
            kwargs["proxy"] = proxy
    _httpx_sync_original_init(self, *args, **kwargs)


def _httpx_async_patched_init(self, *args, **kwargs):
    if "proxy" not in kwargs:
        proxy = _get_proxy()
        if proxy:
            kwargs["proxy"] = proxy
    _httpx_async_original_init(self, *args, **kwargs)


httpx.Client.__init__ = _httpx_sync_patched_init
httpx.AsyncClient.__init__ = _httpx_async_patched_init


class QuotaExhaustedError(Exception):
    """Raised when the daily API quota is exhausted."""
    pass


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
        proxy = _get_proxy()
        if proxy:
            print(f"  [EmbeddingClient] Using proxy: {proxy}")
        else:
            print("  [EmbeddingClient] WARNING: No proxy configured. "
                  "If you cannot reach Google APIs directly, set environment variable:\n"
                  "    export https_proxy=http://your-proxy:port\n"
                  "    export http_proxy=http://your-proxy:port")
        self.client = genai.Client(api_key=self.api_key)
        self._last_call_time = 0

    def _rate_limit(self):
        """Enforce rate limiting between API calls."""
        elapsed = time.time() - self._last_call_time
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)
        self._last_call_time = time.time()

    def _call_with_retry(self, contents, output_dimensionality=None):
        """Call the embedding API with retry logic.

        Handles rate limit (429) errors by waiting the suggested retry delay.
        For daily quota exhaustion, raises QuotaExhaustedError so callers
        can save progress and exit gracefully.

        Args:
            contents: Content to embed.
            output_dimensionality: Optional output dimension (Matryoshka truncation).
                Supported values: 3072 (default), 1536, 768.
        """
        import re

        config = None
        if output_dimensionality is not None:
            config = types.EmbedContentConfig(
                output_dimensionality=output_dimensionality
            )
        quota_retries = 0
        max_quota_retries = 3  # only retry quota errors a few times
        for attempt in range(MAX_RETRIES + max_quota_retries):
            try:
                self._rate_limit()
                result = self.client.models.embed_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )
                return np.array(result.embeddings[0].values)
            except Exception as e:
                err_str = str(e)
                is_quota = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str

                if is_quota:
                    quota_retries += 1
                    # Check if it's a daily quota (PerDay) vs short-term rate limit
                    is_daily = "PerDay" in err_str or "per day" in err_str.lower()

                    if is_daily and quota_retries >= 2:
                        # Daily quota truly exhausted — don't keep retrying
                        raise QuotaExhaustedError(
                            f"Daily API quota exhausted (1000 requests/day). "
                            f"Progress has been saved. Options:\n"
                            f"  1. Wait until tomorrow (quota resets at midnight PT)\n"
                            f"  2. Use an API key from a different Google Cloud project\n"
                            f"  3. Upgrade to paid tier at https://aistudio.google.com/\n"
                            f"Then re-run: python -m experiments.exp1_typographic_basic"
                        )

                    # Extract retry delay from error if available
                    delay_match = re.search(r'retry in ([\d.]+)s', err_str, re.IGNORECASE)
                    if delay_match:
                        wait_time = float(delay_match.group(1)) + 2.0
                    else:
                        wait_time = min(15 * (2 ** quota_retries), 120)

                    print(f"  Rate/quota limit hit (attempt {quota_retries}/{max_quota_retries}). "
                          f"Waiting {wait_time:.0f}s...")
                    time.sleep(wait_time)
                    continue

                if attempt >= MAX_RETRIES - 1:
                    raise
                wait_time = 2 ** (attempt + 1)
                print(f"  API call failed (attempt {attempt + 1}): {e}")
                print(f"  Retrying in {wait_time}s...")
                time.sleep(wait_time)

    def embed_text(self, text: str, output_dimensionality=None) -> np.ndarray:
        """Get embedding for a text string.

        Args:
            text: Input text to embed.
            output_dimensionality: Optional output dimension for Matryoshka truncation.

        Returns:
            Numpy array of embedding values.
        """
        return self._call_with_retry([text], output_dimensionality=output_dimensionality)

    def embed_image(self, image_bytes: bytes, mime_type: str = "image/png",
                    output_dimensionality=None) -> np.ndarray:
        """Get embedding for an image.

        Args:
            image_bytes: Raw image bytes.
            mime_type: MIME type of the image.
            output_dimensionality: Optional output dimension for Matryoshka truncation.

        Returns:
            Numpy array of embedding values.
        """
        contents = [
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        ]
        return self._call_with_retry(contents, output_dimensionality=output_dimensionality)

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
