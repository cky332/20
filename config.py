"""Configuration for Gemini Embedding 2 attack experiments."""

import os

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

MODEL_NAME = "gemini-embedding-2-preview"

EMBEDDING_DIM = 3072

OUTPUT_DIMS = [768, 1536, 3072]

RATE_LIMIT_DELAY = 1.0

MAX_RETRIES = 5

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
RESULTS_DATA_DIR = os.path.join(RESULTS_DIR, "data")
RESULTS_FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

IMAGE_SIZE = (512, 512)
