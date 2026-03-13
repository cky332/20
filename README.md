# Typographic Attacks on Gemini Embedding 2

Research project investigating the vulnerability of Google's Gemini Embedding 2 (the first natively multimodal embedding model) to typographic attacks.

## Overview

This project tests whether typographic attacks — adding misleading text overlays to images — can manipulate the embedding space of Gemini Embedding 2. While such attacks are well-studied on CLIP-based dual-encoder architectures, Gemini Embedding 2's single-Transformer native multimodal architecture has not been evaluated.

## Experiments

1. **Basic Typographic Attack** — Measures cosine similarity shifts when misleading text is added to images
2. **Cross-Modal Alignment Attack** — Tests whether typographic text can pull image embeddings toward arbitrary text embeddings
3. **Document Retrieval Poisoning** — Simulates RAG poisoning via typographic manipulation
4. **Parameter Study** — Systematic study of attack parameters (font size, position, color, repetition)

## Setup

```bash
pip install -r requirements.txt
export GOOGLE_API_KEY="your-api-key-here"
```

## Usage

```bash
# Run all experiments
python -m experiments.run_all

# Run individual experiments
python -m experiments.exp1_typographic_basic
python -m experiments.exp2_cross_modal
python -m experiments.exp3_retrieval_poison
python -m experiments.exp4_parameter_study
```

## Project Structure

```
├── config.py                     # Configuration
├── src/
│   ├── embedding_client.py       # Gemini API wrapper
│   ├── image_generator.py        # Programmatic test image generation
│   ├── typographic_attack.py     # Typographic attack implementation
│   ├── similarity.py             # Cosine similarity & metrics
│   └── visualization.py          # Result visualization
├── experiments/                  # Experiment scripts
├── results/                      # Output data and figures
├── report/                       # Technical report
└── assets/                       # Generated test images
```
