"""Experiment 2: Cross-Modal Alignment Attack.

Tests whether typographic text can systematically pull image embeddings
toward arbitrary text embeddings in Gemini Embedding 2's unified space.

Includes 2D PCA visualization of embedding space movement.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import RESULTS_DATA_DIR, RESULTS_FIGURES_DIR, ASSETS_DIR
from src.embedding_client import GeminiEmbeddingClient
from src.image_generator import SOURCE_IMAGES
from src.typographic_attack import add_typographic_text
from src.similarity import cosine_similarity, cosine_distance
from src.visualization import plot_embedding_space_2d


def run_experiment():
    """Run the cross-modal alignment attack experiment."""
    print("=" * 60)
    print("Experiment 2: Cross-Modal Alignment Attack")
    print("=" * 60)

    client = GeminiEmbeddingClient()

    # Use semantically distant image-text pairs
    test_cases = [
        ("red_circle", "a photograph of a golden retriever dog"),
        ("blue_rectangle", "a delicious pepperoni pizza"),
        ("ocean_scene", "a modern laptop computer on a desk"),
        ("forest_scene", "a red fire truck with sirens"),
    ]

    all_embeddings = {}
    all_labels_groups = {}
    results = []

    for src_key, target_text in test_cases:
        print(f"\n--- {src_key} → '{target_text}' ---")
        src_bytes = SOURCE_IMAGES[src_key]()

        # Clean image embedding
        clean_emb = client.embed_image(src_bytes)
        all_embeddings[f"img:{src_key}"] = clean_emb
        all_labels_groups[f"img:{src_key}"] = "clean_image"

        # Target text embedding
        target_emb = client.embed_text(target_text)
        short_target = target_text[:25]
        all_embeddings[f"txt:{short_target}"] = target_emb
        all_labels_groups[f"txt:{short_target}"] = "target_text"

        # Source description text embedding (what the image "should" match)
        src_desc = {
            "red_circle": "a red circle shape",
            "blue_rectangle": "a blue rectangle shape",
            "ocean_scene": "an ocean with water and sky",
            "forest_scene": "a green forest with trees",
        }[src_key]
        src_text_emb = client.embed_text(src_desc)
        all_embeddings[f"txt:{src_desc[:20]}"] = src_text_emb
        all_labels_groups[f"txt:{src_desc[:20]}"] = "source_desc"

        # Attack with multiple intensities
        for font_size, intensity in [(36, "small"), (60, "medium"), (80, "large")]:
            attacked_bytes = add_typographic_text(
                src_bytes, target_text.split()[-1],  # Use key word
                font_size=font_size, position="center", color=(0, 0, 0)
            )
            attacked_emb = client.embed_image(attacked_bytes)
            emb_key = f"atk:{src_key}_{intensity}"
            all_embeddings[emb_key] = attacked_emb
            all_labels_groups[emb_key] = "attacked_image"

            # Measure distances
            dist_to_target = cosine_distance(attacked_emb, target_emb)
            dist_to_clean = cosine_distance(attacked_emb, clean_emb)
            sim_to_target = cosine_similarity(attacked_emb, target_emb)
            sim_to_clean = cosine_similarity(attacked_emb, clean_emb)
            baseline_sim = cosine_similarity(clean_emb, target_emb)

            result = {
                "source": src_key,
                "target": target_text,
                "intensity": intensity,
                "font_size": font_size,
                "sim_attacked_to_target": sim_to_target,
                "sim_attacked_to_clean": sim_to_clean,
                "sim_clean_to_target": baseline_sim,
                "shift_toward_target": sim_to_target - baseline_sim,
            }
            results.append(result)

            print(f"  [{intensity}] Attacked→Target: {sim_to_target:.4f} "
                  f"(baseline: {baseline_sim:.4f}, shift: {sim_to_target - baseline_sim:+.4f})")
            print(f"           Attacked→Clean:  {sim_to_clean:.4f}")

    # Save results
    os.makedirs(RESULTS_DATA_DIR, exist_ok=True)
    output = {
        "experiment": "exp2_cross_modal",
        "results": results,
    }
    with open(os.path.join(RESULTS_DATA_DIR, "exp2_results.json"), "w") as f:
        json.dump(output, f, indent=2)

    # Embedding space visualization
    os.makedirs(RESULTS_FIGURES_DIR, exist_ok=True)
    plot_embedding_space_2d(
        all_embeddings,
        labels_groups=all_labels_groups,
        title="Exp2: Cross-Modal Attack — Embedding Space (PCA 2D)",
        save_path=os.path.join(RESULTS_FIGURES_DIR, "exp2_embedding_space.png"),
    )

    print(f"\nResults saved to: {os.path.join(RESULTS_DATA_DIR, 'exp2_results.json')}")
    return output


if __name__ == "__main__":
    run_experiment()
