"""Experiment 1: Basic Typographic Attack on Gemini Embedding 2.

Tests whether adding misleading text to images shifts their embeddings
closer to the target text in Gemini Embedding 2's unified embedding space.

Metrics:
- Cosine similarity: clean image vs target text (baseline)
- Cosine similarity: attacked image vs target text (after attack)
- Similarity shift = attacked_sim - clean_sim
- Attack success rate (shift > 0)
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import RESULTS_DATA_DIR, RESULTS_FIGURES_DIR, ASSETS_DIR
from src.embedding_client import GeminiEmbeddingClient
from src.image_generator import SOURCE_IMAGES, TARGET_LABELS, image_to_bytes, bytes_to_image
from src.typographic_attack import add_typographic_text
from src.similarity import cosine_similarity, compute_attack_effectiveness, compute_attack_success_rate
from src.visualization import plot_attack_comparison, plot_similarity_heatmap


def run_experiment():
    """Run the basic typographic attack experiment."""
    print("=" * 60)
    print("Experiment 1: Basic Typographic Attack")
    print("=" * 60)

    client = GeminiEmbeddingClient()

    # Select a subset of source images and targets for manageable API calls
    source_keys = ["red_circle", "blue_rectangle", "green_triangle", "ocean_scene"]
    target_texts = ["fire truck", "iPod", "banana", "laptop computer"]

    results = []
    all_effectiveness = []

    for src_key in source_keys:
        print(f"\n--- Source: {src_key} ---")
        src_bytes = SOURCE_IMAGES[src_key]()

        # Save source image
        os.makedirs(ASSETS_DIR, exist_ok=True)
        with open(os.path.join(ASSETS_DIR, f"{src_key}.png"), "wb") as f:
            f.write(src_bytes)

        # Get clean image embedding
        print(f"  Embedding clean image...")
        clean_emb = client.embed_image(src_bytes)

        for target_text in target_texts:
            print(f"  Target: '{target_text}'")

            # Get target text embedding
            target_emb = client.embed_text(target_text)

            # Create attacked image (large, centered, black text)
            attacked_bytes = add_typographic_text(
                src_bytes, target_text,
                font_size=60, position="center", color=(0, 0, 0)
            )

            # Save attacked image
            attacked_name = f"{src_key}_attack_{target_text.replace(' ', '_')}.png"
            with open(os.path.join(ASSETS_DIR, attacked_name), "wb") as f:
                f.write(attacked_bytes)

            # Get attacked image embedding
            print(f"  Embedding attacked image...")
            attacked_emb = client.embed_image(attacked_bytes)

            # Compute similarities
            clean_sim = cosine_similarity(clean_emb, target_emb)
            attacked_sim = cosine_similarity(attacked_emb, target_emb)
            preservation_sim = cosine_similarity(clean_emb, attacked_emb)

            effectiveness = compute_attack_effectiveness(clean_sim, attacked_sim)

            result = {
                "source": src_key,
                "target": target_text,
                "clean_to_target": clean_sim,
                "attacked_to_target": attacked_sim,
                "clean_to_attacked": preservation_sim,
                **effectiveness,
            }
            results.append(result)
            all_effectiveness.append(effectiveness)

            print(f"    Clean→Target:    {clean_sim:.4f}")
            print(f"    Attacked→Target: {attacked_sim:.4f}")
            print(f"    Shift:           {effectiveness['similarity_shift']:+.4f}")
            print(f"    Clean→Attacked:  {preservation_sim:.4f}")

    # Compute overall success rate
    success = compute_attack_success_rate(all_effectiveness)
    print(f"\n{'=' * 60}")
    print(f"Overall Attack Success Rate: {success['success_rate']:.1%}")
    print(f"  Successful: {success['successful']}/{success['total']}")
    print(f"  Mean Shift: {success['mean_shift']:+.4f}")
    print(f"  Std Shift:  {success['std_shift']:.4f}")
    print(f"  Max Shift:  {success['max_shift']:+.4f}")
    print(f"  Min Shift:  {success['min_shift']:+.4f}")

    # Save results
    os.makedirs(RESULTS_DATA_DIR, exist_ok=True)
    output = {
        "experiment": "exp1_typographic_basic",
        "results": results,
        "summary": success,
    }
    output_path = os.path.join(RESULTS_DATA_DIR, "exp1_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Generate plots
    os.makedirs(RESULTS_FIGURES_DIR, exist_ok=True)

    plot_data = [
        {
            "label": f"{r['source']}→{r['target']}",
            "clean_similarity": r["clean_to_target"],
            "attacked_similarity": r["attacked_to_target"],
        }
        for r in results
    ]
    plot_attack_comparison(
        plot_data,
        title="Exp1: Typographic Attack — Similarity Shift",
        save_path=os.path.join(RESULTS_FIGURES_DIR, "exp1_attack_comparison.png"),
    )

    return output


if __name__ == "__main__":
    run_experiment()
