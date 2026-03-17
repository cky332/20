"""Experiment 7: Semantic Collision Attack on Gemini Embedding 2.

Tests whether typographic manipulation can make semantically different
inputs produce nearly identical embeddings — a "collision" in the
embedding space.

A successful collision means an attacker can craft an image that
is semantically about topic A but has an embedding nearly identical
to text about topic B, breaking the semantic integrity of the space.

Metrics:
- Maximum achievable cosine similarity between attacked image and target text
- Collision gap: 1.0 - max_similarity (how close to perfect collision)
- Progression: how similarity increases with attack intensity
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import RESULTS_DATA_DIR, RESULTS_FIGURES_DIR, ASSETS_DIR
from src.embedding_client import GeminiEmbeddingClient
from src.image_generator import SOURCE_IMAGES
from src.typographic_attack import add_typographic_text
from src.similarity import cosine_similarity
from src.statistics import compute_confidence_interval
from src.visualization import plot_collision_progression


# Collision pairs: (source_image, target_text, key_word)
# These are deliberately semantically distant pairs
COLLISION_PAIRS = [
    {
        "source": "ocean_scene",
        "target_text": "a laptop computer on a wooden desk",
        "keyword": "laptop computer",
        "source_desc": "ocean waves and sky",
    },
    {
        "source": "red_circle",
        "target_text": "a golden retriever dog in a park",
        "keyword": "golden retriever",
        "source_desc": "red circle on white",
    },
    {
        "source": "forest_scene",
        "target_text": "a pepperoni pizza fresh from the oven",
        "keyword": "pepperoni pizza",
        "source_desc": "green forest with trees",
    },
    {
        "source": "city_scene",
        "target_text": "a yellow banana fruit on a table",
        "keyword": "banana",
        "source_desc": "city skyline at night",
    },
]

# Progressive attack intensities
ATTACK_LEVELS = [
    {"name": "baseline", "font_size": 0, "text_type": "none",
     "repeat": 1, "bg_box": False},
    {"name": "keyword_small", "font_size": 24, "text_type": "keyword",
     "repeat": 1, "bg_box": False},
    {"name": "keyword_medium", "font_size": 48, "text_type": "keyword",
     "repeat": 1, "bg_box": False},
    {"name": "keyword_large", "font_size": 80, "text_type": "keyword",
     "repeat": 1, "bg_box": False},
    {"name": "full_text_medium", "font_size": 36, "text_type": "full",
     "repeat": 1, "bg_box": False},
    {"name": "full_text_large_bg", "font_size": 48, "text_type": "full",
     "repeat": 1, "bg_box": True},
    {"name": "full_scattered_3x", "font_size": 28, "text_type": "full",
     "repeat": 3, "bg_box": False},
]


def run_experiment():
    """Run the semantic collision attack experiment."""
    print("=" * 60)
    print("Experiment 7: Semantic Collision Attack")
    print("=" * 60)

    client = GeminiEmbeddingClient()
    os.makedirs(ASSETS_DIR, exist_ok=True)

    results = []
    plot_data = []

    for pair in COLLISION_PAIRS:
        src_key = pair["source"]
        target_text = pair["target_text"]
        keyword = pair["keyword"]

        print(f"\n{'=' * 50}")
        print(f"Collision pair: {src_key} → '{target_text}'")
        print(f"{'=' * 50}")

        pair_result = {
            "source": src_key,
            "target_text": target_text,
            "keyword": keyword,
            "levels": [],
        }
        pair_plot = {
            "pair_label": f"{src_key}→{keyword}",
            "similarities": [],
        }

        # Get source image and target text embedding
        src_bytes = SOURCE_IMAGES[src_key]()
        print("  Embedding target text...")
        target_emb = client.embed_text(target_text)

        # Also embed the source description to measure semantic preservation
        print("  Embedding source description...")
        src_desc_emb = client.embed_text(pair["source_desc"])

        for level in ATTACK_LEVELS:
            level_name = level["name"]

            if level_name == "baseline":
                # Clean image — no attack
                attacked_bytes = src_bytes
            else:
                # Determine text to overlay
                overlay_text = keyword if level["text_type"] == "keyword" else target_text

                if level["repeat"] > 1 and level["font_size"] > 0:
                    # Use scattered position for repeated text
                    attacked_bytes = add_typographic_text(
                        src_bytes, overlay_text,
                        font_size=level["font_size"],
                        position="scattered",
                        color=(0, 0, 0),
                        bg_box=level["bg_box"],
                    )
                else:
                    attacked_bytes = add_typographic_text(
                        src_bytes, overlay_text,
                        font_size=level["font_size"],
                        position="center",
                        color=(0, 0, 0),
                        bg_box=level["bg_box"],
                    )

            # Embed
            print(f"  [{level_name}] Embedding...")
            attacked_emb = client.embed_image(attacked_bytes)

            # Compute metrics
            sim_to_target = cosine_similarity(attacked_emb, target_emb)
            sim_to_src_desc = cosine_similarity(attacked_emb, src_desc_emb)

            level_result = {
                "level": level_name,
                "sim_to_target": sim_to_target,
                "sim_to_source_desc": sim_to_src_desc,
                "collision_gap": 1.0 - sim_to_target,
            }
            pair_result["levels"].append(level_result)
            pair_plot["similarities"].append((level_name, sim_to_target))

            print(f"    → target text: {sim_to_target:.4f} "
                  f"(gap={1.0 - sim_to_target:.4f})")
            print(f"    → source desc: {sim_to_src_desc:.4f}")

        # Best attack for this pair
        best = max(pair_result["levels"], key=lambda x: x["sim_to_target"])
        baseline = pair_result["levels"][0]["sim_to_target"]
        pair_result["best_level"] = best["level"]
        pair_result["best_similarity"] = best["sim_to_target"]
        pair_result["best_collision_gap"] = best["collision_gap"]
        pair_result["total_shift"] = best["sim_to_target"] - baseline

        print(f"\n  Best: [{best['level']}] sim={best['sim_to_target']:.4f}, "
              f"gap={best['collision_gap']:.4f}, "
              f"shift={best['sim_to_target'] - baseline:+.4f}")

        results.append(pair_result)
        plot_data.append(pair_plot)

    # ===== Summary statistics =====
    print(f"\n{'=' * 60}")
    print("Semantic Collision Summary")
    print(f"{'=' * 60}")

    best_sims = [r["best_similarity"] for r in results]
    total_shifts = [r["total_shift"] for r in results]
    collision_gaps = [r["best_collision_gap"] for r in results]

    sim_ci = compute_confidence_interval(best_sims)
    shift_ci = compute_confidence_interval(total_shifts)
    gap_ci = compute_confidence_interval(collision_gaps)

    print(f"  Best similarities: {sim_ci['mean']:.4f} "
          f"[{sim_ci['lower']:.4f}, {sim_ci['upper']:.4f}]")
    print(f"  Total shifts:      {shift_ci['mean']:+.4f} "
          f"[{shift_ci['lower']:+.4f}, {shift_ci['upper']:+.4f}]")
    print(f"  Collision gaps:    {gap_ci['mean']:.4f} "
          f"[{gap_ci['lower']:.4f}, {gap_ci['upper']:.4f}]")

    for r in results:
        print(f"\n  {r['source']}→'{r['keyword']}':")
        print(f"    Best level: {r['best_level']}")
        print(f"    Max sim:    {r['best_similarity']:.4f}")
        print(f"    Gap:        {r['best_collision_gap']:.4f}")

    # Save results
    os.makedirs(RESULTS_DATA_DIR, exist_ok=True)
    output = {
        "experiment": "exp7_semantic_collision",
        "results": results,
        "summary": {
            "best_similarity_ci": sim_ci,
            "total_shift_ci": shift_ci,
            "collision_gap_ci": gap_ci,
        },
    }
    output_path = os.path.join(RESULTS_DATA_DIR, "exp7_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Plot collision progression
    os.makedirs(RESULTS_FIGURES_DIR, exist_ok=True)
    plot_collision_progression(
        plot_data,
        title="Exp7: Semantic Collision — Similarity Progression",
        save_path=os.path.join(RESULTS_FIGURES_DIR, "exp7_collision_progression.png"),
    )

    return output


if __name__ == "__main__":
    run_experiment()
