"""Experiment 4: Parameter Sensitivity Study.

Systematically varies attack parameters (font size, position, color,
repetition) to understand which factors most influence the effectiveness
of typographic attacks on Gemini Embedding 2.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import RESULTS_DATA_DIR, RESULTS_FIGURES_DIR
from src.embedding_client import GeminiEmbeddingClient
from src.image_generator import SOURCE_IMAGES
from src.typographic_attack import add_typographic_text
from src.similarity import cosine_similarity
from src.visualization import plot_parameter_sensitivity


def run_experiment():
    """Run the parameter sensitivity study."""
    print("=" * 60)
    print("Experiment 4: Parameter Sensitivity Study")
    print("=" * 60)

    client = GeminiEmbeddingClient()

    # Fixed source and target for controlled comparison
    src_key = "red_circle"
    target_text = "iPod"

    src_bytes = SOURCE_IMAGES[src_key]()
    print(f"Source: {src_key}, Target: '{target_text}'")

    # Get baseline embeddings
    print("\nGetting baseline embeddings...")
    clean_emb = client.embed_image(src_bytes)
    target_emb = client.embed_text(target_text)
    baseline_sim = cosine_similarity(clean_emb, target_emb)
    print(f"  Baseline similarity (clean→target): {baseline_sim:.4f}")

    all_results = {}

    # --- Study 1: Font Size ---
    print("\n--- Study 1: Font Size ---")
    font_sizes = [16, 24, 36, 48, 60, 80, 100]
    font_size_shifts = []

    for fs in font_sizes:
        attacked = add_typographic_text(src_bytes, target_text,
                                         font_size=fs, position="center", color=(0, 0, 0))
        emb = client.embed_image(attacked)
        sim = cosine_similarity(emb, target_emb)
        shift = sim - baseline_sim
        font_size_shifts.append(shift)
        print(f"  Font size {fs:3d}: sim={sim:.4f}, shift={shift:+.4f}")

    all_results["font_size"] = {
        "values": font_sizes,
        "shifts": font_size_shifts,
        "similarities": [baseline_sim + s for s in font_size_shifts],
    }

    # --- Study 2: Position ---
    print("\n--- Study 2: Text Position ---")
    positions = ["center", "top", "bottom", "top-left", "top-right",
                 "bottom-left", "bottom-right", "scattered"]
    position_shifts = []

    for pos in positions:
        attacked = add_typographic_text(src_bytes, target_text,
                                         font_size=48, position=pos, color=(0, 0, 0))
        emb = client.embed_image(attacked)
        sim = cosine_similarity(emb, target_emb)
        shift = sim - baseline_sim
        position_shifts.append(shift)
        print(f"  Position {pos:15s}: sim={sim:.4f}, shift={shift:+.4f}")

    all_results["position"] = {
        "values": positions,
        "shifts": position_shifts,
    }

    # --- Study 3: Text Color ---
    print("\n--- Study 3: Text Color ---")
    colors = [
        ("black", (0, 0, 0)),
        ("dark_gray", (80, 80, 80)),
        ("gray", (128, 128, 128)),
        ("light_gray", (200, 200, 200)),
        ("near_white", (240, 240, 240)),
        ("red", (220, 30, 30)),
        ("blue", (30, 30, 220)),
        ("green", (30, 180, 30)),
    ]
    color_shifts = []
    color_names = []

    for name, color in colors:
        attacked = add_typographic_text(src_bytes, target_text,
                                         font_size=48, position="center", color=color)
        emb = client.embed_image(attacked)
        sim = cosine_similarity(emb, target_emb)
        shift = sim - baseline_sim
        color_shifts.append(shift)
        color_names.append(name)
        print(f"  Color {name:12s}: sim={sim:.4f}, shift={shift:+.4f}")

    all_results["color"] = {
        "values": color_names,
        "shifts": color_shifts,
    }

    # --- Study 4: Repetition ---
    print("\n--- Study 4: Text Repetition ---")
    repetitions = [1, 2, 3, 4, 5]
    rep_shifts = []

    for rep in repetitions:
        attacked = add_typographic_text(src_bytes, target_text,
                                         font_size=36, position="center",
                                         color=(0, 0, 0), repeat=rep)
        emb = client.embed_image(attacked)
        sim = cosine_similarity(emb, target_emb)
        shift = sim - baseline_sim
        rep_shifts.append(shift)
        print(f"  Repeat {rep}x: sim={sim:.4f}, shift={shift:+.4f}")

    all_results["repetition"] = {
        "values": repetitions,
        "shifts": rep_shifts,
    }

    # --- Study 5: Background Box ---
    print("\n--- Study 5: Background Box ---")
    bg_options = [("no_box", False), ("with_box", True)]
    bg_shifts = []
    bg_names = []

    for name, use_box in bg_options:
        attacked = add_typographic_text(src_bytes, target_text,
                                         font_size=48, position="center",
                                         color=(0, 0, 0), bg_box=use_box)
        emb = client.embed_image(attacked)
        sim = cosine_similarity(emb, target_emb)
        shift = sim - baseline_sim
        bg_shifts.append(shift)
        bg_names.append(name)
        print(f"  {name:10s}: sim={sim:.4f}, shift={shift:+.4f}")

    all_results["background_box"] = {
        "values": bg_names,
        "shifts": bg_shifts,
    }

    # Save results
    os.makedirs(RESULTS_DATA_DIR, exist_ok=True)
    output = {
        "experiment": "exp4_parameter_study",
        "source": src_key,
        "target": target_text,
        "baseline_similarity": baseline_sim,
        "studies": all_results,
    }
    with open(os.path.join(RESULTS_DATA_DIR, "exp4_results.json"), "w") as f:
        json.dump(output, f, indent=2)

    # Generate plots
    os.makedirs(RESULTS_FIGURES_DIR, exist_ok=True)

    plot_parameter_sensitivity(
        "Font Size (px)", [str(s) for s in font_sizes], font_size_shifts,
        title="Exp4: Attack Sensitivity to Font Size",
        save_path=os.path.join(RESULTS_FIGURES_DIR, "exp4_font_size.png"),
    )
    plot_parameter_sensitivity(
        "Text Position", positions, position_shifts,
        title="Exp4: Attack Sensitivity to Text Position",
        save_path=os.path.join(RESULTS_FIGURES_DIR, "exp4_position.png"),
    )
    plot_parameter_sensitivity(
        "Text Color", color_names, color_shifts,
        title="Exp4: Attack Sensitivity to Text Color",
        save_path=os.path.join(RESULTS_FIGURES_DIR, "exp4_color.png"),
    )
    plot_parameter_sensitivity(
        "Repetition Count", [str(r) for r in repetitions], rep_shifts,
        title="Exp4: Attack Sensitivity to Text Repetition",
        save_path=os.path.join(RESULTS_FIGURES_DIR, "exp4_repetition.png"),
    )

    print(f"\nResults saved to: {os.path.join(RESULTS_DATA_DIR, 'exp4_results.json')}")
    return output


if __name__ == "__main__":
    run_experiment()
