"""Experiment 1: Basic Typographic Attack on Gemini Embedding 2.

Tests whether adding misleading text to images shifts their embeddings
closer to the target text in Gemini Embedding 2's unified embedding space.

Includes random-text control group and mismatch analysis to verify that
the observed shift is semantically specific to the overlaid text content.

Metrics:
- Cosine similarity: clean image vs target text (baseline)
- Cosine similarity: attacked image vs target text (after attack)
- Cosine similarity: random-text image vs target text (control)
- Similarity shift = attacked_sim - clean_sim
- Random shift = random_sim - clean_sim
- Targeted advantage = target_shift - random_shift
- Attack success rate (target_shift > random baseline + 2*std)
"""

import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import RESULTS_DATA_DIR, RESULTS_FIGURES_DIR, ASSETS_DIR
from src.embedding_client import GeminiEmbeddingClient
from src.image_generator import SOURCE_IMAGES, TARGET_LABELS, image_to_bytes, bytes_to_image
from src.typographic_attack import add_typographic_text
from src.similarity import cosine_similarity, compute_attack_effectiveness, compute_attack_success_rate
from src.statistics import paired_ttest, compute_cohens_d, compute_confidence_interval
from src.visualization import (
    plot_attack_comparison,
    plot_target_vs_random_shift,
    plot_mismatch_heatmap,
    plot_shift_by_target_boxplot,
)

# Random control texts: semantically unrelated to any target label
RANDOM_CONTROL_TEXTS = [
    "quantum",
    "philosophy",
    "umbrella",
    "harmony",
    "elephant",
]


def run_experiment():
    """Run the basic typographic attack experiment with controls."""
    print("=" * 60)
    print("Experiment 1: Basic Typographic Attack")
    print("=" * 60)

    client = GeminiEmbeddingClient()

    # Use all 8 source images and all 8 target labels
    source_keys = list(SOURCE_IMAGES.keys())
    target_texts = TARGET_LABELS

    os.makedirs(ASSETS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_FIGURES_DIR, exist_ok=True)

    # ── Phase 1: Embed all clean images and target texts ──
    print("\n--- Phase 1: Embedding clean images and target texts ---")

    clean_embeddings = {}
    for src_key in source_keys:
        src_bytes = SOURCE_IMAGES[src_key]()
        with open(os.path.join(ASSETS_DIR, f"{src_key}.png"), "wb") as f:
            f.write(src_bytes)
        print(f"  Embedding clean image: {src_key}")
        clean_embeddings[src_key] = {
            "bytes": src_bytes,
            "emb": client.embed_image(src_bytes),
        }

    target_embeddings = {}
    for target_text in target_texts:
        print(f"  Embedding target text: '{target_text}'")
        target_embeddings[target_text] = client.embed_text(target_text)

    # ── Phase 2: Attack and control for each (source, target) pair ──
    print("\n--- Phase 2: Running attacks with controls ---")

    results = []
    all_effectiveness = []

    # Track mismatch data: for each attacked image, similarity to ALL targets
    # Key: (src_key, overlaid_text) -> dict of {measured_target: shift}
    mismatch_data = []

    for src_key in source_keys:
        print(f"\n--- Source: {src_key} ---")
        src_bytes = clean_embeddings[src_key]["bytes"]
        clean_emb = clean_embeddings[src_key]["emb"]

        for target_text in target_texts:
            print(f"  Target: '{target_text}'")
            target_emb = target_embeddings[target_text]

            # ── Target attack ──
            attacked_bytes = add_typographic_text(
                src_bytes, target_text,
                font_size=60, position="center", color=(0, 0, 0)
            )
            attacked_name = f"{src_key}_attack_{target_text.replace(' ', '_')}.png"
            with open(os.path.join(ASSETS_DIR, attacked_name), "wb") as f:
                f.write(attacked_bytes)

            print(f"    Embedding attacked image...")
            attacked_emb = client.embed_image(attacked_bytes)

            # ── Random control attack ──
            # Pick a random control text (cycle through the list)
            ctrl_idx = target_texts.index(target_text) % len(RANDOM_CONTROL_TEXTS)
            random_text = RANDOM_CONTROL_TEXTS[ctrl_idx]

            random_attacked_bytes = add_typographic_text(
                src_bytes, random_text,
                font_size=60, position="center", color=(0, 0, 0)
            )
            print(f"    Embedding random-text control ('{random_text}')...")
            random_emb = client.embed_image(random_attacked_bytes)

            # ── Compute similarities ──
            clean_sim = cosine_similarity(clean_emb, target_emb)
            attacked_sim = cosine_similarity(attacked_emb, target_emb)
            random_sim = cosine_similarity(random_emb, target_emb)
            preservation_sim = cosine_similarity(clean_emb, attacked_emb)

            effectiveness = compute_attack_effectiveness(
                clean_sim, attacked_sim, random_sim=random_sim
            )

            result = {
                "source": src_key,
                "target": target_text,
                "random_control_text": random_text,
                "clean_to_target": clean_sim,
                "attacked_to_target": attacked_sim,
                "random_to_target": random_sim,
                "clean_to_attacked": preservation_sim,
                **effectiveness,
            }
            results.append(result)
            all_effectiveness.append(effectiveness)

            print(f"    Clean→Target:    {clean_sim:.4f}")
            print(f"    Attacked→Target: {attacked_sim:.4f}")
            print(f"    Random→Target:   {random_sim:.4f}")
            print(f"    Target Shift:    {effectiveness['similarity_shift']:+.4f}")
            print(f"    Random Shift:    {effectiveness['random_shift']:+.4f}")
            print(f"    Targeted Adv:    {effectiveness['targeted_advantage']:+.4f}")
            print(f"    Clean→Attacked:  {preservation_sim:.4f}")

            # ── Mismatch analysis: attacked image vs ALL other targets ──
            mismatch_row = {}
            for other_target in target_texts:
                other_emb = target_embeddings[other_target]
                sim_to_other = cosine_similarity(attacked_emb, other_emb)
                clean_sim_to_other = cosine_similarity(clean_emb, other_emb)
                mismatch_row[other_target] = sim_to_other - clean_sim_to_other
            mismatch_data.append({
                "source": src_key,
                "overlaid_text": target_text,
                "shifts_to_all_targets": mismatch_row,
            })

    # ── Phase 3: Statistical Analysis ──
    print(f"\n{'=' * 60}")
    print("Statistical Analysis")
    print(f"{'=' * 60}")

    # 3a. Overall success rate with shift > 0 threshold (original)
    success_basic = compute_attack_success_rate(all_effectiveness, threshold=0.0)
    print(f"\n[Basic ASR (shift > 0)]")
    print(f"  Success Rate: {success_basic['success_rate']:.1%}")
    print(f"  Successful: {success_basic['successful']}/{success_basic['total']}")
    print(f"  Mean Shift: {success_basic['mean_shift']:+.4f}")
    print(f"  Std Shift:  {success_basic['std_shift']:.4f}")

    # 3b. Success rate with random baseline threshold
    success_strict = compute_attack_success_rate(
        all_effectiveness, use_random_baseline=True
    )
    print(f"\n[Strict ASR (shift > random_mean + 2*std)]")
    print(f"  Threshold:    {success_strict['threshold_used']:+.4f}")
    print(f"  Success Rate: {success_strict['success_rate']:.1%}")
    print(f"  Successful:   {success_strict['successful']}/{success_strict['total']}")

    # 3c. Paired t-test: target shift vs random shift
    target_shifts = [r["similarity_shift"] for r in all_effectiveness]
    random_shifts = [r["random_shift"] for r in all_effectiveness]
    advantages = [r["targeted_advantage"] for r in all_effectiveness]

    ttest_result = paired_ttest(random_shifts, target_shifts)
    cohens_d = compute_cohens_d(random_shifts, target_shifts)
    ci = compute_confidence_interval(advantages)

    print(f"\n[Paired t-test: target shift vs random shift]")
    print(f"  t = {ttest_result['t_statistic']:.3f}, p = {ttest_result['p_value']:.6f}")
    print(f"  Significant: {ttest_result['significant']}")
    print(f"  Cohen's d: {cohens_d:.3f}")
    print(f"  Mean targeted advantage: {ci['mean']:+.4f}")
    print(f"  95% CI: [{ci['lower']:+.4f}, {ci['upper']:+.4f}]")

    # 3d. Wilcoxon signed-rank test (non-parametric alternative)
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(target_shifts, random_shifts)
    print(f"\n[Wilcoxon signed-rank test]")
    print(f"  W = {wilcoxon_stat:.3f}, p = {wilcoxon_p:.6f}")
    print(f"  Significant: {wilcoxon_p < 0.05}")

    # 3e. By-target analysis
    print(f"\n[Per-target breakdown]")
    shifts_by_target = {}
    for r in results:
        t = r["target"]
        if t not in shifts_by_target:
            shifts_by_target[t] = {"target_shifts": [], "random_shifts": []}
        shifts_by_target[t]["target_shifts"].append(r["similarity_shift"])
        shifts_by_target[t]["random_shifts"].append(r["random_shift"])

    for t in target_texts:
        ts = shifts_by_target[t]["target_shifts"]
        rs = shifts_by_target[t]["random_shifts"]
        print(f"  '{t}': target_shift={np.mean(ts):+.4f} "
              f"random_shift={np.mean(rs):+.4f} "
              f"advantage={np.mean(ts) - np.mean(rs):+.4f}")

    # 3f. Kruskal-Wallis: do different targets have different shifts?
    groups = [shifts_by_target[t]["target_shifts"] for t in target_texts]
    kw_stat, kw_p = stats.kruskal(*groups)
    print(f"\n[Kruskal-Wallis across targets]")
    print(f"  H = {kw_stat:.3f}, p = {kw_p:.6f}")
    print(f"  Significant difference between targets: {kw_p < 0.05}")

    # 3g. Mismatch analysis summary
    print(f"\n[Mismatch Analysis]")
    matched_shifts = []
    mismatched_shifts = []
    for entry in mismatch_data:
        overlaid = entry["overlaid_text"]
        for measured_target, shift in entry["shifts_to_all_targets"].items():
            if measured_target == overlaid:
                matched_shifts.append(shift)
            else:
                mismatched_shifts.append(shift)

    print(f"  Matched (overlaid == measured) mean shift:   {np.mean(matched_shifts):+.4f}")
    print(f"  Mismatched (overlaid != measured) mean shift: {np.mean(mismatched_shifts):+.4f}")
    mismatch_ttest = stats.ttest_ind(matched_shifts, mismatched_shifts)
    print(f"  t = {mismatch_ttest.statistic:.3f}, p = {mismatch_ttest.pvalue:.6f}")
    print(f"  Confirms semantic specificity: {mismatch_ttest.pvalue < 0.05}")

    # ── Phase 4: Save results ──
    output = {
        "experiment": "exp1_typographic_basic",
        "config": {
            "source_images": source_keys,
            "target_texts": target_texts,
            "random_control_texts": RANDOM_CONTROL_TEXTS,
            "font_size": 60,
            "position": "center",
            "color": [0, 0, 0],
        },
        "results": results,
        "summary": {
            "basic_asr": success_basic,
            "strict_asr": success_strict,
            "paired_ttest": ttest_result,
            "cohens_d": cohens_d,
            "targeted_advantage_ci": ci,
            "wilcoxon": {
                "statistic": float(wilcoxon_stat),
                "p_value": float(wilcoxon_p),
                "significant": wilcoxon_p < 0.05,
            },
            "kruskal_wallis": {
                "statistic": float(kw_stat),
                "p_value": float(kw_p),
                "significant": kw_p < 0.05,
            },
            "mismatch_analysis": {
                "matched_mean_shift": float(np.mean(matched_shifts)),
                "mismatched_mean_shift": float(np.mean(mismatched_shifts)),
                "t_statistic": float(mismatch_ttest.statistic),
                "p_value": float(mismatch_ttest.pvalue),
                "semantic_specificity_confirmed": mismatch_ttest.pvalue < 0.05,
            },
        },
        "mismatch_data": mismatch_data,
    }
    output_path = os.path.join(RESULTS_DATA_DIR, "exp1_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # ── Phase 5: Visualizations ──

    # 5a. Attack comparison bar chart (clean vs attacked)
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
        title="Exp1: Typographic Attack - Similarity Shift",
        save_path=os.path.join(RESULTS_FIGURES_DIR, "exp1_attack_comparison.png"),
    )

    # 5b. Target shift vs random shift comparison
    shift_plot_data = [
        {
            "label": f"{r['source']}→{r['target']}",
            "target_shift": r["similarity_shift"],
            "random_shift": r["random_shift"],
        }
        for r in results
    ]
    plot_target_vs_random_shift(
        shift_plot_data,
        title="Exp1: Target vs Random Text Shift",
        save_path=os.path.join(RESULTS_FIGURES_DIR, "exp1_target_vs_random.png"),
    )

    # 5c. Mismatch heatmap (aggregate by target: avg shift across all sources)
    # Build matrix: rows = overlaid text, cols = measured target
    mismatch_matrix = np.zeros((len(target_texts), len(target_texts)))
    count_matrix = np.zeros((len(target_texts), len(target_texts)))
    for entry in mismatch_data:
        i = target_texts.index(entry["overlaid_text"])
        for j, t in enumerate(target_texts):
            mismatch_matrix[i, j] += entry["shifts_to_all_targets"][t]
            count_matrix[i, j] += 1
    mismatch_matrix /= np.maximum(count_matrix, 1)

    plot_mismatch_heatmap(
        mismatch_matrix,
        source_labels=[f"overlaid: {t}" for t in target_texts],
        target_labels=target_texts,
        title="Exp1: Mismatch Analysis (avg shift across sources)",
        save_path=os.path.join(RESULTS_FIGURES_DIR, "exp1_mismatch_heatmap.png"),
    )

    # 5d. Boxplot by target
    boxplot_data = {t: shifts_by_target[t]["target_shifts"] for t in target_texts}
    plot_shift_by_target_boxplot(
        boxplot_data,
        title="Exp1: Shift Distribution by Target Text",
        save_path=os.path.join(RESULTS_FIGURES_DIR, "exp1_shift_by_target.png"),
    )

    return output


if __name__ == "__main__":
    run_experiment()
