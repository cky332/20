"""Post-hoc Statistical Analysis for all experiments.

Loads results from exp1-exp7 and performs:
- Paired t-tests (clean vs attacked)
- Confidence intervals on shifts
- Cohen's d effect sizes
- Random baseline comparison (requires API calls)
- Generates summary plots with p-values
"""

import json
import os
import random
import string
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import RESULTS_DATA_DIR, RESULTS_FIGURES_DIR
from src.statistics import (
    paired_ttest,
    compute_cohens_d,
    compute_confidence_interval,
)


def _load_json(filename):
    """Load a JSON results file, return None if not found or corrupted."""
    path = os.path.join(RESULTS_DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"  Warning: {filename} not found, skipping.")
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"  Warning: {filename} is corrupted (incomplete write?), skipping.")
        return None


def analyze_exp1(data):
    """Statistical analysis for Experiment 1: Basic Typographic Attack."""
    print("\n--- Exp1: Basic Typographic Attack ---")
    results = data.get("results", [])
    if not results:
        return None

    clean_sims = [r["clean_to_target"] for r in results]
    attacked_sims = [r["attacked_to_target"] for r in results]
    shifts = [r["similarity_shift"] for r in results]

    ttest = paired_ttest(clean_sims, attacked_sims)
    d = compute_cohens_d(clean_sims, attacked_sims)
    shift_ci = compute_confidence_interval(shifts)

    analysis = {
        "paired_ttest": ttest,
        "cohens_d": d,
        "shift_ci": shift_ci,
        "n_samples": len(results),
    }

    print(f"  n={len(results)}")
    print(f"  Mean shift: {shift_ci['mean']:+.4f} [{shift_ci['lower']:+.4f}, {shift_ci['upper']:+.4f}]")
    print(f"  t={ttest['t_statistic']:.3f}, p={ttest['p_value']:.4f}, "
          f"significant={ttest['significant']}")
    print(f"  Cohen's d={d:.3f}")

    # Enhanced analysis with control group (if available)
    has_controls = all("random_shift" in r for r in results)
    if has_controls:
        random_shifts = [r["random_shift"] for r in results]
        advantages = [r["targeted_advantage"] for r in results]

        ttest_vs_random = paired_ttest(random_shifts, shifts)
        d_vs_random = compute_cohens_d(random_shifts, shifts)
        advantage_ci = compute_confidence_interval(advantages)

        analysis["control_group"] = {
            "ttest_target_vs_random": ttest_vs_random,
            "cohens_d_vs_random": d_vs_random,
            "targeted_advantage_ci": advantage_ci,
            "mean_random_shift": float(sum(random_shifts) / len(random_shifts)),
        }

        print(f"\n  [Control group analysis]")
        print(f"  Mean random shift: {analysis['control_group']['mean_random_shift']:+.4f}")
        print(f"  Mean targeted advantage: {advantage_ci['mean']:+.4f} "
              f"[{advantage_ci['lower']:+.4f}, {advantage_ci['upper']:+.4f}]")
        print(f"  Target vs Random t={ttest_vs_random['t_statistic']:.3f}, "
              f"p={ttest_vs_random['p_value']:.6f}")
        print(f"  Cohen's d (vs random): {d_vs_random:.3f}")

    # Mismatch analysis (if available)
    summary = data.get("summary", {})
    mismatch = summary.get("mismatch_analysis")
    if mismatch:
        analysis["mismatch_analysis"] = mismatch
        print(f"\n  [Mismatch analysis]")
        print(f"  Matched shift:    {mismatch['matched_mean_shift']:+.4f}")
        print(f"  Mismatched shift: {mismatch['mismatched_mean_shift']:+.4f}")
        print(f"  Semantic specificity: {mismatch['semantic_specificity_confirmed']}")

    return analysis


def analyze_exp2(data):
    """Statistical analysis for Experiment 2: Cross-Modal Alignment."""
    print("\n--- Exp2: Cross-Modal Alignment Attack ---")
    results = data.get("results", [])
    if not results:
        return None

    # Group by intensity
    by_intensity = {}
    for r in results:
        intensity = r["intensity"]
        if intensity not in by_intensity:
            by_intensity[intensity] = {"clean": [], "attacked": [], "shifts": []}
        by_intensity[intensity]["clean"].append(r["sim_clean_to_target"])
        by_intensity[intensity]["attacked"].append(r["sim_attacked_to_target"])
        by_intensity[intensity]["shifts"].append(r["shift_toward_target"])

    analysis = {"by_intensity": {}}
    for intensity, sims in by_intensity.items():
        ttest = paired_ttest(sims["clean"], sims["attacked"])
        d = compute_cohens_d(sims["clean"], sims["attacked"])
        shift_ci = compute_confidence_interval(sims["shifts"])
        analysis["by_intensity"][intensity] = {
            "paired_ttest": ttest,
            "cohens_d": d,
            "shift_ci": shift_ci,
        }
        print(f"  [{intensity}] shift={shift_ci['mean']:+.4f}, "
              f"p={ttest['p_value']:.4f}, d={d:.3f}")

    return analysis


def analyze_exp3(data):
    """Statistical analysis for Experiment 3: Retrieval Poisoning."""
    print("\n--- Exp3: Document Retrieval Poisoning ---")
    sims = data.get("similarity_comparisons", {})
    if not sims:
        return None

    analysis = {
        "shift_invisible": sims.get("shift_invisible", 0),
        "shift_visible": sims.get("shift_visible", 0),
        "baseline_text_sim": sims.get("text_to_query", 0),
        "clean_img_sim": sims.get("clean_img_to_query", 0),
    }

    print(f"  Invisible poison shift: {analysis['shift_invisible']:+.4f}")
    print(f"  Visible poison shift:   {analysis['shift_visible']:+.4f}")

    return analysis


def analyze_exp5(data):
    """Statistical analysis for Experiment 5: Adversarial Hubness."""
    print("\n--- Exp5: Adversarial Hubness ---")
    stats = data.get("statistical_analysis", {})
    hubness = data.get("hubness_results", {})

    if not stats:
        return None

    analysis = {
        "adversarial_ci_k1": stats.get("adversarial_ci_k1"),
        "random_ci_k1": stats.get("random_ci_k1"),
        "ttest": stats.get("ttest_adv_vs_clean"),
    }

    ttest = stats.get("ttest_adv_vs_clean", {})
    print(f"  Adversarial N_1 mean: {stats.get('adversarial_ci_k1', {}).get('mean', 'N/A')}")
    print(f"  Random N_1 mean:      {stats.get('random_ci_k1', {}).get('mean', 'N/A')}")
    print(f"  t-test p-value:       {ttest.get('p_value', 'N/A')}")

    return analysis


def analyze_exp7(data):
    """Statistical analysis for Experiment 7: Semantic Collision."""
    print("\n--- Exp7: Semantic Collision ---")
    summary = data.get("summary", {})
    results = data.get("results", [])

    if not results:
        return None

    best_sims = [r["best_similarity"] for r in results]
    shifts = [r["total_shift"] for r in results]

    sim_ci = compute_confidence_interval(best_sims)
    shift_ci = compute_confidence_interval(shifts)

    analysis = {
        "best_similarity_ci": sim_ci,
        "total_shift_ci": shift_ci,
        "per_pair": [
            {"pair": f"{r['source']}→{r['keyword']}",
             "best_sim": r["best_similarity"],
             "best_level": r["best_level"]}
            for r in results
        ],
    }

    print(f"  Best similarity: {sim_ci['mean']:.4f} "
          f"[{sim_ci['lower']:.4f}, {sim_ci['upper']:.4f}]")
    print(f"  Total shift:     {shift_ci['mean']:+.4f} "
          f"[{shift_ci['lower']:+.4f}, {shift_ci['upper']:+.4f}]")

    return analysis


def run_experiment():
    """Run post-hoc statistical analysis on all experiment results."""
    print("=" * 60)
    print("Statistical Analysis: Post-hoc Analysis of All Experiments")
    print("=" * 60)

    all_analyses = {}

    # Analyze each experiment
    exp_analyzers = {
        "exp1": ("exp1_results.json", analyze_exp1),
        "exp2": ("exp2_results.json", analyze_exp2),
        "exp3": ("exp3_results.json", analyze_exp3),
        "exp5": ("exp5_results.json", analyze_exp5),
        "exp7": ("exp7_results.json", analyze_exp7),
    }

    for exp_id, (filename, analyzer) in exp_analyzers.items():
        data = _load_json(filename)
        if data:
            analysis = analyzer(data)
            if analysis:
                all_analyses[exp_id] = analysis

    # Summary
    print(f"\n{'=' * 60}")
    print("Overall Statistical Summary")
    print(f"{'=' * 60}")

    significant_exps = []
    for exp_id, analysis in all_analyses.items():
        if isinstance(analysis, dict):
            # Check for significant t-test
            ttest = analysis.get("paired_ttest") or analysis.get("ttest")
            if ttest and ttest.get("significant"):
                significant_exps.append(exp_id)
            # Check nested
            by_intensity = analysis.get("by_intensity", {})
            for intensity, int_data in by_intensity.items():
                if int_data.get("paired_ttest", {}).get("significant"):
                    significant_exps.append(f"{exp_id}_{intensity}")

    print(f"  Experiments with significant results: {significant_exps or 'None'}")

    # Save
    os.makedirs(RESULTS_DATA_DIR, exist_ok=True)
    output = {
        "experiment": "statistical_analysis",
        "analyses": all_analyses,
        "significant_experiments": significant_exps,
    }

    # Convert numpy types for JSON serialization
    def _convert(obj):
        if hasattr(obj, "item"):
            return obj.item()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return obj

    output_path = os.path.join(RESULTS_DATA_DIR, "statistical_analysis.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=_convert)
    print(f"\nResults saved to: {output_path}")

    return output


if __name__ == "__main__":
    run_experiment()
