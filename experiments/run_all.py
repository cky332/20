"""Run all experiments sequentially.

Usage:
    python -m experiments.run_all              # Run all
    python -m experiments.run_all --skip 3 4   # Skip experiments 3 and 4
"""

import argparse
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import RESULTS_DATA_DIR


def main():
    parser = argparse.ArgumentParser(description="Run Gemini Embedding 2 attack experiments")
    parser.add_argument("--skip", nargs="*", type=int, default=[],
                        help="Experiment numbers to skip (e.g., --skip 3 4)")
    args = parser.parse_args()

    experiments = {
        1: ("Basic Typographic Attack", "experiments.exp1_typographic_basic"),
        2: ("Cross-Modal Alignment Attack", "experiments.exp2_cross_modal"),
        3: ("Document Retrieval Poisoning", "experiments.exp3_retrieval_poison"),
        4: ("Parameter Sensitivity Study", "experiments.exp4_parameter_study"),
    }

    results_summary = {}

    for exp_num, (name, module_path) in experiments.items():
        if exp_num in args.skip:
            print(f"\n{'#' * 60}")
            print(f"# Skipping Experiment {exp_num}: {name}")
            print(f"{'#' * 60}")
            continue

        print(f"\n{'#' * 60}")
        print(f"# Running Experiment {exp_num}: {name}")
        print(f"{'#' * 60}\n")

        try:
            module = __import__(module_path, fromlist=["run_experiment"])
            result = module.run_experiment()
            results_summary[f"exp{exp_num}"] = {"status": "success", "name": name}
        except Exception as e:
            print(f"\n!!! Experiment {exp_num} FAILED: {e}")
            traceback.print_exc()
            results_summary[f"exp{exp_num}"] = {
                "status": "failed",
                "name": name,
                "error": str(e),
            }

    # Print summary
    print(f"\n{'=' * 60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'=' * 60}")

    for exp_id, info in results_summary.items():
        status = "OK" if info["status"] == "success" else "FAILED"
        print(f"  {exp_id}: [{status}] {info['name']}")

    # Save summary
    os.makedirs(RESULTS_DATA_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DATA_DIR, "run_summary.json"), "w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\nSummary saved to: {os.path.join(RESULTS_DATA_DIR, 'run_summary.json')}")


if __name__ == "__main__":
    main()
