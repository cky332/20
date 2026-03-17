"""Experiment 5: Adversarial Hubness Attack on Gemini Embedding 2.

Tests whether typographic manipulation can create adversarial "hub" images
that become nearest neighbors for many unrelated queries in the embedding space.

Background:
In high-dimensional spaces, some points naturally become "hubs" — nearest
neighbors of disproportionately many other points (Radovanovic et al., 2010).
This experiment tests whether an attacker can deliberately craft hub images
by overlaying keywords from multiple unrelated domains.

Metrics:
- N_k hubness score: how many queries have the adversarial image as top-k neighbor
- Hub score ratio: N_k(adversarial) / max(N_k(clean))
- Capture rate: fraction of queries "captured" by adversarial hub
- Statistical significance vs random text baseline
"""

import json
import os
import random
import string
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import RESULTS_DATA_DIR, RESULTS_FIGURES_DIR, ASSETS_DIR
from src.embedding_client import GeminiEmbeddingClient
from src.image_generator import SOURCE_IMAGES
from src.typographic_attack import add_typographic_text, add_multi_text_overlay
from src.similarity import cosine_similarity
from src.statistics import (
    compute_hubness_distribution,
    compute_hubness_stats,
    paired_ttest,
    compute_confidence_interval,
)
from src.visualization import plot_hubness_distribution, plot_similarity_heatmap


# Diverse query set spanning unrelated domains
QUERY_SET = {
    "q_baking": "how to bake sourdough bread at home",
    "q_physics": "quantum physics wave particle duality",
    "q_stocks": "stock market investment portfolio strategy",
    "q_cats": "domestic cat breeds and characteristics",
    "q_basketball": "basketball rules and scoring system",
    "q_climate": "climate change effects on polar ice caps",
    "q_pasta": "Italian pasta carbonara recipe ingredients",
    "q_mars": "Mars exploration rover mission updates",
    "q_crypto": "cryptocurrency bitcoin trading blockchain",
    "q_ml": "machine learning neural network training",
    "q_music": "classical piano sonata beethoven",
    "q_garden": "organic vegetable garden planting guide",
    "q_yoga": "yoga meditation breathing techniques",
    "q_history": "ancient Roman empire political structure",
    "q_photo": "digital camera photography composition tips",
}

# Keywords from multiple domains for hub construction
DOMAIN_KEYWORDS = {
    "food": "bread pizza pasta recipe cooking",
    "science": "physics quantum neural network",
    "finance": "stock market bitcoin trading investment",
    "animals": "cat dog breeds characteristics",
    "sports": "basketball soccer football rules",
    "nature": "climate ocean forest garden",
    "tech": "computer camera digital machine learning",
    "culture": "music history Roman classical yoga",
}


def _generate_random_text(length=40):
    """Generate random gibberish text of similar visual density."""
    words = []
    for _ in range(6):
        word_len = random.randint(3, 8)
        words.append("".join(random.choices(string.ascii_lowercase, k=word_len)))
    return " ".join(words)


def run_experiment():
    """Run the adversarial hubness attack experiment."""
    print("=" * 60)
    print("Experiment 5: Adversarial Hubness Attack")
    print("=" * 60)

    client = GeminiEmbeddingClient()

    # ===== Step A: Build query embeddings =====
    print("\n--- Step A: Embedding queries ---")
    query_embeddings = {}
    for qid, query_text in QUERY_SET.items():
        print(f"  Embedding query: {qid}")
        query_embeddings[qid] = client.embed_text(query_text)

    # ===== Step B: Build clean corpus embeddings =====
    print("\n--- Step B: Embedding clean corpus ---")
    corpus_keys = [
        "red_circle", "blue_rectangle", "green_triangle", "yellow_star",
        "ocean_scene", "forest_scene", "sunset_scene", "city_scene",
    ]
    clean_corpus = {}
    for key in corpus_keys:
        print(f"  Embedding corpus image: {key}")
        img_bytes = SOURCE_IMAGES[key]()
        clean_corpus[key] = client.embed_image(img_bytes)

    # ===== Step C: Measure baseline hubness =====
    print("\n--- Step C: Baseline hubness distribution ---")
    for k_val in [1, 3, 5]:
        baseline_hubness = compute_hubness_distribution(
            clean_corpus, query_embeddings, k=k_val
        )
        baseline_stats = compute_hubness_stats(baseline_hubness)
        print(f"  k={k_val}: mean={baseline_stats['mean']:.2f}, "
              f"max={baseline_stats['max']:.0f}, skew={baseline_stats['skewness']:.2f}")
        for item_id, count in sorted(baseline_hubness.items(),
                                      key=lambda x: x[1], reverse=True):
            if count > 0:
                print(f"    {item_id}: N_{k_val}={count}")

    # ===== Step D: Construct adversarial hub images =====
    print("\n--- Step D: Constructing adversarial hub images ---")
    os.makedirs(ASSETS_DIR, exist_ok=True)

    # Base image for hub construction
    base_img = SOURCE_IMAGES["red_circle"]()

    # Hub Type 1: Keyword soup — single long text with keywords from all domains
    keyword_soup = " ".join(DOMAIN_KEYWORDS.values())
    hub1_bytes = add_typographic_text(
        base_img, keyword_soup,
        font_size=20, position="center", color=(0, 0, 0),
    )
    with open(os.path.join(ASSETS_DIR, "hub1_keyword_soup.png"), "wb") as f:
        f.write(hub1_bytes)
    print("  Created hub1: keyword soup")

    # Hub Type 2: Multi-position — different domain keywords at different positions
    multi_texts = [
        (DOMAIN_KEYWORDS["food"], "top-left"),
        (DOMAIN_KEYWORDS["science"], "top-right"),
        (DOMAIN_KEYWORDS["finance"], "center"),
        (DOMAIN_KEYWORDS["animals"], "bottom-left"),
        (DOMAIN_KEYWORDS["sports"], "bottom-right"),
    ]
    hub2_bytes = add_multi_text_overlay(
        base_img, multi_texts, font_size=16, color=(0, 0, 0),
    )
    with open(os.path.join(ASSETS_DIR, "hub2_multi_position.png"), "wb") as f:
        f.write(hub2_bytes)
    print("  Created hub2: multi-position keywords")

    # Hub Type 3: Dense repetition — most common query terms repeated
    dense_text = "bread physics stock cat basketball climate pasta Mars bitcoin computer"
    hub3_bytes = add_typographic_text(
        base_img, dense_text,
        font_size=28, position="scattered", color=(0, 0, 0),
    )
    with open(os.path.join(ASSETS_DIR, "hub3_dense_repeat.png"), "wb") as f:
        f.write(hub3_bytes)
    print("  Created hub3: dense repetition")

    # Embed adversarial hubs
    adversarial_hubs = {}
    for hub_name, hub_bytes in [("hub1_soup", hub1_bytes),
                                 ("hub2_multi", hub2_bytes),
                                 ("hub3_dense", hub3_bytes)]:
        print(f"  Embedding {hub_name}...")
        adversarial_hubs[hub_name] = client.embed_image(hub_bytes)

    # ===== Step E: Random text baselines =====
    print("\n--- Step E: Creating random text baselines ---")
    random_hubs = {}
    for i in range(3):
        rand_text = _generate_random_text()
        rand_bytes = add_typographic_text(
            base_img, rand_text,
            font_size=20, position="center", color=(0, 0, 0),
        )
        rand_name = f"random_{i}"
        print(f"  Embedding {rand_name}...")
        random_hubs[rand_name] = client.embed_image(rand_bytes)

    # ===== Step F: Measure adversarial hubness =====
    print("\n--- Step F: Measuring adversarial hubness ---")
    results = {"baseline": {}, "adversarial": {}, "random": {}}

    for k_val in [1, 3, 5]:
        print(f"\n  === k={k_val} ===")

        # Baseline (clean corpus only)
        baseline_hubness = compute_hubness_distribution(
            clean_corpus, query_embeddings, k=k_val
        )
        results["baseline"][f"k{k_val}"] = baseline_hubness

        # With adversarial hubs added to corpus
        augmented_corpus = dict(clean_corpus)
        augmented_corpus.update(adversarial_hubs)

        adv_hubness = compute_hubness_distribution(
            augmented_corpus, query_embeddings, k=k_val
        )
        adv_stats = compute_hubness_stats(adv_hubness)
        results["adversarial"][f"k{k_val}"] = adv_hubness

        print(f"  Adversarial hubness (k={k_val}):")
        for hub_name in adversarial_hubs:
            nk = adv_hubness.get(hub_name, 0)
            capture_rate = nk / len(query_embeddings)
            print(f"    {hub_name}: N_{k_val}={nk}, "
                  f"capture_rate={capture_rate:.1%}")

        # With random hubs added to corpus
        random_corpus = dict(clean_corpus)
        random_corpus.update(random_hubs)

        rand_hubness = compute_hubness_distribution(
            random_corpus, query_embeddings, k=k_val
        )
        results["random"][f"k{k_val}"] = rand_hubness

        print(f"  Random baseline hubness (k={k_val}):")
        for rand_name in random_hubs:
            nk = rand_hubness.get(rand_name, 0)
            print(f"    {rand_name}: N_{k_val}={nk}")

    # ===== Step G: Statistical comparison =====
    print("\n--- Step G: Statistical analysis ---")

    # Compare adversarial vs random hub scores at k=1
    adv_scores_k1 = [results["adversarial"]["k1"].get(h, 0) for h in adversarial_hubs]
    rand_scores_k1 = [results["random"]["k1"].get(r, 0) for r in random_hubs]

    # Use all hub scores for CI
    adv_ci = compute_confidence_interval(adv_scores_k1)
    rand_ci = compute_confidence_interval(rand_scores_k1)
    print(f"  Adversarial N_1 CI: {adv_ci['mean']:.2f} "
          f"[{adv_ci['lower']:.2f}, {adv_ci['upper']:.2f}]")
    print(f"  Random N_1 CI:      {rand_ci['mean']:.2f} "
          f"[{rand_ci['lower']:.2f}, {rand_ci['upper']:.2f}]")

    # Compute per-query similarity analysis
    # For each query, compare max similarity to adversarial vs max sim to clean corpus
    adv_max_sims = []
    clean_max_sims = []
    for qid, q_emb in query_embeddings.items():
        adv_max = max(cosine_similarity(q_emb, adv_emb)
                      for adv_emb in adversarial_hubs.values())
        clean_max = max(cosine_similarity(q_emb, c_emb)
                        for c_emb in clean_corpus.values())
        adv_max_sims.append(adv_max)
        clean_max_sims.append(clean_max)

    ttest_result = paired_ttest(clean_max_sims, adv_max_sims)
    print(f"  Paired t-test (adv vs clean max similarity):")
    print(f"    t={ttest_result['t_statistic']:.3f}, "
          f"p={ttest_result['p_value']:.4f}, "
          f"significant={ttest_result['significant']}")
    print(f"    mean_diff={ttest_result['mean_diff']:+.4f} "
          f"({ttest_result['direction']})")

    # ===== Save results =====
    os.makedirs(RESULTS_DATA_DIR, exist_ok=True)
    output = {
        "experiment": "exp5_adversarial_hubness",
        "num_queries": len(QUERY_SET),
        "num_clean_corpus": len(clean_corpus),
        "num_adversarial_hubs": len(adversarial_hubs),
        "hubness_results": {
            "baseline": {k: {ki: int(vi) for ki, vi in v.items()}
                         for k, v in results["baseline"].items()},
            "adversarial": {k: {ki: int(vi) for ki, vi in v.items()}
                            for k, v in results["adversarial"].items()},
            "random": {k: {ki: int(vi) for ki, vi in v.items()}
                       for k, v in results["random"].items()},
        },
        "statistical_analysis": {
            "adversarial_ci_k1": adv_ci,
            "random_ci_k1": rand_ci,
            "ttest_adv_vs_clean": ttest_result,
        },
        "per_query_max_sims": {
            "adversarial": adv_max_sims,
            "clean": clean_max_sims,
        },
    }
    output_path = os.path.join(RESULTS_DATA_DIR, "exp5_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # ===== Plots =====
    os.makedirs(RESULTS_FIGURES_DIR, exist_ok=True)

    # Hubness distribution at k=1 with adversarial items
    plot_hubness_distribution(
        results["adversarial"]["k1"],
        adversarial_ids=set(adversarial_hubs.keys()),
        title="Exp5: Adversarial Hubness Distribution (k=1)",
        save_path=os.path.join(RESULTS_FIGURES_DIR, "exp5_hubness_k1.png"),
    )

    # Hubness distribution at k=3
    plot_hubness_distribution(
        results["adversarial"]["k3"],
        adversarial_ids=set(adversarial_hubs.keys()),
        title="Exp5: Adversarial Hubness Distribution (k=3)",
        save_path=os.path.join(RESULTS_FIGURES_DIR, "exp5_hubness_k3.png"),
    )

    # Similarity heatmap: queries vs adversarial hubs
    all_items = dict(adversarial_hubs)
    all_items.update(random_hubs)
    sim_labels = list(QUERY_SET.keys()) + list(all_items.keys())
    sim_embeddings = {}
    for qid, emb in query_embeddings.items():
        sim_embeddings[qid] = emb
    for iid, emb in all_items.items():
        sim_embeddings[iid] = emb

    sim_matrix = []
    for qid in QUERY_SET:
        row = []
        for iid in all_items:
            row.append(cosine_similarity(
                query_embeddings[qid], all_items[iid]
            ))
        sim_matrix.append(row)

    plot_similarity_heatmap(
        sim_matrix,
        labels=list(QUERY_SET.keys()),
        title="Exp5: Query-Hub Similarity (Adversarial vs Random)",
        save_path=os.path.join(RESULTS_FIGURES_DIR, "exp5_query_hub_heatmap.png"),
    )

    # Summary
    print(f"\n{'=' * 60}")
    print("Experiment 5 Summary: Adversarial Hubness")
    print(f"{'=' * 60}")
    for hub_name in adversarial_hubs:
        n1 = results["adversarial"]["k1"].get(hub_name, 0)
        n3 = results["adversarial"]["k3"].get(hub_name, 0)
        n5 = results["adversarial"]["k5"].get(hub_name, 0)
        print(f"  {hub_name}: N_1={n1}, N_3={n3}, N_5={n5}")
    max_clean_n1 = max(results["baseline"]["k1"].values()) if results["baseline"]["k1"] else 0
    max_adv_n1 = max(results["adversarial"]["k1"].get(h, 0) for h in adversarial_hubs)
    print(f"  Max clean hub N_1: {max_clean_n1}")
    print(f"  Max adversarial hub N_1: {max_adv_n1}")
    if max_clean_n1 > 0:
        print(f"  Hub score ratio: {max_adv_n1 / max_clean_n1:.2f}x")

    return output


if __name__ == "__main__":
    run_experiment()
