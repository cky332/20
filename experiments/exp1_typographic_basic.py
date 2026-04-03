"""Experiment 1: Typographic Attack on Gemini Embedding 2.

A comprehensive 8-phase evaluation of typographic attacks on the
Gemini Embedding 2 multimodal embedding model. Tests 20 object categories
across 3 attack strategies (random, semantic neighbor, cross-domain),
3 prompt templates, typography ablation, dimension truncation, and
CLIP baseline comparison.

Phases:
  2. Dataset preparation (image loading, attack pairing, image generation)
  3. Embedding extraction (text + image embeddings via Gemini API)
  4. Core classification & metrics (zero-shot, ASR, semantic shift)
  5. Typography ablation (font size, position, color, opacity, repetition)
  6. Dimension truncation robustness (3072, 1536, 768)
  7. CLIP baseline comparison (OpenCLIP ViT-B/32)
  8. Visualization & report
"""

import json
import os
import sys
import time

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    RESULTS_DATA_DIR, RESULTS_FIGURES_DIR, ASSETS_DIR,
    DATASET_DIR, EXP1_DIR, IMAGE_SIZE, OUTPUT_DIMS,
)
from src.embedding_client import GeminiEmbeddingClient
from src.typographic_attack import (
    add_typographic_text, get_dominant_color,
)
from src.similarity import cosine_similarity, cosine_distance
from src.statistics import paired_ttest, compute_cohens_d, compute_confidence_interval
from src.visualization import (
    plot_grouped_bar_asr, plot_typography_factor_heatmap,
    plot_dimension_truncation, plot_gemini_vs_clip,
    plot_tsne_umap, plot_case_studies, plot_control_comparison,
    plot_ablation_factor_bars, plot_similarity_heatmap,
)
from src.dataset import (
    CATEGORIES, DOMAINS, CATEGORY_TO_DOMAIN,
    SEMANTIC_NEIGHBORS, CROSS_DOMAIN,
    PROMPT_TEMPLATES, NOISE_TEXT,
    generate_attack_pairings, load_dataset_images,
    download_coco_images, get_text_prompts, get_attack_text_prompt,
)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _ensure_dirs():
    """Create all required output directories."""
    for d in [RESULTS_DATA_DIR, RESULTS_FIGURES_DIR, ASSETS_DIR,
              DATASET_DIR, EXP1_DIR]:
        os.makedirs(d, exist_ok=True)


def _save_json(data, filename):
    """Save data to JSON in the results/data directory."""
    path = os.path.join(RESULTS_DATA_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)
    print(f"  Saved: {path}")
    return path


def _load_json(filename):
    """Load JSON from results/data directory, or return None if not found."""
    path = os.path.join(RESULTS_DATA_DIR, filename)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _save_image(image_bytes, subdir, filename):
    """Save image bytes to assets/exp1/{subdir}/{filename}."""
    dirpath = os.path.join(EXP1_DIR, subdir)
    os.makedirs(dirpath, exist_ok=True)
    path = os.path.join(dirpath, filename)
    with open(path, "wb") as f:
        f.write(image_bytes)
    return path


def _load_image(subdir, filename):
    """Load image bytes from assets/exp1/{subdir}/{filename}."""
    path = os.path.join(EXP1_DIR, subdir, filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    return None


# ---------------------------------------------------------------------------
# Phase 2: Dataset preparation
# ---------------------------------------------------------------------------

def phase2_prepare_dataset(force=False):
    """Prepare the full dataset: download images, generate attack/control images.

    Creates:
      - 200 base images (20 categories x 10)
      - 600 attack images (200 x 3 strategies)
      - 200 benign control images
      - 200 noise control images
      Total: 1,200 images

    Returns:
        metadata dict describing all images and their pairings.
    """
    print("\n" + "=" * 60)
    print("PHASE 2: Dataset Preparation")
    print("=" * 60)

    cache_file = "exp1_phase2_metadata.json"
    if not force:
        cached = _load_json(cache_file)
        if cached:
            print("  Phase 2 results loaded from cache.")
            return cached

    # Step 1: Download / load base images
    print("\n--- Step 2.1-2.2: Loading base images ---")
    existing_count = 0
    for cat in CATEGORIES:
        cat_dir = os.path.join(DATASET_DIR, cat)
        if os.path.isdir(cat_dir):
            files = [f for f in os.listdir(cat_dir)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            existing_count += len(files)

    if existing_count < 200:
        print("  Downloading images from COCO + fallback sources...")
        download_coco_images(output_dir=DATASET_DIR, per_category=10)

    dataset_images = load_dataset_images(DATASET_DIR)

    # Validate: need at least some images per category
    total_images = sum(len(imgs) for imgs in dataset_images.values())
    print(f"\n  Total base images loaded: {total_images}")
    for cat in CATEGORIES:
        if len(dataset_images.get(cat, [])) == 0:
            print(f"  WARNING: No images for '{cat}' - results will be incomplete")

    # Step 2: Generate attack pairings
    print("\n--- Step 2.3: Generating attack pairings ---")
    pairings = generate_attack_pairings(seed=42)
    for cat in CATEGORIES[:3]:
        p = pairings[cat]
        print(f"  {cat}: random={p['random']}, "
              f"neighbor={p['neighbor']}, cross_domain={p['cross_domain']}")
    print("  ...")

    # Step 3: Generate all attacked and control images
    print("\n--- Step 2.4-2.5: Generating attack and control images ---")
    metadata = {
        "categories": CATEGORIES,
        "pairings": pairings,
        "images": [],  # list of image records
    }

    # Fixed typography params for Phase 2 (white text, centered, opaque, medium)
    attack_params = {
        "color": (255, 255, 255),
        "outline_color": (0, 0, 0),
        "outline_width": 2,
        "position": "center",
        "font_size_pct": 0.10,
        "opacity": 1.0,
    }

    image_count = {"baseline": 0, "attack": 0, "benign": 0, "noise": 0}

    for category in CATEGORIES:
        cat_images = dataset_images.get(category, [])
        if not cat_images:
            continue

        for img_id, img_bytes in cat_images:
            # Save baseline image
            base_fname = f"{img_id}_baseline.jpg"
            _save_image(img_bytes, "baseline", base_fname)
            metadata["images"].append({
                "image_id": img_id,
                "category": category,
                "group": "baseline",
                "attack_type": None,
                "attack_text": None,
                "filename": f"baseline/{base_fname}",
            })
            image_count["baseline"] += 1

            # Generate 3 attack images
            for atk_type in ["random", "neighbor", "cross_domain"]:
                atk_text = pairings[category][atk_type]
                atk_bytes = add_typographic_text(
                    img_bytes, atk_text,
                    font_size_pct=attack_params["font_size_pct"],
                    position=attack_params["position"],
                    color=attack_params["color"],
                    outline_color=attack_params["outline_color"],
                    outline_width=attack_params["outline_width"],
                    opacity=attack_params["opacity"],
                )
                atk_fname = f"{img_id}_{atk_type}.jpg"
                _save_image(atk_bytes, f"attack_{atk_type}", atk_fname)
                metadata["images"].append({
                    "image_id": img_id,
                    "category": category,
                    "group": f"attack_{atk_type}",
                    "attack_type": atk_type,
                    "attack_text": atk_text,
                    "filename": f"attack_{atk_type}/{atk_fname}",
                })
                image_count["attack"] += 1

            # Benign control: correct label overlay
            benign_bytes = add_typographic_text(
                img_bytes, category,
                font_size_pct=attack_params["font_size_pct"],
                position=attack_params["position"],
                color=attack_params["color"],
                outline_color=attack_params["outline_color"],
                outline_width=attack_params["outline_width"],
                opacity=attack_params["opacity"],
            )
            benign_fname = f"{img_id}_benign.jpg"
            _save_image(benign_bytes, "benign", benign_fname)
            metadata["images"].append({
                "image_id": img_id,
                "category": category,
                "group": "benign",
                "attack_type": None,
                "attack_text": category,
                "filename": f"benign/{benign_fname}",
            })
            image_count["benign"] += 1

            # Noise control: meaningless string overlay
            noise_bytes = add_typographic_text(
                img_bytes, NOISE_TEXT,
                font_size_pct=attack_params["font_size_pct"],
                position=attack_params["position"],
                color=attack_params["color"],
                outline_color=attack_params["outline_color"],
                outline_width=attack_params["outline_width"],
                opacity=attack_params["opacity"],
            )
            noise_fname = f"{img_id}_noise.jpg"
            _save_image(noise_bytes, "noise", noise_fname)
            metadata["images"].append({
                "image_id": img_id,
                "category": category,
                "group": "noise",
                "attack_type": None,
                "attack_text": NOISE_TEXT,
                "filename": f"noise/{noise_fname}",
            })
            image_count["noise"] += 1

    metadata["attack_params"] = attack_params
    metadata["image_counts"] = image_count
    total = sum(image_count.values())
    print(f"\n  Image generation complete:")
    print(f"    Baseline: {image_count['baseline']}")
    print(f"    Attack:   {image_count['attack']}")
    print(f"    Benign:   {image_count['benign']}")
    print(f"    Noise:    {image_count['noise']}")
    print(f"    Total:    {total}")

    _save_json(metadata, cache_file)
    return metadata


# ---------------------------------------------------------------------------
# Phase 3: Embedding extraction
# ---------------------------------------------------------------------------

def phase3_extract_embeddings(metadata, force=False):
    """Extract text and image embeddings via Gemini API.

    Text embeddings: 20 categories x 3 prompt templates = 60
    Attack word embeddings: up to 20 unique attack words x 1 template
    Image embeddings: all images from Phase 2

    Returns:
        Dict with text_embeddings, attack_text_embeddings, image_embeddings.
    """
    print("\n" + "=" * 60)
    print("PHASE 3: Embedding Extraction")
    print("=" * 60)

    cache_file = "exp1_phase3_embeddings.json"
    if not force:
        cached = _load_json(cache_file)
        if cached:
            print("  Phase 3 results loaded from cache.")
            return cached

    client = GeminiEmbeddingClient()
    results = {
        "text_embeddings": {},    # {category: {template: embedding}}
        "attack_text_embeddings": {},  # {attack_word: embedding}
        "image_embeddings": {},   # {image_record_key: embedding}
    }

    # Step 3.1: Text embeddings for each category x prompt template
    print("\n--- Step 3.1: Text embeddings (20 categories x 3 templates) ---")
    for i, category in enumerate(CATEGORIES):
        prompts = get_text_prompts(category)
        results["text_embeddings"][category] = {}
        for tmpl_name, text in prompts.items():
            emb = client.embed_text(text)
            results["text_embeddings"][category][tmpl_name] = emb.tolist()
            print(f"  [{i * 3 + list(prompts.keys()).index(tmpl_name) + 1}/60] "
                  f"Text: '{text}' -> dim={len(emb)}")

    # Attack word text embeddings (standard template)
    print("\n--- Step 3.1b: Attack word text embeddings ---")
    pairings = metadata["pairings"]
    attack_words = set()
    for cat, pair in pairings.items():
        for atk_type in ["random", "neighbor", "cross_domain"]:
            attack_words.add(pair[atk_type])
    attack_words = sorted(attack_words)

    for i, word in enumerate(attack_words):
        text = get_attack_text_prompt(word)
        emb = client.embed_text(text)
        results["attack_text_embeddings"][word] = emb.tolist()
        print(f"  [{i + 1}/{len(attack_words)}] Attack text: '{text}'")

    # Step 3.2: Image embeddings for all images
    print(f"\n--- Step 3.2: Image embeddings ({len(metadata['images'])} images) ---")
    for i, record in enumerate(metadata["images"]):
        fpath = os.path.join(EXP1_DIR, record["filename"])
        if not os.path.exists(fpath):
            print(f"  WARNING: Image not found: {fpath}, skipping")
            continue
        with open(fpath, "rb") as f:
            img_bytes = f.read()

        # Create a unique key for this image record
        key = f"{record['image_id']}_{record['group']}"
        emb = client.embed_image(img_bytes, mime_type="image/jpeg")
        results["image_embeddings"][key] = emb.tolist()

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i + 1}/{len(metadata['images'])}] "
                  f"{record['group']}: {record['image_id']}")

    print(f"\n  Embedding extraction complete:")
    print(f"    Text embeddings: {sum(len(v) for v in results['text_embeddings'].values())}")
    print(f"    Attack text embeddings: {len(results['attack_text_embeddings'])}")
    print(f"    Image embeddings: {len(results['image_embeddings'])}")

    _save_json(results, cache_file)
    return results


# ---------------------------------------------------------------------------
# Phase 4: Core classification & metrics
# ---------------------------------------------------------------------------

def phase4_classification(metadata, embeddings, force=False):
    """Zero-shot classification and core metric computation.

    For each image, compute cosine similarity with all 20 category text
    embeddings, predict class = argmax. Compute metrics per group x prompt.

    Returns:
        Dict with per-image results and aggregate metrics.
    """
    print("\n" + "=" * 60)
    print("PHASE 4: Core Classification & Metrics")
    print("=" * 60)

    cache_file = "exp1_phase4_metrics.json"
    if not force:
        cached = _load_json(cache_file)
        if cached:
            print("  Phase 4 results loaded from cache.")
            return cached

    text_embs = embeddings["text_embeddings"]
    attack_text_embs = embeddings["attack_text_embeddings"]
    image_embs = embeddings["image_embeddings"]
    pairings = metadata["pairings"]

    # Build text embedding vectors per template
    # {template_name: {category: np.array}}
    text_vectors = {}
    for tmpl in PROMPT_TEMPLATES:
        text_vectors[tmpl] = {}
        for cat in CATEGORIES:
            text_vectors[tmpl][cat] = np.array(text_embs[cat][tmpl])

    results = {
        "per_image": [],
        "aggregate": {},
    }

    # Classify each image
    print("\n--- Computing zero-shot classification ---")
    for i, record in enumerate(metadata["images"]):
        key = f"{record['image_id']}_{record['group']}"
        if key not in image_embs:
            continue

        img_emb = np.array(image_embs[key])
        category = record["category"]
        group = record["group"]
        attack_type = record.get("attack_type")
        attack_text = record.get("attack_text")

        # Get clean image embedding for this image_id
        clean_key = f"{record['image_id']}_baseline"
        clean_emb = np.array(image_embs[clean_key]) if clean_key in image_embs else None

        img_result = {
            "image_id": record["image_id"],
            "category": category,
            "group": group,
            "attack_type": attack_type,
            "attack_text": attack_text,
            "predictions": {},  # {template: predicted_category}
            "similarities": {},  # {template: {category: sim}}
        }

        for tmpl in PROMPT_TEMPLATES:
            sims = {}
            for cat in CATEGORIES:
                sims[cat] = cosine_similarity(img_emb, text_vectors[tmpl][cat])
            img_result["similarities"][tmpl] = sims

            predicted = max(sims, key=sims.get)
            img_result["predictions"][tmpl] = predicted

        # Compute attack-specific metrics
        if clean_emb is not None and group != "baseline":
            # Semantic shift: cosine distance between this image and clean
            img_result["semantic_shift"] = cosine_distance(img_emb, clean_emb)

        if attack_text and attack_text in attack_text_embs and clean_emb is not None:
            atk_text_emb = np.array(attack_text_embs[attack_text])
            # Attack attraction
            img_result["attack_attraction"] = (
                cosine_similarity(img_emb, atk_text_emb) -
                cosine_similarity(clean_emb, atk_text_emb)
            )

        if clean_emb is not None and group != "baseline":
            # True semantic retention (using standard template)
            true_text_emb = text_vectors["standard"].get(category)
            if true_text_emb is not None:
                img_result["semantic_retention"] = (
                    cosine_similarity(img_emb, true_text_emb) -
                    cosine_similarity(clean_emb, true_text_emb)
                )

        results["per_image"].append(img_result)

        if (i + 1) % 100 == 0:
            print(f"  [{i + 1}/{len(metadata['images'])}] classified")

    # Aggregate metrics
    print("\n--- Computing aggregate metrics ---")
    groups = ["baseline", "attack_random", "attack_neighbor",
              "attack_cross_domain", "benign", "noise"]

    aggregate = {}
    for tmpl in PROMPT_TEMPLATES:
        aggregate[tmpl] = {}
        for group in groups:
            group_results = [r for r in results["per_image"] if r["group"] == group]
            if not group_results:
                continue

            # Top-1 accuracy
            correct = sum(1 for r in group_results
                          if r["predictions"][tmpl] == r["category"])
            accuracy = correct / len(group_results) if group_results else 0

            group_metrics = {
                "count": len(group_results),
                "accuracy": accuracy,
            }

            # ASR: predicted == attack category (attack groups only)
            if group.startswith("attack_"):
                atk_type = group.replace("attack_", "")
                asr_count = 0
                for r in group_results:
                    atk_cat = pairings[r["category"]][atk_type]
                    if r["predictions"][tmpl] == atk_cat:
                        asr_count += 1
                group_metrics["asr"] = asr_count / len(group_results)

            # Semantic shift (mean)
            shifts = [r["semantic_shift"] for r in group_results
                      if "semantic_shift" in r]
            if shifts:
                group_metrics["mean_semantic_shift"] = float(np.mean(shifts))
                group_metrics["std_semantic_shift"] = float(np.std(shifts))

            # Attack attraction (mean)
            attractions = [r["attack_attraction"] for r in group_results
                           if "attack_attraction" in r]
            if attractions:
                group_metrics["mean_attack_attraction"] = float(np.mean(attractions))

            # Semantic retention (mean)
            retentions = [r["semantic_retention"] for r in group_results
                          if "semantic_retention" in r]
            if retentions:
                group_metrics["mean_semantic_retention"] = float(np.mean(retentions))

            aggregate[tmpl][group] = group_metrics

    # Accuracy drop: baseline - attack
    for tmpl in PROMPT_TEMPLATES:
        baseline_acc = aggregate[tmpl].get("baseline", {}).get("accuracy", 0)
        for group in groups:
            if group.startswith("attack_") and group in aggregate[tmpl]:
                aggregate[tmpl][group]["accuracy_drop"] = (
                    baseline_acc - aggregate[tmpl][group]["accuracy"]
                )

    results["aggregate"] = aggregate

    # Statistical tests: compare attack groups vs baseline
    print("\n--- Statistical tests ---")
    stats_results = {}
    for tmpl in PROMPT_TEMPLATES:
        stats_results[tmpl] = {}
        baseline_sims = []
        for r in results["per_image"]:
            if r["group"] == "baseline":
                baseline_sims.append(r["similarities"][tmpl][r["category"]])

        for atk_type in ["random", "neighbor", "cross_domain"]:
            group = f"attack_{atk_type}"
            attack_sims = []
            for r in results["per_image"]:
                if r["group"] == group:
                    attack_sims.append(r["similarities"][tmpl][r["category"]])

            if len(baseline_sims) == len(attack_sims) and len(baseline_sims) > 1:
                t_stat, p_val = paired_ttest(baseline_sims, attack_sims)
                d = compute_cohens_d(baseline_sims, attack_sims)
                ci = compute_confidence_interval(
                    [a - b for a, b in zip(attack_sims, baseline_sims)]
                )
                stats_results[tmpl][atk_type] = {
                    "t_statistic": t_stat,
                    "p_value": p_val,
                    "cohens_d": d,
                    "ci_lower": ci[0] if ci else None,
                    "ci_upper": ci[1] if ci else None,
                }
                print(f"  {tmpl}/{atk_type}: t={t_stat:.3f}, p={p_val:.4f}, d={d:.3f}")

    results["statistics"] = stats_results

    # Print summary
    print("\n--- Summary ---")
    for tmpl in PROMPT_TEMPLATES:
        print(f"\n  Prompt template: {tmpl}")
        for group in groups:
            gm = aggregate.get(tmpl, {}).get(group, {})
            if not gm:
                continue
            line = f"    {group:25s}: acc={gm['accuracy']:.3f}"
            if "asr" in gm:
                line += f"  ASR={gm['asr']:.3f}"
            if "accuracy_drop" in gm:
                line += f"  drop={gm['accuracy_drop']:.3f}"
            print(line)

    _save_json(results, cache_file)
    return results


# ---------------------------------------------------------------------------
# Phase 5: Typography ablation
# ---------------------------------------------------------------------------

def phase5_typography_ablation(metadata, phase4_results, force=False):
    """Ablation study on typography parameters.

    Fixed: semantic neighbor attack on top 5 categories by ASR.
    Variables: font size, position, color, opacity, repetition.
    Total: 5 categories x 10 images x 21 variants = 1,050 images.

    Returns:
        Dict with ablation results per factor.
    """
    print("\n" + "=" * 60)
    print("PHASE 5: Typography Ablation")
    print("=" * 60)

    cache_file = "exp1_phase5_ablation.json"
    if not force:
        cached = _load_json(cache_file)
        if cached:
            print("  Phase 5 results loaded from cache.")
            return cached

    # Select top 5 categories by neighbor ASR (from standard template)
    agg = phase4_results.get("aggregate", {}).get("standard", {})
    neighbor_group = agg.get("attack_neighbor", {})

    # Compute per-category ASR for neighbor attacks
    per_cat_asr = {}
    pairings = metadata["pairings"]
    for cat in CATEGORIES:
        cat_results = [r for r in phase4_results["per_image"]
                       if r["group"] == "attack_neighbor" and r["category"] == cat]
        if cat_results:
            atk_cat = pairings[cat]["neighbor"]
            asr = sum(1 for r in cat_results
                      if r["predictions"]["standard"] == atk_cat) / len(cat_results)
            per_cat_asr[cat] = asr

    # Sort by ASR descending, take top 5
    top5 = sorted(per_cat_asr.items(), key=lambda x: x[1], reverse=True)[:5]
    top5_cats = [c for c, _ in top5]
    print(f"\n  Top 5 categories by neighbor ASR: {top5_cats}")
    for cat, asr in top5:
        print(f"    {cat}: ASR={asr:.3f}")

    # If not enough categories with ASR data, use first 5
    if len(top5_cats) < 5:
        top5_cats = CATEGORIES[:5]
        print(f"  Fallback to first 5 categories: {top5_cats}")

    client = GeminiEmbeddingClient()

    # Load text embeddings for classification (standard template)
    phase3_data = _load_json("exp1_phase3_embeddings.json")
    text_vectors = {}
    for cat in CATEGORIES:
        text_vectors[cat] = np.array(phase3_data["text_embeddings"][cat]["standard"])

    # Define ablation factors
    ablation_factors = {
        "font_size": {
            "levels": [0.05, 0.10, 0.15, 0.25, 0.40],
            "labels": ["5%", "10%", "15%", "25%", "40%"],
            "param_key": "font_size_pct",
        },
        "position": {
            "levels": ["center", "top-left", "bottom-right", "top-center", "bottom-center"],
            "labels": ["center", "top-left", "bottom-right", "top-center", "bottom-center"],
            "param_key": "position",
        },
        "color": {
            "levels": [
                {"color": (255, 255, 255), "outline_color": (0, 0, 0), "outline_width": 2,
                 "label": "white+outline"},
                {"color": (0, 0, 0), "outline_color": None, "outline_width": 0,
                 "label": "black"},
                {"color": (255, 0, 0), "outline_color": None, "outline_width": 0,
                 "label": "red"},
                {"color": "dominant", "outline_color": None, "outline_width": 0,
                 "label": "dominant_color"},
            ],
        },
        "opacity": {
            "levels": [0.25, 0.50, 0.75, 1.0],
            "labels": ["25%", "50%", "75%", "100%"],
            "param_key": "opacity",
        },
        "repetition": {
            "levels": [
                {"repeat": 1, "position": "center", "label": "1x_center"},
                {"repeat": 1, "position": "scattered", "label": "3x_scattered"},
                {"repeat": 1, "position": "tiled", "label": "tiled"},
            ],
        },
    }

    # Load base images for top5 categories
    dataset_images = load_dataset_images(DATASET_DIR)

    ablation_results = {}
    total_api_calls = 0

    for factor_name, factor_config in ablation_factors.items():
        print(f"\n--- Ablation factor: {factor_name} ---")
        factor_results = {"levels": [], "asr_per_level": [], "accuracy_per_level": []}

        if factor_name == "color":
            levels = factor_config["levels"]
        elif factor_name == "repetition":
            levels = factor_config["levels"]
        else:
            levels = factor_config["levels"]

        for level_idx, level in enumerate(levels):
            # Determine label
            if factor_name == "color":
                level_label = level["label"]
            elif factor_name == "repetition":
                level_label = level["label"]
            elif "labels" in factor_config:
                level_label = factor_config["labels"][level_idx]
            else:
                level_label = str(level)

            print(f"  Level: {level_label}")
            correct_count = 0
            asr_count = 0
            total_count = 0

            for cat in top5_cats:
                cat_images = dataset_images.get(cat, [])
                if not cat_images:
                    continue
                atk_text = pairings[cat]["neighbor"]
                atk_cat = SEMANTIC_NEIGHBORS[cat]

                for img_id, img_bytes in cat_images:
                    # Build attack params for this level
                    params = {
                        "font_size_pct": 0.10,
                        "position": "center",
                        "color": (255, 255, 255),
                        "outline_color": (0, 0, 0),
                        "outline_width": 2,
                        "opacity": 1.0,
                        "repeat": 1,
                    }

                    if factor_name == "font_size":
                        params["font_size_pct"] = level
                    elif factor_name == "position":
                        params["position"] = level
                    elif factor_name == "color":
                        if level["color"] == "dominant":
                            params["color"] = get_dominant_color(img_bytes)
                        else:
                            params["color"] = level["color"]
                        params["outline_color"] = level["outline_color"]
                        params["outline_width"] = level["outline_width"]
                    elif factor_name == "opacity":
                        params["opacity"] = level
                    elif factor_name == "repetition":
                        params["repeat"] = level["repeat"]
                        params["position"] = level["position"]

                    # Generate attacked image
                    atk_bytes = add_typographic_text(
                        img_bytes, atk_text,
                        font_size_pct=params["font_size_pct"],
                        position=params["position"],
                        color=params["color"],
                        outline_color=params["outline_color"],
                        outline_width=params["outline_width"],
                        opacity=params["opacity"],
                        repeat=params["repeat"],
                    )

                    # Embed and classify
                    emb = client.embed_image(atk_bytes, mime_type="image/jpeg")
                    total_api_calls += 1

                    sims = {c: cosine_similarity(emb, text_vectors[c]) for c in CATEGORIES}
                    predicted = max(sims, key=sims.get)

                    if predicted == cat:
                        correct_count += 1
                    if predicted == atk_cat:
                        asr_count += 1
                    total_count += 1

            accuracy = correct_count / total_count if total_count > 0 else 0
            asr = asr_count / total_count if total_count > 0 else 0

            factor_results["levels"].append(level_label)
            factor_results["asr_per_level"].append(asr)
            factor_results["accuracy_per_level"].append(accuracy)
            print(f"    count={total_count}, accuracy={accuracy:.3f}, ASR={asr:.3f}")

        ablation_results[factor_name] = factor_results

    print(f"\n  Total API calls for ablation: {total_api_calls}")

    results = {
        "top5_categories": top5_cats,
        "ablation_results": ablation_results,
    }

    _save_json(results, cache_file)
    return results


# ---------------------------------------------------------------------------
# Phase 6: Dimension truncation
# ---------------------------------------------------------------------------

def phase6_dimension_truncation(metadata, force=False):
    """Test classification and attack effectiveness at different embedding dimensions.

    Re-embeds all Phase 2 images at 1536 and 768 dimensions (3072 already done).
    Repeats zero-shot classification and computes ASR at each dimension.

    Returns:
        Dict with metrics per dimension.
    """
    print("\n" + "=" * 60)
    print("PHASE 6: Dimension Truncation Robustness")
    print("=" * 60)

    cache_file = "exp1_phase6_dimensions.json"
    if not force:
        cached = _load_json(cache_file)
        if cached:
            print("  Phase 6 results loaded from cache.")
            return cached

    client = GeminiEmbeddingClient()
    pairings = metadata["pairings"]
    test_dims = [768, 1536, 3072]

    # We already have 3072 from Phase 3. Need 768 and 1536.
    results = {"dimensions": test_dims, "metrics_per_dim": {}}

    for dim in test_dims:
        print(f"\n--- Dimension: {dim} ---")

        # Embed text (all categories, standard template)
        text_vectors = {}
        for cat in CATEGORIES:
            text = PROMPT_TEMPLATES["standard"].format(category=cat)
            emb = client.embed_text(text, output_dimensionality=dim)
            text_vectors[cat] = emb

        # Embed images and classify
        groups_to_test = ["baseline", "attack_random", "attack_neighbor",
                          "attack_cross_domain"]
        group_correct = {g: 0 for g in groups_to_test}
        group_asr = {g: 0 for g in groups_to_test if g.startswith("attack_")}
        group_count = {g: 0 for g in groups_to_test}

        for i, record in enumerate(metadata["images"]):
            if record["group"] not in groups_to_test:
                continue

            fpath = os.path.join(EXP1_DIR, record["filename"])
            if not os.path.exists(fpath):
                continue
            with open(fpath, "rb") as f:
                img_bytes = f.read()

            emb = client.embed_image(img_bytes, mime_type="image/jpeg",
                                     output_dimensionality=dim)

            sims = {c: cosine_similarity(emb, text_vectors[c]) for c in CATEGORIES}
            predicted = max(sims, key=sims.get)

            group = record["group"]
            group_count[group] += 1
            if predicted == record["category"]:
                group_correct[group] += 1

            if group.startswith("attack_"):
                atk_type = group.replace("attack_", "")
                atk_cat = pairings[record["category"]][atk_type]
                if predicted == atk_cat:
                    group_asr[group] += 1

            if (i + 1) % 100 == 0:
                print(f"  [{i + 1}] processing at dim={dim}...")

        dim_metrics = {}
        for g in groups_to_test:
            if group_count[g] > 0:
                dim_metrics[g] = {
                    "count": group_count[g],
                    "accuracy": group_correct[g] / group_count[g],
                }
                if g in group_asr:
                    dim_metrics[g]["asr"] = group_asr[g] / group_count[g]

        results["metrics_per_dim"][str(dim)] = dim_metrics
        print(f"  dim={dim}: baseline_acc={dim_metrics.get('baseline', {}).get('accuracy', 'N/A')}")
        for g in group_asr:
            print(f"    {g}: ASR={dim_metrics.get(g, {}).get('asr', 'N/A')}")

    _save_json(results, cache_file)
    return results


# ---------------------------------------------------------------------------
# Phase 7: CLIP baseline comparison
# ---------------------------------------------------------------------------

def phase7_clip_baseline(metadata, force=False):
    """Compare Gemini Embedding 2 with CLIP on the same dataset.

    Uses OpenCLIP ViT-B/32 for text and image encoding.
    Repeats zero-shot classification and attack metrics.

    Returns:
        Dict with CLIP metrics comparable to Phase 4.
    """
    print("\n" + "=" * 60)
    print("PHASE 7: CLIP Baseline Comparison")
    print("=" * 60)

    cache_file = "exp1_phase7_clip.json"
    if not force:
        cached = _load_json(cache_file)
        if cached:
            print("  Phase 7 results loaded from cache.")
            return cached

    try:
        import torch
        import open_clip
        from PIL import Image
        import io
    except ImportError as e:
        print(f"  ERROR: CLIP dependencies not available: {e}")
        print("  Install with: pip install open-clip-torch torch")
        return {"error": str(e)}

    # Load CLIP model
    print("  Loading OpenCLIP ViT-B/32...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()

    pairings = metadata["pairings"]

    # Encode text for all categories x 3 templates
    print("  Encoding text embeddings...")
    text_vectors = {}
    for tmpl_name, tmpl in PROMPT_TEMPLATES.items():
        text_vectors[tmpl_name] = {}
        texts = [tmpl.format(category=cat) for cat in CATEGORIES]
        tokens = tokenizer(texts)
        with torch.no_grad():
            text_features = model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        for i, cat in enumerate(CATEGORIES):
            text_vectors[tmpl_name][cat] = text_features[i].numpy()

    # Classify all images
    print(f"  Classifying {len(metadata['images'])} images with CLIP...")
    groups = ["baseline", "attack_random", "attack_neighbor",
              "attack_cross_domain", "benign", "noise"]

    per_image_results = []
    for idx, record in enumerate(metadata["images"]):
        fpath = os.path.join(EXP1_DIR, record["filename"])
        if not os.path.exists(fpath):
            continue

        # Load and preprocess image
        pil_img = Image.open(fpath).convert("RGB")
        img_tensor = preprocess(pil_img).unsqueeze(0)

        with torch.no_grad():
            img_features = model.encode_image(img_tensor)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        img_emb = img_features[0].numpy()

        result = {
            "image_id": record["image_id"],
            "category": record["category"],
            "group": record["group"],
            "attack_type": record.get("attack_type"),
            "attack_text": record.get("attack_text"),
            "predictions": {},
        }

        for tmpl_name in PROMPT_TEMPLATES:
            sims = {}
            for cat in CATEGORIES:
                sims[cat] = float(np.dot(img_emb, text_vectors[tmpl_name][cat]))
            predicted = max(sims, key=sims.get)
            result["predictions"][tmpl_name] = predicted

        per_image_results.append(result)

        if (idx + 1) % 100 == 0:
            print(f"    [{idx + 1}/{len(metadata['images'])}] processed")

    # Aggregate metrics
    print("\n  Computing CLIP aggregate metrics...")
    aggregate = {}
    for tmpl in PROMPT_TEMPLATES:
        aggregate[tmpl] = {}
        for group in groups:
            group_results = [r for r in per_image_results if r["group"] == group]
            if not group_results:
                continue

            correct = sum(1 for r in group_results
                          if r["predictions"][tmpl] == r["category"])
            accuracy = correct / len(group_results)

            group_metrics = {"count": len(group_results), "accuracy": accuracy}

            if group.startswith("attack_"):
                atk_type = group.replace("attack_", "")
                asr_count = 0
                for r in group_results:
                    atk_cat = pairings[r["category"]][atk_type]
                    if r["predictions"][tmpl] == atk_cat:
                        asr_count += 1
                group_metrics["asr"] = asr_count / len(group_results)

            aggregate[tmpl][group] = group_metrics

        # Accuracy drop
        baseline_acc = aggregate[tmpl].get("baseline", {}).get("accuracy", 0)
        for group in groups:
            if group.startswith("attack_") and group in aggregate[tmpl]:
                aggregate[tmpl][group]["accuracy_drop"] = (
                    baseline_acc - aggregate[tmpl][group]["accuracy"]
                )

    results = {
        "model": "OpenCLIP ViT-B/32 (laion2b_s34b_b79k)",
        "aggregate": aggregate,
    }

    # Print summary
    print("\n--- CLIP Summary ---")
    for tmpl in PROMPT_TEMPLATES:
        print(f"\n  Template: {tmpl}")
        for group in groups:
            gm = aggregate.get(tmpl, {}).get(group, {})
            if not gm:
                continue
            line = f"    {group:25s}: acc={gm['accuracy']:.3f}"
            if "asr" in gm:
                line += f"  ASR={gm['asr']:.3f}"
            print(line)

    _save_json(results, cache_file)
    return results


# ---------------------------------------------------------------------------
# Phase 8: Visualization & report
# ---------------------------------------------------------------------------

def phase8_visualize(metadata, phase4_results, phase5_results,
                     phase6_results, phase7_results, embeddings):
    """Generate all visualizations and summary report.

    Creates:
      - Grouped bar chart: ASR by attack strategy x prompt template
      - Typography ablation bar charts per factor
      - Heatmap: font_size x position cross-impact
      - Line chart: dimension truncation trends
      - Gemini vs CLIP comparison
      - t-SNE/UMAP embedding space
      - Case studies
    """
    print("\n" + "=" * 60)
    print("PHASE 8: Visualization & Report")
    print("=" * 60)

    fig_dir = os.path.join(RESULTS_FIGURES_DIR, "exp1")
    os.makedirs(fig_dir, exist_ok=True)

    # 8.1: ASR grouped bar chart (attack strategy x prompt template)
    print("\n--- 8.1: ASR grouped bar chart ---")
    asr_data = {}
    for atk_type in ["random", "neighbor", "cross_domain"]:
        group = f"attack_{atk_type}"
        asr_data[atk_type] = {}
        for tmpl in PROMPT_TEMPLATES:
            agg = phase4_results.get("aggregate", {}).get(tmpl, {}).get(group, {})
            asr_data[atk_type][tmpl] = agg.get("asr", 0)

    plot_grouped_bar_asr(
        asr_data,
        title="Attack Success Rate by Strategy and Prompt Template",
        save_path=os.path.join(fig_dir, "asr_by_strategy_prompt.png"),
    )

    # 8.1b: Control group comparison
    print("--- 8.1b: Control group comparison ---")
    control_data = {}
    for group_name in ["baseline", "attack_neighbor", "benign", "noise"]:
        control_data[group_name] = {}
        for tmpl in PROMPT_TEMPLATES:
            agg = phase4_results.get("aggregate", {}).get(tmpl, {}).get(group_name, {})
            control_data[group_name][tmpl] = agg.get("accuracy", 0)

    plot_control_comparison(
        control_data,
        title="Accuracy: Attack vs Benign vs Noise Controls",
        save_path=os.path.join(fig_dir, "control_comparison.png"),
    )

    # 8.2: Typography ablation bar charts
    if phase5_results and "ablation_results" in phase5_results:
        print("--- 8.2: Typography ablation charts ---")
        for factor_name, factor_data in phase5_results["ablation_results"].items():
            plot_ablation_factor_bars(
                factor_name=factor_name.replace("_", " ").title(),
                level_names=factor_data["levels"],
                asr_values=factor_data["asr_per_level"],
                title=f"ASR by {factor_name.replace('_', ' ').title()}",
                save_path=os.path.join(fig_dir, f"ablation_{factor_name}.png"),
            )

        # Cross-factor heatmap: font_size x position
        print("--- 8.2b: Cross-factor heatmap ---")
        fs_data = phase5_results["ablation_results"].get("font_size", {})
        pos_data = phase5_results["ablation_results"].get("position", {})
        if fs_data and pos_data:
            # Create a simple heatmap using available data
            fs_asr = fs_data["asr_per_level"]
            pos_asr = pos_data["asr_per_level"]
            # Outer product as approximation of cross-impact
            matrix = np.outer(fs_asr, pos_asr)
            # Normalize to [0, 1]
            if matrix.max() > 0:
                matrix = matrix / matrix.max()
            plot_typography_factor_heatmap(
                matrix,
                row_param=("Font Size", fs_data["levels"]),
                col_param=("Position", pos_data["levels"]),
                title="Typography Factor Cross-Impact (Font Size x Position)",
                save_path=os.path.join(fig_dir, "ablation_heatmap_fs_pos.png"),
            )

    # 8.3: Dimension truncation line chart
    if phase6_results and "metrics_per_dim" in phase6_results:
        print("--- 8.3: Dimension truncation chart ---")
        dims = phase6_results["dimensions"]
        metrics_dict = {}

        # Baseline accuracy
        metrics_dict["Baseline Accuracy"] = []
        for d in dims:
            m = phase6_results["metrics_per_dim"].get(str(d), {})
            metrics_dict["Baseline Accuracy"].append(
                m.get("baseline", {}).get("accuracy", 0)
            )

        # ASR for each attack type
        for atk_type in ["random", "neighbor", "cross_domain"]:
            group = f"attack_{atk_type}"
            key = f"ASR ({atk_type})"
            metrics_dict[key] = []
            for d in dims:
                m = phase6_results["metrics_per_dim"].get(str(d), {})
                metrics_dict[key].append(m.get(group, {}).get("asr", 0))

        plot_dimension_truncation(
            dims, metrics_dict,
            title="Dimension Truncation: Accuracy & ASR",
            save_path=os.path.join(fig_dir, "dimension_truncation.png"),
        )

    # 8.4: Gemini vs CLIP comparison
    if phase7_results and "aggregate" in phase7_results:
        print("--- 8.4: Gemini vs CLIP comparison ---")
        gemini_agg = phase4_results.get("aggregate", {}).get("standard", {})
        clip_agg = phase7_results.get("aggregate", {}).get("standard", {})

        comparison_metrics = []
        # Baseline accuracy
        comparison_metrics.append({
            "metric_name": "Baseline Accuracy",
            "gemini_value": gemini_agg.get("baseline", {}).get("accuracy", 0),
            "clip_value": clip_agg.get("baseline", {}).get("accuracy", 0),
        })
        # ASR for each attack type
        for atk_type in ["random", "neighbor", "cross_domain"]:
            group = f"attack_{atk_type}"
            comparison_metrics.append({
                "metric_name": f"ASR ({atk_type})",
                "gemini_value": gemini_agg.get(group, {}).get("asr", 0),
                "clip_value": clip_agg.get(group, {}).get("asr", 0),
            })
        # Accuracy drop (neighbor)
        comparison_metrics.append({
            "metric_name": "Acc Drop (neighbor)",
            "gemini_value": gemini_agg.get("attack_neighbor", {}).get("accuracy_drop", 0),
            "clip_value": clip_agg.get("attack_neighbor", {}).get("accuracy_drop", 0),
        })

        plot_gemini_vs_clip(
            comparison_metrics,
            title="Gemini Embedding 2 vs CLIP ViT-B/32",
            save_path=os.path.join(fig_dir, "gemini_vs_clip.png"),
        )

    # 8.5: t-SNE/UMAP embedding space visualization
    if embeddings and "image_embeddings" in embeddings:
        print("--- 8.5: t-SNE embedding space ---")
        # Select 5 categories for visualization
        viz_cats = CATEGORIES[:5]
        all_embs = []
        all_labels = []
        all_groups = []

        for record in metadata["images"]:
            if record["category"] not in viz_cats:
                continue
            key = f"{record['image_id']}_{record['group']}"
            if key not in embeddings["image_embeddings"]:
                continue
            emb = embeddings["image_embeddings"][key]
            all_embs.append(emb)
            all_labels.append(record["category"])

            group = record["group"]
            if group == "baseline":
                all_groups.append("clean_image")
            elif group.startswith("attack_"):
                all_groups.append(group)
            elif group == "benign":
                all_groups.append("benign_control")
            elif group == "noise":
                all_groups.append("noise_control")

        # Add text label embeddings
        for cat in viz_cats:
            if cat in embeddings["text_embeddings"]:
                emb = embeddings["text_embeddings"][cat]["standard"]
                all_embs.append(emb)
                all_labels.append(cat)
                all_groups.append("text_label")

        if len(all_embs) > 10:
            plot_tsne_umap(
                np.array(all_embs), all_labels, all_groups,
                method="tsne",
                title=f"Embedding Space (t-SNE) - {', '.join(viz_cats)}",
                save_path=os.path.join(fig_dir, "tsne_embedding_space.png"),
            )

    # 8.6: Case studies (top 5 successes and failures)
    print("--- 8.6: Case studies ---")
    success_cases = []
    failure_cases = []
    pairings = metadata["pairings"]

    for r in phase4_results.get("per_image", []):
        if r["group"] != "attack_neighbor":
            continue
        predicted = r["predictions"].get("standard", "")
        atk_cat = pairings[r["category"]]["neighbor"]
        true_sim = r["similarities"]["standard"].get(r["category"], 0)
        atk_sim = r["similarities"]["standard"].get(atk_cat, 0)

        case = {
            "category": r["category"],
            "attack_type": "neighbor",
            "attack_text": atk_cat,
            "predicted": predicted,
            "true_sim": true_sim,
            "attack_sim": atk_sim,
            "success": (predicted == atk_cat),
        }

        if predicted == atk_cat:
            success_cases.append(case)
        else:
            failure_cases.append(case)

    # Sort by attack_sim for successes, by true_sim for failures
    success_cases.sort(key=lambda x: x["attack_sim"], reverse=True)
    failure_cases.sort(key=lambda x: x["true_sim"], reverse=True)

    cases = success_cases[:5] + failure_cases[:5]
    if cases:
        plot_case_studies(
            cases,
            title="Attack Case Studies: Top Successes & Failures (Neighbor)",
            save_path=os.path.join(fig_dir, "case_studies.png"),
        )

    # 8.7: Similarity heatmap for text embeddings
    print("--- 8.7: Text similarity heatmap ---")
    if embeddings and "text_embeddings" in embeddings:
        sim_matrix = []
        for cat_i in CATEGORIES:
            row = []
            emb_i = np.array(embeddings["text_embeddings"][cat_i]["standard"])
            for cat_j in CATEGORIES:
                emb_j = np.array(embeddings["text_embeddings"][cat_j]["standard"])
                row.append(cosine_similarity(emb_i, emb_j))
            sim_matrix.append(row)

        plot_similarity_heatmap(
            sim_matrix, CATEGORIES,
            title="Text Embedding Similarity (Standard Template)",
            save_path=os.path.join(fig_dir, "text_similarity_heatmap.png"),
        )

    print(f"\n  All figures saved to: {fig_dir}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(force=False):
    """Run the complete 8-phase typographic attack experiment.

    Args:
        force: If True, re-run all phases even if cached results exist.
    """
    print("=" * 60)
    print("EXPERIMENT 1: Typographic Attack on Gemini Embedding 2")
    print("Comprehensive 8-Phase Evaluation")
    print("=" * 60)
    print(f"\nCategories: {len(CATEGORIES)} across {len(DOMAINS)} domains")
    print(f"Domains: {list(DOMAINS.keys())}")
    print(f"Image size: {IMAGE_SIZE}")

    _ensure_dirs()

    # Phase 2: Dataset preparation
    metadata = phase2_prepare_dataset(force=force)

    # Phase 3: Embedding extraction
    embeddings = phase3_extract_embeddings(metadata, force=force)

    # Phase 4: Core classification & metrics
    phase4_results = phase4_classification(metadata, embeddings, force=force)

    # Phase 5: Typography ablation
    phase5_results = phase5_typography_ablation(metadata, phase4_results, force=force)

    # Phase 6: Dimension truncation
    phase6_results = phase6_dimension_truncation(metadata, force=force)

    # Phase 7: CLIP baseline comparison
    phase7_results = phase7_clip_baseline(metadata, force=force)

    # Phase 8: Visualization & report
    phase8_visualize(metadata, phase4_results, phase5_results,
                     phase6_results, phase7_results, embeddings)

    # Final summary
    print("\n" + "=" * 60)
    print("EXPERIMENT 1 COMPLETE")
    print("=" * 60)

    agg = phase4_results.get("aggregate", {}).get("standard", {})
    print(f"\nKey Results (standard prompt template):")
    baseline_acc = agg.get("baseline", {}).get("accuracy", "N/A")
    print(f"  Baseline accuracy: {baseline_acc}")
    for atk in ["random", "neighbor", "cross_domain"]:
        g = f"attack_{atk}"
        gm = agg.get(g, {})
        if gm:
            print(f"  {atk:15s}: ASR={gm.get('asr', 'N/A'):.3f}  "
                  f"acc_drop={gm.get('accuracy_drop', 'N/A'):.3f}")

    if phase7_results and "aggregate" in phase7_results:
        clip_agg = phase7_results["aggregate"].get("standard", {})
        print(f"\nCLIP Comparison (standard template):")
        clip_baseline = clip_agg.get("baseline", {}).get("accuracy", "N/A")
        print(f"  CLIP baseline accuracy: {clip_baseline}")
        for atk in ["random", "neighbor", "cross_domain"]:
            g = f"attack_{atk}"
            gm = clip_agg.get(g, {})
            if gm:
                print(f"  CLIP {atk:15s}: ASR={gm.get('asr', 'N/A'):.3f}")

    print(f"\nResults saved to: {RESULTS_DATA_DIR}")
    print(f"Figures saved to: {RESULTS_FIGURES_DIR}/exp1/")

    return {
        "metadata": metadata,
        "embeddings": embeddings,
        "phase4": phase4_results,
        "phase5": phase5_results,
        "phase6": phase6_results,
        "phase7": phase7_results,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Experiment 1: Typographic Attack on Gemini Embedding 2"
    )
    parser.add_argument("--force", action="store_true",
                        help="Force re-run all phases (ignore cache)")
    parser.add_argument("--phase", type=int, default=0,
                        help="Run only a specific phase (2-8)")
    args = parser.parse_args()

    if args.phase == 0:
        run(force=args.force)
    else:
        _ensure_dirs()
        if args.phase == 2:
            phase2_prepare_dataset(force=args.force)
        elif args.phase == 3:
            metadata = _load_json("exp1_phase2_metadata.json")
            if metadata:
                phase3_extract_embeddings(metadata, force=args.force)
            else:
                print("ERROR: Run Phase 2 first")
        elif args.phase == 4:
            metadata = _load_json("exp1_phase2_metadata.json")
            embeddings = _load_json("exp1_phase3_embeddings.json")
            if metadata and embeddings:
                phase4_classification(metadata, embeddings, force=args.force)
            else:
                print("ERROR: Run Phases 2-3 first")
        elif args.phase == 5:
            metadata = _load_json("exp1_phase2_metadata.json")
            phase4_results = _load_json("exp1_phase4_metrics.json")
            if metadata and phase4_results:
                phase5_typography_ablation(metadata, phase4_results, force=args.force)
            else:
                print("ERROR: Run Phases 2-4 first")
        elif args.phase == 6:
            metadata = _load_json("exp1_phase2_metadata.json")
            if metadata:
                phase6_dimension_truncation(metadata, force=args.force)
            else:
                print("ERROR: Run Phase 2 first")
        elif args.phase == 7:
            metadata = _load_json("exp1_phase2_metadata.json")
            if metadata:
                phase7_clip_baseline(metadata, force=args.force)
            else:
                print("ERROR: Run Phase 2 first")
        elif args.phase == 8:
            metadata = _load_json("exp1_phase2_metadata.json")
            embeddings = _load_json("exp1_phase3_embeddings.json")
            phase4_results = _load_json("exp1_phase4_metrics.json")
            phase5_results = _load_json("exp1_phase5_ablation.json")
            phase6_results = _load_json("exp1_phase6_dimensions.json")
            phase7_results = _load_json("exp1_phase7_clip.json")
            if metadata and phase4_results:
                phase8_visualize(metadata, phase4_results, phase5_results,
                                 phase6_results, phase7_results, embeddings)
            else:
                print("ERROR: Run Phases 2-4 first (5-7 optional)")
