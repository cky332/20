"""Experiment 6: Cross-Modal Transfer Attack on Gemini Embedding 2.

Tests whether attacks designed for one retrieval direction (image→text)
also affect other directions (text→image, multimodal→image) in the
unified embedding space.

If Gemini Embedding 2 truly unifies all modalities into a single space,
an attack crafted for one direction should transfer to other directions.

Metrics:
- Rank change of attacked items across retrieval directions
- Transfer rate: fraction of attacks succeeding in a different direction
- Similarity shift comparison across directions
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import RESULTS_DATA_DIR, RESULTS_FIGURES_DIR, ASSETS_DIR
from src.embedding_client import GeminiEmbeddingClient
from src.image_generator import SOURCE_IMAGES
from src.typographic_attack import add_typographic_text
from src.similarity import cosine_similarity, rank_documents
from src.visualization import plot_transfer_matrix


# Attack configurations: (source_image, target_text, attack_keyword)
ATTACK_CASES = [
    ("ocean_scene", "a modern laptop computer on a desk", "laptop computer"),
    ("red_circle", "a golden retriever dog playing", "golden retriever"),
    ("forest_scene", "a pepperoni pizza fresh from oven", "pepperoni pizza"),
]

# Image descriptions for reverse retrieval
IMAGE_DESCRIPTIONS = {
    "red_circle": "a red circle shape on white background",
    "blue_rectangle": "a blue rectangle shape on white background",
    "green_triangle": "a green triangle shape on white background",
    "ocean_scene": "an ocean with water waves and sky",
    "forest_scene": "a green forest with trees",
    "city_scene": "a city skyline at night with buildings",
}


def run_experiment():
    """Run the cross-modal transfer attack experiment."""
    print("=" * 60)
    print("Experiment 6: Cross-Modal Transfer Attack")
    print("=" * 60)

    client = GeminiEmbeddingClient()
    os.makedirs(ASSETS_DIR, exist_ok=True)

    results = []

    for src_key, target_text, attack_keyword in ATTACK_CASES:
        print(f"\n{'=' * 50}")
        print(f"Attack case: {src_key} + '{attack_keyword}'")
        print(f"{'=' * 50}")

        case_result = {
            "source": src_key,
            "target_text": target_text,
            "attack_keyword": attack_keyword,
            "directions": {},
        }

        # Generate clean and attacked images
        clean_bytes = SOURCE_IMAGES[src_key]()
        attacked_bytes = add_typographic_text(
            clean_bytes, attack_keyword,
            font_size=60, position="center", color=(0, 0, 0),
        )

        # Save attacked image
        atk_name = f"transfer_{src_key}_{attack_keyword.replace(' ', '_')}.png"
        with open(os.path.join(ASSETS_DIR, atk_name), "wb") as f:
            f.write(attacked_bytes)

        # Get embeddings
        print("  Embedding clean image...")
        clean_img_emb = client.embed_image(clean_bytes)
        print("  Embedding attacked image...")
        attacked_img_emb = client.embed_image(attacked_bytes)
        print("  Embedding target text...")
        target_text_emb = client.embed_text(target_text)

        # ===== Direction 1: Image → Text (original attack direction) =====
        print("\n  --- Direction 1: Image → Text ---")
        # How similar is the attacked image to the target text?
        sim_clean_target = cosine_similarity(clean_img_emb, target_text_emb)
        sim_attacked_target = cosine_similarity(attacked_img_emb, target_text_emb)
        shift_img2txt = sim_attacked_target - sim_clean_target

        print(f"    Clean→Target text:    {sim_clean_target:.4f}")
        print(f"    Attacked→Target text: {sim_attacked_target:.4f}")
        print(f"    Shift:                {shift_img2txt:+.4f}")

        case_result["directions"]["img2txt"] = {
            "clean_sim": sim_clean_target,
            "attacked_sim": sim_attacked_target,
            "shift": shift_img2txt,
        }

        # ===== Direction 2: Text → Image (reverse direction) =====
        print("\n  --- Direction 2: Text → Image ---")
        # Build image corpus and rank by similarity to target text query
        image_corpus_names = [
            "red_circle", "blue_rectangle", "green_triangle",
            "ocean_scene", "forest_scene", "city_scene",
        ]
        # Build corpus with clean images
        image_corpus_clean = {}
        for img_key in image_corpus_names:
            if img_key == src_key:
                image_corpus_clean[img_key] = clean_img_emb
            else:
                image_corpus_clean[img_key] = client.embed_image(SOURCE_IMAGES[img_key]())

        ranking_clean = rank_documents(target_text_emb, image_corpus_clean)
        clean_rank = next(
            i for i, (did, _) in enumerate(ranking_clean, 1) if did == src_key
        )

        # Replace with attacked image
        image_corpus_attacked = dict(image_corpus_clean)
        image_corpus_attacked[f"{src_key}_attacked"] = attacked_img_emb
        del image_corpus_attacked[src_key]

        ranking_attacked = rank_documents(target_text_emb, image_corpus_attacked)
        attacked_rank = next(
            i for i, (did, _) in enumerate(ranking_attacked, 1)
            if did == f"{src_key}_attacked"
        )

        print(f"    Query: '{target_text[:40]}...'")
        print(f"    Clean {src_key} rank:    #{clean_rank}")
        print(f"    Attacked {src_key} rank: #{attacked_rank}")
        print(f"    Rank improvement:        {clean_rank - attacked_rank}")

        case_result["directions"]["txt2img"] = {
            "clean_rank": clean_rank,
            "attacked_rank": attacked_rank,
            "rank_improvement": clean_rank - attacked_rank,
            "ranking_clean": [(d, s) for d, s in ranking_clean],
            "ranking_attacked": [(d, s) for d, s in ranking_attacked],
        }

        # ===== Direction 3: Reverse — Attacked Image → Text corpus =====
        print("\n  --- Direction 3: Attacked Image → Text corpus ---")
        # Build text corpus
        text_corpus = {}
        for img_key, desc in IMAGE_DESCRIPTIONS.items():
            text_corpus[f"desc:{img_key}"] = client.embed_text(desc)
        text_corpus["target_text"] = target_text_emb

        # Rank text corpus using clean image as query
        ranking_clean_reverse = rank_documents(clean_img_emb, text_corpus)
        # Rank text corpus using attacked image as query
        ranking_attacked_reverse = rank_documents(attacked_img_emb, text_corpus)

        # Find rank of target_text
        clean_target_rank = next(
            i for i, (did, _) in enumerate(ranking_clean_reverse, 1)
            if did == "target_text"
        )
        attacked_target_rank = next(
            i for i, (did, _) in enumerate(ranking_attacked_reverse, 1)
            if did == "target_text"
        )

        print(f"    Target text rank (clean query):    #{clean_target_rank}")
        print(f"    Target text rank (attacked query): #{attacked_target_rank}")
        print(f"    Rank improvement:                  {clean_target_rank - attacked_target_rank}")

        case_result["directions"]["img2txt_reverse"] = {
            "clean_target_rank": clean_target_rank,
            "attacked_target_rank": attacked_target_rank,
            "rank_improvement": clean_target_rank - attacked_target_rank,
        }

        # ===== Direction 4: Multimodal → Image =====
        print("\n  --- Direction 4: Multimodal query → Image ---")
        try:
            # Create multimodal query: target text + a small generic image
            generic_img = SOURCE_IMAGES["blue_rectangle"]()
            multimodal_emb = client.embed_multimodal(attack_keyword, generic_img)

            # Rank image corpus
            ranking_mm_clean = rank_documents(multimodal_emb, image_corpus_clean)
            ranking_mm_attacked = rank_documents(multimodal_emb, image_corpus_attacked)

            mm_clean_rank = next(
                i for i, (did, _) in enumerate(ranking_mm_clean, 1) if did == src_key
            )
            mm_attacked_rank = next(
                i for i, (did, _) in enumerate(ranking_mm_attacked, 1)
                if did == f"{src_key}_attacked"
            )

            print(f"    Multimodal query: '{attack_keyword}' + blue_rectangle")
            print(f"    Clean {src_key} rank:    #{mm_clean_rank}")
            print(f"    Attacked {src_key} rank: #{mm_attacked_rank}")

            case_result["directions"]["multimodal2img"] = {
                "clean_rank": mm_clean_rank,
                "attacked_rank": mm_attacked_rank,
                "rank_improvement": mm_clean_rank - mm_attacked_rank,
            }
        except Exception as e:
            print(f"    Multimodal embedding failed: {e}")
            case_result["directions"]["multimodal2img"] = {"error": str(e)}

        results.append(case_result)

    # ===== Summary =====
    print(f"\n{'=' * 60}")
    print("Transfer Attack Summary")
    print(f"{'=' * 60}")

    direction_names = ["img2txt", "txt2img", "img2txt_reverse", "multimodal2img"]
    direction_labels = [
        "Image→Text\n(original)",
        "Text→Image\n(reverse)",
        "AttackedImg→Text\n(reverse query)",
        "Multimodal→Image",
    ]

    # Build transfer matrix: rows=attack cases, cols=directions
    transfer_matrix = []
    row_labels = []
    for case in results:
        row = []
        row_labels.append(f"{case['source']}+{case['attack_keyword']}")
        for d in direction_names:
            d_result = case["directions"].get(d, {})
            if "error" in d_result:
                row.append(0.0)
            elif "shift" in d_result:
                row.append(d_result["shift"])
            elif "rank_improvement" in d_result:
                # Normalize rank improvement to [0, 1] scale
                ri = d_result["rank_improvement"]
                row.append(max(0, ri) / len(IMAGE_DESCRIPTIONS))
            else:
                row.append(0.0)
        transfer_matrix.append(row)
        print(f"  {row_labels[-1]}:")
        for i, d in enumerate(direction_names):
            print(f"    {d}: {row[i]:.4f}")

    # Save results
    os.makedirs(RESULTS_DATA_DIR, exist_ok=True)
    output = {
        "experiment": "exp6_cross_modal_transfer",
        "results": results,
        "transfer_matrix": transfer_matrix,
        "row_labels": row_labels,
        "direction_names": direction_names,
    }
    output_path = os.path.join(RESULTS_DATA_DIR, "exp6_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Plot transfer matrix
    os.makedirs(RESULTS_FIGURES_DIR, exist_ok=True)
    plot_transfer_matrix(
        row_labels, direction_labels, transfer_matrix,
        title="Exp6: Cross-Modal Transfer Attack Effectiveness",
        save_path=os.path.join(RESULTS_FIGURES_DIR, "exp6_transfer_matrix.png"),
    )

    return output


if __name__ == "__main__":
    run_experiment()
