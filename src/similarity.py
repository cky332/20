"""Cosine similarity and attack metrics computation."""

import numpy as np


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors.

    Args:
        a, b: 1D numpy arrays.

    Returns:
        Float in [-1, 1].
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_distance(a, b):
    """Compute cosine distance (1 - cosine_similarity)."""
    return 1.0 - cosine_similarity(a, b)


def compute_similarity_matrix(embeddings_dict):
    """Compute pairwise cosine similarity matrix.

    Args:
        embeddings_dict: Dict mapping label -> numpy array.

    Returns:
        Dict with 'labels' (list) and 'matrix' (2D list).
    """
    labels = list(embeddings_dict.keys())
    n = len(labels)
    matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            matrix[i][j] = cosine_similarity(
                embeddings_dict[labels[i]],
                embeddings_dict[labels[j]]
            )

    return {"labels": labels, "matrix": matrix}


def compute_attack_effectiveness(clean_sim, attacked_sim, threshold=0.0,
                                  random_sim=None):
    """Compute attack effectiveness metrics.

    Args:
        clean_sim: Cosine similarity between clean image and target text.
        attacked_sim: Cosine similarity between attacked image and target text.
        threshold: Minimum improvement to consider attack successful.
        random_sim: Optional cosine similarity between random-text attacked
            image and target text, used to compute targeted advantage.

    Returns:
        Dict with metrics.
    """
    shift = attacked_sim - clean_sim
    result = {
        "clean_similarity": clean_sim,
        "attacked_similarity": attacked_sim,
        "similarity_shift": shift,
        "relative_shift_pct": (shift / max(abs(clean_sim), 1e-10)) * 100,
        "attack_successful": shift > threshold,
    }
    if random_sim is not None:
        random_shift = random_sim - clean_sim
        result["random_similarity"] = random_sim
        result["random_shift"] = random_shift
        result["targeted_advantage"] = shift - random_shift
    return result


def compute_attack_success_rate(results_list, threshold=0.0,
                                 use_random_baseline=False):
    """Compute overall attack success rate.

    Args:
        results_list: List of dicts from compute_attack_effectiveness.
        threshold: Minimum shift to count as successful. Ignored when
            use_random_baseline is True.
        use_random_baseline: If True and results contain random_shift data,
            define success as target_shift > mean(random_shift) + 2*std(random_shift).

    Returns:
        Dict with aggregate metrics.
    """
    if not results_list:
        return {"success_rate": 0.0, "total": 0}

    shifts = [r["similarity_shift"] for r in results_list]

    # Determine dynamic threshold from random baseline if requested
    effective_threshold = threshold
    if use_random_baseline:
        random_shifts = [r["random_shift"] for r in results_list
                         if "random_shift" in r]
        if random_shifts:
            effective_threshold = float(
                np.mean(random_shifts) + 2 * np.std(random_shifts)
            )

    successful = sum(1 for r in results_list
                     if r["similarity_shift"] > effective_threshold)

    result = {
        "success_rate": successful / len(results_list),
        "total": len(results_list),
        "successful": successful,
        "mean_shift": float(np.mean(shifts)),
        "std_shift": float(np.std(shifts)),
        "max_shift": float(np.max(shifts)),
        "min_shift": float(np.min(shifts)),
        "threshold_used": effective_threshold,
    }

    # Add random baseline stats if available
    random_shifts = [r.get("random_shift") for r in results_list
                     if r.get("random_shift") is not None]
    if random_shifts:
        result["mean_random_shift"] = float(np.mean(random_shifts))
        result["std_random_shift"] = float(np.std(random_shifts))
        advantages = [r["targeted_advantage"] for r in results_list
                      if "targeted_advantage" in r]
        if advantages:
            result["mean_targeted_advantage"] = float(np.mean(advantages))

    return result


def rank_documents(query_embedding, doc_embeddings):
    """Rank documents by similarity to a query.

    Args:
        query_embedding: Query vector.
        doc_embeddings: Dict mapping doc_id -> embedding vector.

    Returns:
        List of (doc_id, similarity) tuples, sorted descending.
    """
    scores = []
    for doc_id, emb in doc_embeddings.items():
        sim = cosine_similarity(query_embedding, emb)
        scores.append((doc_id, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores
