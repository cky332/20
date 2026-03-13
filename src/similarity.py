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


def compute_attack_effectiveness(clean_sim, attacked_sim, threshold=0.0):
    """Compute attack effectiveness metrics.

    Args:
        clean_sim: Cosine similarity between clean image and target text.
        attacked_sim: Cosine similarity between attacked image and target text.
        threshold: Minimum improvement to consider attack successful.

    Returns:
        Dict with metrics.
    """
    shift = attacked_sim - clean_sim
    return {
        "clean_similarity": clean_sim,
        "attacked_similarity": attacked_sim,
        "similarity_shift": shift,
        "relative_shift_pct": (shift / max(abs(clean_sim), 1e-10)) * 100,
        "attack_successful": shift > threshold,
    }


def compute_attack_success_rate(results_list, threshold=0.0):
    """Compute overall attack success rate.

    Args:
        results_list: List of dicts from compute_attack_effectiveness.
        threshold: Minimum shift to count as successful.

    Returns:
        Dict with aggregate metrics.
    """
    if not results_list:
        return {"success_rate": 0.0, "total": 0}

    successful = sum(1 for r in results_list if r["similarity_shift"] > threshold)
    shifts = [r["similarity_shift"] for r in results_list]

    return {
        "success_rate": successful / len(results_list),
        "total": len(results_list),
        "successful": successful,
        "mean_shift": float(np.mean(shifts)),
        "std_shift": float(np.std(shifts)),
        "max_shift": float(np.max(shifts)),
        "min_shift": float(np.min(shifts)),
    }


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
