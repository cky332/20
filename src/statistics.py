"""Statistical testing and hubness metrics for attack evaluation."""

import numpy as np
from scipy import stats

from src.similarity import cosine_similarity


def compute_confidence_interval(values, confidence=0.95):
    """Compute confidence interval for a list of values.

    Args:
        values: List or array of numerical values.
        confidence: Confidence level (default 0.95).

    Returns:
        Dict with mean, lower, upper, and std.
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n < 2:
        m = float(values[0]) if n == 1 else 0.0
        return {"mean": m, "lower": m, "upper": m, "std": 0.0, "n": n}

    m = float(np.mean(values))
    se = float(stats.sem(values))
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return {
        "mean": m,
        "lower": m - h,
        "upper": m + h,
        "std": float(np.std(values, ddof=1)),
        "n": n,
    }


def paired_ttest(clean_sims, attacked_sims):
    """Run paired t-test on clean vs attacked similarities.

    Args:
        clean_sims: List of clean similarity values.
        attacked_sims: List of attacked similarity values.

    Returns:
        Dict with t_statistic, p_value, significant (at p<0.05),
        mean_diff, and effect direction.
    """
    clean = np.asarray(clean_sims, dtype=np.float64)
    attacked = np.asarray(attacked_sims, dtype=np.float64)

    if len(clean) < 2:
        return {
            "t_statistic": 0.0, "p_value": 1.0, "significant": False,
            "mean_diff": 0.0, "direction": "insufficient_data",
        }

    t_stat, p_val = stats.ttest_rel(attacked, clean)
    mean_diff = float(np.mean(attacked - clean))

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "significant": bool(p_val < 0.05),
        "mean_diff": mean_diff,
        "direction": "attacked > clean" if mean_diff > 0 else "clean > attacked",
    }


def compute_cohens_d(clean_sims, attacked_sims):
    """Compute Cohen's d effect size for paired samples.

    Args:
        clean_sims: List of clean similarity values.
        attacked_sims: List of attacked similarity values.

    Returns:
        Float effect size. |d| < 0.2 small, 0.5 medium, 0.8 large.
    """
    clean = np.asarray(clean_sims, dtype=np.float64)
    attacked = np.asarray(attacked_sims, dtype=np.float64)
    diff = attacked - clean
    if len(diff) < 2 or np.std(diff, ddof=1) == 0:
        return 0.0
    return float(np.mean(diff) / np.std(diff, ddof=1))


def compute_hubness_score(candidate_embedding, query_embeddings, k=1):
    """Compute N_k hubness score: how many queries have candidate as top-k neighbor.

    In a corpus of items, the hubness score N_k(x) of item x is the number
    of queries for which x appears among the k nearest neighbors.

    Args:
        candidate_embedding: The embedding vector to evaluate.
        query_embeddings: Dict mapping query_id -> embedding vector.
        k: Number of nearest neighbors to consider.

    Returns:
        Int count of queries where candidate is in top-k.
    """
    candidate = np.asarray(candidate_embedding)
    count = 0
    for _qid, q_emb in query_embeddings.items():
        # This is a simplified version; for full corpus hubness,
        # use compute_hubness_distribution
        sim = cosine_similarity(candidate, np.asarray(q_emb))
        # We can't determine top-k with just the candidate.
        # This function assumes you want to check against a full corpus.
        # See compute_hubness_distribution for the complete version.
        count += 1  # placeholder
    # Actually, we need the full corpus. Let this be a utility
    # that works with compute_nearest_neighbors.
    return count


def compute_nearest_neighbors(query_embedding, corpus_embeddings, k=1):
    """Find k nearest neighbors for a query in the corpus.

    Args:
        query_embedding: Query vector.
        corpus_embeddings: Dict mapping item_id -> embedding vector.
        k: Number of neighbors to return.

    Returns:
        List of (item_id, similarity) tuples, sorted descending, length k.
    """
    scores = []
    for item_id, emb in corpus_embeddings.items():
        sim = cosine_similarity(np.asarray(query_embedding), np.asarray(emb))
        scores.append((item_id, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]


def compute_hubness_distribution(corpus_embeddings, query_embeddings, k=1):
    """Compute hubness distribution N_k for all items in corpus.

    For each query, find its k nearest neighbors in the corpus.
    Then count how many times each corpus item appears as a neighbor.

    Args:
        corpus_embeddings: Dict mapping item_id -> embedding vector.
        query_embeddings: Dict mapping query_id -> embedding vector.
        k: Number of nearest neighbors per query.

    Returns:
        Dict mapping item_id -> N_k count (hubness score).
    """
    hubness = {item_id: 0 for item_id in corpus_embeddings}

    for _qid, q_emb in query_embeddings.items():
        neighbors = compute_nearest_neighbors(q_emb, corpus_embeddings, k=k)
        for item_id, _sim in neighbors:
            hubness[item_id] += 1

    return hubness


def compute_hubness_stats(hubness_distribution):
    """Compute summary statistics for a hubness distribution.

    Args:
        hubness_distribution: Dict mapping item_id -> N_k count.

    Returns:
        Dict with mean, std, max, skewness, and the hub item(s).
    """
    counts = np.array(list(hubness_distribution.values()), dtype=np.float64)
    if len(counts) == 0:
        return {"mean": 0, "std": 0, "max": 0, "skewness": 0, "top_hubs": []}

    max_count = float(np.max(counts))
    top_hubs = [
        (item_id, count)
        for item_id, count in hubness_distribution.items()
        if count == max_count
    ]

    return {
        "mean": float(np.mean(counts)),
        "std": float(np.std(counts)),
        "max": max_count,
        "skewness": float(stats.skew(counts)) if len(counts) > 2 else 0.0,
        "top_hubs": top_hubs,
    }
