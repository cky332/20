"""Visualization utilities for experiment results."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import sys
sys.path.insert(0, ".")
from config import RESULTS_FIGURES_DIR


def _ensure_dir(path):
    os.makedirs(os.path.dirname(path) if not os.path.isdir(path) else path, exist_ok=True)


def plot_similarity_heatmap(sim_matrix, labels, title="Cosine Similarity Matrix",
                            save_path=None):
    """Plot a cosine similarity heatmap.

    Args:
        sim_matrix: 2D list or numpy array of similarities.
        labels: List of labels for rows/columns.
        title: Plot title.
        save_path: Path to save figure (optional).
    """
    matrix = np.array(sim_matrix)
    fig, ax = plt.subplots(figsize=(max(8, len(labels)), max(6, len(labels) * 0.8)))

    cmap = LinearSegmentedColormap.from_list("sim", ["#2166AC", "#F7F7F7", "#B2182B"])
    im = ax.imshow(matrix, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                    fontsize=7, color="black" if abs(matrix[i, j]) < 0.5 else "white")

    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def plot_attack_comparison(results, title="Typographic Attack: Similarity Shift",
                           save_path=None):
    """Plot clean vs attacked similarity comparison.

    Args:
        results: List of dicts with keys 'label', 'clean_similarity', 'attacked_similarity'.
        title: Plot title.
        save_path: Path to save figure.
    """
    labels = [r["label"] for r in results]
    clean = [r["clean_similarity"] for r in results]
    attacked = [r["attacked_similarity"] for r in results]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.8), 6))
    bars1 = ax.bar(x - width / 2, clean, width, label="Clean Image → Target Text",
                   color="#4393C3", alpha=0.8)
    bars2 = ax.bar(x + width / 2, attacked, width, label="Attacked Image → Target Text",
                   color="#D6604D", alpha=0.8)

    ax.set_xlabel("Image-Target Pair")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def plot_parameter_sensitivity(param_name, param_values, shifts,
                                title=None, save_path=None):
    """Plot parameter sensitivity analysis.

    Args:
        param_name: Name of the parameter being varied.
        param_values: List of parameter values (x-axis).
        shifts: List of similarity shifts (y-axis).
        title: Plot title.
        save_path: Path to save figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(param_values)), shifts, "o-", color="#D6604D",
            markersize=8, linewidth=2)
    ax.set_xticks(range(len(param_values)))
    ax.set_xticklabels(param_values, rotation=30, ha="right")
    ax.set_xlabel(param_name)
    ax.set_ylabel("Similarity Shift (attacked - clean)")
    ax.set_title(title or f"Attack Sensitivity to {param_name}",
                 fontsize=12, fontweight="bold")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def plot_embedding_space_2d(embeddings_dict, labels_groups=None,
                             title="Embedding Space (PCA 2D)",
                             save_path=None):
    """Plot 2D PCA projection of embeddings.

    Args:
        embeddings_dict: Dict mapping label -> numpy array.
        labels_groups: Optional dict mapping label -> group name for coloring.
        title: Plot title.
        save_path: Path to save figure.
    """
    from sklearn.decomposition import PCA

    labels = list(embeddings_dict.keys())
    vectors = np.array([embeddings_dict[l] for l in labels])

    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors)

    fig, ax = plt.subplots(figsize=(10, 8))

    if labels_groups:
        groups = set(labels_groups.values())
        colors = plt.cm.Set2(np.linspace(0, 1, len(groups)))
        group_color = {g: colors[i] for i, g in enumerate(groups)}

        for i, label in enumerate(labels):
            group = labels_groups.get(label, "other")
            ax.scatter(coords[i, 0], coords[i, 1],
                       c=[group_color[group]], s=100, zorder=5)
            ax.annotate(label, (coords[i, 0], coords[i, 1]),
                        fontsize=7, ha="left", va="bottom")

        for g in groups:
            ax.scatter([], [], c=[group_color[g]], label=g, s=60)
        ax.legend(fontsize=9)
    else:
        ax.scatter(coords[:, 0], coords[:, 1], c="#4393C3", s=100, zorder=5)
        for i, label in enumerate(labels):
            ax.annotate(label, (coords[i, 0], coords[i, 1]),
                        fontsize=7, ha="left", va="bottom")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def plot_hubness_distribution(hubness_counts, adversarial_ids=None,
                               title="Hubness Distribution (N_k)",
                               save_path=None):
    """Plot hubness distribution as a bar chart.

    Args:
        hubness_counts: Dict mapping item_id -> N_k count.
        adversarial_ids: Set of item IDs to highlight in red.
        title: Plot title.
        save_path: Path to save figure.
    """
    adversarial_ids = adversarial_ids or set()
    items = sorted(hubness_counts.items(), key=lambda x: x[1], reverse=True)
    labels = [item[0] for item in items]
    counts = [item[1] for item in items]
    colors = ["#D6604D" if l in adversarial_ids else "#4393C3" for l in labels]

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.8), 6))
    ax.bar(range(len(labels)), counts, color=colors, alpha=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("N_k (Hubness Score)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#D6604D", label="Adversarial"),
        Patch(facecolor="#4393C3", label="Clean"),
    ]
    ax.legend(handles=legend_elements)

    plt.tight_layout()
    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def plot_collision_progression(pairs_data, title="Semantic Collision Progression",
                                save_path=None):
    """Plot similarity progression as attack intensity increases.

    Args:
        pairs_data: List of dicts with 'pair_label' and 'similarities'
            (list of (intensity_label, similarity) tuples).
        title: Plot title.
        save_path: Path to save figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(pairs_data)))

    for i, pair in enumerate(pairs_data):
        intensities = [s[0] for s in pair["similarities"]]
        sims = [s[1] for s in pair["similarities"]]
        ax.plot(range(len(intensities)), sims, "o-", color=colors[i],
                label=pair["pair_label"], linewidth=2, markersize=8)

    ax.set_xticks(range(len(pairs_data[0]["similarities"])))
    ax.set_xticklabels(
        [s[0] for s in pairs_data[0]["similarities"]],
        rotation=30, ha="right", fontsize=9,
    )
    ax.set_xlabel("Attack Intensity")
    ax.set_ylabel("Cosine Similarity to Target Text")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def plot_transfer_matrix(row_labels, col_labels, values,
                          title="Cross-Modal Transfer Matrix",
                          save_path=None):
    """Plot a heatmap of transfer effectiveness across modality directions.

    Args:
        row_labels: List of row labels (source directions).
        col_labels: List of column labels (target directions).
        values: 2D list of transfer scores.
        title: Plot title.
        save_path: Path to save figure.
    """
    matrix = np.array(values)
    fig, ax = plt.subplots(figsize=(max(8, len(col_labels)), max(6, len(row_labels) * 0.8)))

    cmap = LinearSegmentedColormap.from_list("transfer", ["#F7F7F7", "#D6604D"])
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0)

    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(row_labels, fontsize=9)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                    fontsize=9, color="black" if matrix[i, j] < 0.5 else "white")

    plt.colorbar(im, ax=ax, shrink=0.8, label="Similarity Shift")
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def plot_target_vs_random_shift(results, title="Target vs Random Text Shift",
                                save_path=None):
    """Plot target shift vs random shift side-by-side for each pair.

    Args:
        results: List of dicts with 'label', 'target_shift', 'random_shift'.
        title: Plot title.
        save_path: Path to save figure.
    """
    labels = [r["label"] for r in results]
    target_shifts = [r["target_shift"] for r in results]
    random_shifts = [r["random_shift"] for r in results]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.5), 6))
    ax.bar(x - width / 2, target_shifts, width, label="Target Text Shift",
           color="#D6604D", alpha=0.8)
    ax.bar(x + width / 2, random_shifts, width, label="Random Text Shift",
           color="#4393C3", alpha=0.8)

    ax.set_xlabel("Image-Target Pair")
    ax.set_ylabel("Similarity Shift")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, ha="center", fontsize=6)
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def plot_mismatch_heatmap(matrix, source_labels, target_labels,
                           title="Mismatch Analysis: Shift per Target",
                           save_path=None):
    """Plot heatmap showing shift of each attacked image toward all targets.

    Args:
        matrix: 2D numpy array of shape (n_attacked, n_targets) with shift values.
        source_labels: Row labels (attacked image descriptions).
        target_labels: Column labels (target texts).
        title: Plot title.
        save_path: Path to save figure.
    """
    matrix = np.array(matrix)
    fig, ax = plt.subplots(figsize=(max(8, len(target_labels) * 1.2),
                                    max(6, len(source_labels) * 0.4)))

    cmap = LinearSegmentedColormap.from_list("shift", ["#2166AC", "#F7F7F7", "#B2182B"])
    vmax = max(abs(matrix.min()), abs(matrix.max()))
    im = ax.imshow(matrix, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(target_labels)))
    ax.set_yticks(range(len(source_labels)))
    ax.set_xticklabels(target_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(source_labels, fontsize=7)

    for i in range(len(source_labels)):
        for j in range(len(target_labels)):
            ax.text(j, i, f"{matrix[i, j]:+.3f}", ha="center", va="center",
                    fontsize=6, color="black" if abs(matrix[i, j]) < vmax * 0.6 else "white")

    plt.colorbar(im, ax=ax, shrink=0.8, label="Similarity Shift")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Measured Against Target Text")
    ax.set_ylabel("Attacked Image (overlaid text)")
    plt.tight_layout()

    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def plot_shift_by_target_boxplot(results_by_target, title="Shift Distribution by Target",
                                  save_path=None):
    """Plot boxplot of shifts grouped by target text.

    Args:
        results_by_target: Dict mapping target_text -> list of shifts.
        title: Plot title.
        save_path: Path to save figure.
    """
    labels = list(results_by_target.keys())
    data = [results_by_target[l] for l in labels]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 6))
    bp = ax.boxplot(data, labels=labels, patch_artist=True)

    for patch in bp["boxes"]:
        patch.set_facecolor("#D6604D")
        patch.set_alpha(0.6)

    ax.set_xlabel("Target Text")
    ax.set_ylabel("Similarity Shift")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def plot_retrieval_ranking(rankings_before, rankings_after,
                            poisoned_doc_id, title="Retrieval Ranking Change",
                            save_path=None):
    """Plot retrieval ranking changes before/after poisoning.

    Args:
        rankings_before: List of (doc_id, score) tuples before poisoning.
        rankings_after: List of (doc_id, score) tuples after poisoning.
        poisoned_doc_id: ID of the poisoned document.
        title: Plot title.
        save_path: Path to save figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Before
    doc_ids = [r[0] for r in rankings_before]
    scores = [r[1] for r in rankings_before]
    colors = ["#D6604D" if d == poisoned_doc_id else "#4393C3" for d in doc_ids]
    ax1.barh(range(len(doc_ids)), scores, color=colors)
    ax1.set_yticks(range(len(doc_ids)))
    ax1.set_yticklabels(doc_ids, fontsize=8)
    ax1.set_xlabel("Cosine Similarity")
    ax1.set_title("Before Poisoning")
    ax1.invert_yaxis()

    # After
    doc_ids = [r[0] for r in rankings_after]
    scores = [r[1] for r in rankings_after]
    colors = ["#D6604D" if d == poisoned_doc_id else "#4393C3" for d in doc_ids]
    ax2.barh(range(len(doc_ids)), scores, color=colors)
    ax2.set_yticks(range(len(doc_ids)))
    ax2.set_yticklabels(doc_ids, fontsize=8)
    ax2.set_xlabel("Cosine Similarity")
    ax2.set_title("After Poisoning")
    ax2.invert_yaxis()

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()
