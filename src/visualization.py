"""Visualization utilities for experiment results."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

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


def plot_ablation_comparison(variant_data, title="FigStep Ablation: Mean Similarity to Target",
                            save_path=None):
    """Plot mean similarity across ablation variants as a bar chart.

    Args:
        variant_data: List of dicts with 'variant_name' and 'mean_similarity'.
        title: Plot title.
        save_path: Path to save figure.
    """
    names = [d["variant_name"] for d in variant_data]
    sims = [d["mean_similarity"] for d in variant_data]

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.2), 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    bars = ax.bar(range(len(names)), sims, color=colors, alpha=0.85, edgecolor="gray")

    for bar, val in zip(bars, sims):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Mean Cosine Similarity to Target Text")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def plot_figstep_vs_simple_scatter(pairs, title="FigStep vs Simple Overlay Shift",
                                   save_path=None):
    """Scatter plot comparing FigStep shift vs simple overlay shift.

    Points above the y=x line indicate FigStep is more effective.

    Args:
        pairs: List of dicts with 'label', 'simple_shift', 'figstep_shift'.
        title: Plot title.
        save_path: Path to save figure.
    """
    simple = [p["simple_shift"] for p in pairs]
    figstep = [p["figstep_shift"] for p in pairs]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(simple, figstep, c="#D6604D", s=60, alpha=0.7, edgecolors="gray", zorder=5)

    # y=x reference line
    lo = min(min(simple), min(figstep)) - 0.02
    hi = max(max(simple), max(figstep)) + 0.02
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="y = x")

    ax.set_xlabel("Simple Overlay Shift")
    ax.set_ylabel("FigStep Shift")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
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


# ---------------------------------------------------------------------------
# New visualization functions for exp1 rewrite (Phase 8)
# ---------------------------------------------------------------------------

def plot_grouped_bar_asr(data, title="ASR by Attack Strategy and Prompt Template",
                         save_path=None):
    """Plot grouped bar chart of ASR by attack strategy and prompt template.

    Args:
        data: Dict mapping attack_strategy -> {prompt_template: asr_value}.
        title: Plot title.
        save_path: Path to save figure.
    """
    strategies = list(data.keys())
    templates = list(data[strategies[0]].keys())
    n_groups = len(strategies)
    n_bars = len(templates)

    x = np.arange(n_groups)
    width = 0.8 / n_bars
    colors = ["#4393C3", "#D6604D", "#5AAE61"]

    fig, ax = plt.subplots(figsize=(max(8, n_groups * 2), 6))

    for i, tmpl in enumerate(templates):
        values = [data[s][tmpl] for s in strategies]
        bars = ax.bar(x + i * width - (n_bars - 1) * width / 2, values, width,
                      label=tmpl, color=colors[i % len(colors)], alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.1%}", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Attack Strategy")
    ax.set_ylabel("Attack Success Rate (ASR)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=10)
    ax.legend(title="Prompt Template")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def plot_typography_factor_heatmap(data, row_param, col_param,
                                    title="Typography Factor Impact on ASR",
                                    save_path=None):
    """Plot heatmap showing cross-impact of two typography factors on ASR.

    Args:
        data: 2D numpy array of ASR values.
        row_param: (name, labels) tuple for rows.
        col_param: (name, labels) tuple for columns.
        title: Plot title.
        save_path: Path to save figure.
    """
    matrix = np.array(data)
    row_name, row_labels = row_param
    col_name, col_labels = col_param

    fig, ax = plt.subplots(figsize=(max(8, len(col_labels) * 1.5),
                                    max(5, len(row_labels) * 0.8)))
    cmap = LinearSegmentedColormap.from_list("asr", ["#F7F7F7", "#D6604D"])
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel(col_name)
    ax.set_ylabel(row_name)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            ax.text(j, i, f"{matrix[i, j]:.1%}", ha="center", va="center",
                    fontsize=9, color="black" if matrix[i, j] < 0.5 else "white")

    plt.colorbar(im, ax=ax, shrink=0.8, label="ASR")
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def plot_dimension_truncation(dims, metrics_dict,
                               title="Dimension Truncation: Accuracy & ASR",
                               save_path=None):
    """Plot line chart showing how accuracy and ASR change with dimension truncation.

    Args:
        dims: List of dimension values (e.g., [768, 1536, 3072]).
        metrics_dict: Dict mapping metric_name -> list of values (same order as dims).
        title: Plot title.
        save_path: Path to save figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4393C3", "#D6604D", "#5AAE61", "#FDB863"]
    markers = ["o", "s", "^", "D"]

    for i, (name, values) in enumerate(metrics_dict.items()):
        ax.plot(range(len(dims)), values, f"{markers[i % len(markers)]}-",
                color=colors[i % len(colors)], label=name, linewidth=2, markersize=8)

    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels([str(d) for d in dims])
    ax.set_xlabel("Embedding Dimension")
    ax.set_ylabel("Rate")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()

    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def plot_gemini_vs_clip(metrics, title="Gemini Embedding 2 vs CLIP Comparison",
                         save_path=None):
    """Plot side-by-side comparison of Gemini and CLIP metrics.

    Args:
        metrics: List of dicts with 'metric_name', 'gemini_value', 'clip_value'.
        title: Plot title.
        save_path: Path to save figure.
    """
    names = [m["metric_name"] for m in metrics]
    gemini = [m["gemini_value"] for m in metrics]
    clip_vals = [m["clip_value"] for m in metrics]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.5), 6))
    bars1 = ax.bar(x - width / 2, gemini, width, label="Gemini Embedding 2",
                   color="#4393C3", alpha=0.85)
    bars2 = ax.bar(x + width / 2, clip_vals, width, label="CLIP ViT-B/32",
                   color="#D6604D", alpha=0.85)

    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def plot_tsne_umap(embeddings, labels, groups, method="tsne",
                    title=None, save_path=None):
    """Plot t-SNE or UMAP 2D projection of embeddings.

    Args:
        embeddings: 2D numpy array (n_samples, n_dims).
        labels: List of text labels for each point.
        groups: List of group names for each point (used for coloring).
        method: 'tsne' or 'umap'.
        title: Plot title.
        save_path: Path to save figure.
    """
    if method == "umap":
        try:
            from umap import UMAP
            reducer = UMAP(n_components=2, random_state=42)
            coords = reducer.fit_transform(embeddings)
        except ImportError:
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42,
                           perplexity=min(30, len(embeddings) - 1))
            coords = reducer.fit_transform(embeddings)
            method = "tsne (umap unavailable)"
    else:
        from sklearn.manifold import TSNE
        perp = min(30, max(5, len(embeddings) - 1))
        reducer = TSNE(n_components=2, random_state=42, perplexity=perp)
        coords = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(12, 10))

    unique_groups = sorted(set(groups))
    group_colors = {
        "clean_image": "#4393C3",
        "attack_random": "#FDB863",
        "attack_neighbor": "#D6604D",
        "attack_cross_domain": "#B2182B",
        "text_label": "#5AAE61",
        "benign_control": "#92C5DE",
        "noise_control": "#999999",
    }
    fallback_colors = plt.cm.Set2(np.linspace(0, 1, len(unique_groups)))

    for i, group in enumerate(unique_groups):
        mask = [j for j, g in enumerate(groups) if g == group]
        color = group_colors.get(group, fallback_colors[i])
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[color],
                   s=80, alpha=0.7, label=group, edgecolors="gray", linewidths=0.5)

    for j, (lbl, grp) in enumerate(zip(labels, groups)):
        if grp == "text_label":
            ax.annotate(lbl, (coords[j, 0], coords[j, 1]),
                        fontsize=8, fontweight="bold", ha="left", va="bottom")

    ax.set_title(title or f"Embedding Space ({method.upper()})",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def plot_case_studies(cases, title="Attack Case Studies", save_path=None):
    """Plot grid of attack case studies showing success/failure.

    Args:
        cases: List of dicts with:
            'category', 'attack_type', 'attack_text',
            'predicted', 'true_sim', 'attack_sim', 'success' (bool),
            'image_bytes' (optional).
        title: Plot title.
        save_path: Path to save figure.
    """
    import io as _io

    n = len(cases)
    cols = min(5, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.5))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for idx, case in enumerate(cases):
        r, c = idx // cols, idx % cols
        ax = axes[r, c]

        if "image_bytes" in case and case["image_bytes"]:
            from PIL import Image as PILImage
            img = PILImage.open(_io.BytesIO(case["image_bytes"]))
            ax.imshow(img)
        else:
            ax.set_facecolor("#f0f0f0")

        status = "SUCCESS" if case["success"] else "FAIL"
        color = "#D6604D" if case["success"] else "#4393C3"
        ax.set_title(f"{case['category']} -> \"{case['attack_text']}\"\n"
                     f"Pred: {case['predicted']} [{status}]\n"
                     f"True: {case['true_sim']:.3f} | Atk: {case['attack_sim']:.3f}",
                     fontsize=7, color=color)
        ax.axis("off")

    for idx in range(n, rows * cols):
        r, c = idx // cols, idx % cols
        axes[r, c].axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def plot_control_comparison(data, title="Attack vs Control Groups",
                            save_path=None):
    """Plot comparison between attack, benign control, and noise control.

    Args:
        data: Dict mapping group_name -> {prompt: accuracy}.
        title: Plot title.
        save_path: Path to save figure.
    """
    groups = list(data.keys())
    prompts = list(data[groups[0]].keys())
    n_groups = len(groups)
    n_prompts = len(prompts)

    x = np.arange(n_prompts)
    width = 0.8 / n_groups
    colors = ["#D6604D", "#4393C3", "#999999", "#5AAE61"]

    fig, ax = plt.subplots(figsize=(max(8, n_prompts * 2.5), 6))

    for i, group in enumerate(groups):
        values = [data[group][p] for p in prompts]
        bars = ax.bar(x + i * width - (n_groups - 1) * width / 2, values, width,
                      label=group, color=colors[i % len(colors)], alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.1%}", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Prompt Template")
    ax.set_ylabel("Top-1 Accuracy")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(prompts, fontsize=10)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def plot_ablation_factor_bars(factor_name, level_names, asr_values,
                               title=None, save_path=None):
    """Plot bar chart for a single ablation factor.

    Args:
        factor_name: Name of the factor (e.g., "Font Size").
        level_names: List of level labels.
        asr_values: List of ASR values per level.
        title: Plot title.
        save_path: Path to save figure.
    """
    fig, ax = plt.subplots(figsize=(max(8, len(level_names) * 1.2), 5))
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(level_names)))
    bars = ax.bar(range(len(level_names)), asr_values, color=colors, alpha=0.85,
                  edgecolor="gray")

    for bar, val in zip(bars, asr_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.1%}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(level_names)))
    ax.set_xticklabels(level_names, rotation=30, ha="right", fontsize=9)
    ax.set_xlabel(factor_name)
    ax.set_ylabel("ASR")
    ax.set_title(title or f"Attack Success Rate by {factor_name}",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()
