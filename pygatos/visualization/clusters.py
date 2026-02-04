"""Cluster and embedding visualization using UMAP."""

import logging
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

logger = logging.getLogger(__name__)


def plot_clusters_2d(
    embeddings: np.ndarray,
    labels: np.ndarray,
    title: str = "Cluster Visualization",
    figsize: tuple[int, int] = (12, 10),
    point_size: int = 30,
    alpha: float = 0.6,
    show_legend: bool = True,
    max_legend_items: int = 20,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Create a 2D scatter plot of clustered embeddings.

    Expects 2D embeddings (e.g., from UMAP with n_components=2).

    Args:
        embeddings: 2D array of shape (n_samples, 2).
        labels: Cluster labels for each point.
        title: Chart title.
        figsize: Figure size.
        point_size: Size of scatter points.
        alpha: Transparency of points.
        show_legend: If True, show cluster legend.
        max_legend_items: Maximum legend items to show.
        save_path: If set, save figure to this path.
        dpi: Resolution for saved figure.

    Returns:
        matplotlib Figure object.
    """
    if embeddings.shape[1] != 2:
        raise ValueError(
            f"Expected 2D embeddings, got shape {embeddings.shape}. "
            "Use UMAP with n_components=2 first."
        )

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Get colors
    if n_clusters <= 10:
        colors = list(mcolors.TABLEAU_COLORS.values())
    elif n_clusters <= 20:
        colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())[:10]
    else:
        cmap = plt.cm.get_cmap("tab20", n_clusters)
        colors = [cmap(i) for i in range(n_clusters)]

    fig, ax = plt.subplots(figsize=figsize)

    for i, label in enumerate(unique_labels):
        mask = labels == label
        color = colors[i % len(colors)]

        if label == -1:  # Noise cluster (common in HDBSCAN)
            ax.scatter(
                embeddings[mask, 0],
                embeddings[mask, 1],
                c="gray",
                s=point_size // 2,
                alpha=alpha / 2,
                label="Noise" if show_legend else None,
            )
        else:
            ax.scatter(
                embeddings[mask, 0],
                embeddings[mask, 1],
                c=[color],
                s=point_size,
                alpha=alpha,
                label=f"Cluster {label}" if show_legend and i < max_legend_items else None,
            )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)

    if show_legend and n_clusters <= max_legend_items:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    elif show_legend:
        ax.text(
            1.02,
            0.5,
            f"{n_clusters} clusters\n(legend omitted)",
            transform=ax.transAxes,
            fontsize=10,
            va="center",
        )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved cluster plot to {save_path}")

    return fig


def plot_codes_2d(
    embeddings: np.ndarray,
    code_names: list[str],
    accepted_mask: Optional[np.ndarray] = None,
    theme_labels: Optional[list[str]] = None,
    title: str = "Code Embeddings",
    figsize: tuple[int, int] = (14, 10),
    point_size: int = 100,
    annotate: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Visualize code embeddings in 2D space.

    Args:
        embeddings: 2D array of code embeddings.
        code_names: Names of codes.
        accepted_mask: Boolean mask for accepted codes (optional).
        theme_labels: Theme label for each code (optional).
        title: Chart title.
        figsize: Figure size.
        point_size: Size of scatter points.
        annotate: If True, add code name labels.
        save_path: If set, save figure to this path.
        dpi: Resolution for saved figure.

    Returns:
        matplotlib Figure object.
    """
    if embeddings.shape[1] != 2:
        # Apply UMAP if needed
        try:
            import umap

            reducer = umap.UMAP(n_components=2, random_state=42)
            embeddings = reducer.fit_transform(embeddings)
            logger.info("Applied UMAP to reduce to 2D for visualization")
        except ImportError:
            raise ValueError(
                "Embeddings are not 2D and UMAP is not installed. "
                "Install with: pip install umap-learn"
            )

    fig, ax = plt.subplots(figsize=figsize)

    if theme_labels is not None:
        # Color by theme
        unique_themes = list(set(theme_labels))
        colors = list(mcolors.TABLEAU_COLORS.values())

        for i, theme in enumerate(unique_themes):
            mask = np.array([t == theme for t in theme_labels])
            color = colors[i % len(colors)]

            ax.scatter(
                embeddings[mask, 0],
                embeddings[mask, 1],
                c=[color],
                s=point_size,
                alpha=0.7,
                label=theme,
            )

        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    elif accepted_mask is not None:
        # Color by accepted/rejected
        ax.scatter(
            embeddings[accepted_mask, 0],
            embeddings[accepted_mask, 1],
            c="green",
            s=point_size,
            alpha=0.7,
            label="Accepted",
        )
        ax.scatter(
            embeddings[~accepted_mask, 0],
            embeddings[~accepted_mask, 1],
            c="red",
            s=point_size // 2,
            alpha=0.4,
            label="Rejected",
        )
        ax.legend()

    else:
        ax.scatter(
            embeddings[:, 0],
            embeddings[:, 1],
            c="steelblue",
            s=point_size,
            alpha=0.7,
        )

    if annotate:
        for i, name in enumerate(code_names):
            ax.annotate(
                name,
                (embeddings[i, 0], embeddings[i, 1]),
                fontsize=8,
                alpha=0.8,
                ha="center",
                va="bottom",
            )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved code visualization to {save_path}")

    return fig


def create_umap_embeddings(
    embeddings: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
) -> np.ndarray:
    """
    Create UMAP embeddings for visualization.

    Args:
        embeddings: High-dimensional embeddings.
        n_components: Output dimensions (2 for plotting).
        n_neighbors: UMAP n_neighbors parameter.
        min_dist: UMAP min_dist parameter.
        metric: Distance metric.
        random_state: Random seed.

    Returns:
        Reduced embeddings.
    """
    try:
        import umap

        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        )

        return reducer.fit_transform(embeddings)

    except ImportError:
        logger.error("UMAP not installed. Install with: pip install umap-learn")
        raise


def plot_cluster_sizes(
    labels: np.ndarray,
    title: str = "Cluster Size Distribution",
    figsize: tuple[int, int] = (12, 6),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Create a bar chart of cluster sizes.

    Args:
        labels: Cluster labels.
        title: Chart title.
        figsize: Figure size.
        save_path: If set, save figure to this path.
        dpi: Resolution for saved figure.

    Returns:
        matplotlib Figure object.
    """
    unique, counts = np.unique(labels, return_counts=True)

    # Sort by size
    sorted_idx = np.argsort(-counts)
    unique = unique[sorted_idx]
    counts = counts[sorted_idx]

    fig, ax = plt.subplots(figsize=figsize)

    colors = ["gray" if label == -1 else "steelblue" for label in unique]

    ax.bar(range(len(unique)), counts, color=colors)

    ax.set_xticks(range(len(unique)))
    ax.set_xticklabels(
        [f"Noise" if l == -1 else f"C{l}" for l in unique],
        rotation=45,
        ha="right",
        fontsize=8,
    )

    ax.set_xlabel("Cluster")
    ax.set_ylabel("Size")
    ax.set_title(title)

    # Add stats
    non_noise = counts[unique != -1]
    if len(non_noise) > 0:
        ax.text(
            0.98,
            0.98,
            f"Mean: {np.mean(non_noise):.1f}\nMedian: {np.median(non_noise):.0f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved cluster size chart to {save_path}")

    return fig
