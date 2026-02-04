"""Code frequency visualization."""

import logging
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from pygatos.core.codebook import Codebook

logger = logging.getLogger(__name__)


def plot_code_frequencies(
    frequencies: dict[str, int],
    title: str = "Code Frequencies",
    top_n: Optional[int] = None,
    figsize: tuple[int, int] = (10, 8),
    color: str = "steelblue",
    horizontal: bool = True,
    show_values: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Create a bar chart of code frequencies.

    Args:
        frequencies: Dict mapping code name to count.
        title: Chart title.
        top_n: If set, only show top N codes.
        figsize: Figure size (width, height).
        color: Bar color.
        horizontal: If True, use horizontal bars.
        show_values: If True, show count values on bars.
        save_path: If set, save figure to this path.
        dpi: Resolution for saved figure.

    Returns:
        matplotlib Figure object.
    """
    # Sort by frequency
    sorted_items = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)

    if top_n:
        sorted_items = sorted_items[:top_n]
        title = f"{title} (Top {top_n})"

    names = [item[0] for item in sorted_items]
    counts = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=figsize)

    if horizontal:
        # Reverse for horizontal (top item at top)
        names = names[::-1]
        counts = counts[::-1]

        bars = ax.barh(names, counts, color=color)

        if show_values:
            for bar, count in zip(bars, counts):
                ax.text(
                    bar.get_width() + 0.5,
                    bar.get_y() + bar.get_height() / 2,
                    str(count),
                    va="center",
                    fontsize=9,
                )

        ax.set_xlabel("Frequency")
        ax.set_ylabel("Code")
    else:
        bars = ax.bar(names, counts, color=color)

        if show_values:
            for bar, count in zip(bars, counts):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    str(count),
                    ha="center",
                    fontsize=9,
                )

        ax.set_ylabel("Frequency")
        ax.set_xlabel("Code")
        plt.xticks(rotation=45, ha="right")

    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved frequency chart to {save_path}")

    return fig


def plot_code_distribution(
    frequencies: dict[str, int],
    title: str = "Code Distribution",
    figsize: tuple[int, int] = (10, 6),
    bins: int = 20,
    color: str = "steelblue",
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Create a histogram showing the distribution of code frequencies.

    This helps understand the balance of code usage.

    Args:
        frequencies: Dict mapping code name to count.
        title: Chart title.
        figsize: Figure size.
        bins: Number of histogram bins.
        color: Bar color.
        save_path: If set, save figure to this path.
        dpi: Resolution for saved figure.

    Returns:
        matplotlib Figure object.
    """
    counts = list(frequencies.values())

    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(counts, bins=bins, color=color, edgecolor="white", alpha=0.8)

    ax.set_xlabel("Frequency (times code applied)")
    ax.set_ylabel("Number of Codes")
    ax.set_title(title)

    # Add summary stats
    mean_freq = np.mean(counts)
    median_freq = np.median(counts)

    ax.axvline(mean_freq, color="red", linestyle="--", label=f"Mean: {mean_freq:.1f}")
    ax.axvline(median_freq, color="green", linestyle="--", label=f"Median: {median_freq:.1f}")
    ax.legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved distribution chart to {save_path}")

    return fig


def plot_theme_frequencies(
    codebook: Codebook,
    frequencies: dict[str, int],
    title: str = "Frequencies by Theme",
    figsize: tuple[int, int] = (12, 8),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Create a grouped bar chart showing code frequencies organized by theme.

    Args:
        codebook: Codebook with themes.
        frequencies: Dict mapping code name to count.
        title: Chart title.
        figsize: Figure size.
        save_path: If set, save figure to this path.
        dpi: Resolution for saved figure.

    Returns:
        matplotlib Figure object.
    """
    if not codebook.themes:
        logger.warning("No themes in codebook, falling back to regular frequency plot")
        return plot_code_frequencies(frequencies, title=title, figsize=figsize, save_path=save_path)

    # Get colors for themes
    colors = list(mcolors.TABLEAU_COLORS.values())

    fig, ax = plt.subplots(figsize=figsize)

    current_x = 0
    tick_positions = []
    tick_labels = []
    theme_boundaries = []

    for i, theme in enumerate(codebook.themes):
        theme_color = colors[i % len(colors)]
        theme_start = current_x

        for code in theme.codes:
            count = frequencies.get(code.name, 0)
            ax.bar(current_x, count, color=theme_color, width=0.8)
            tick_positions.append(current_x)
            tick_labels.append(code.name)
            current_x += 1

        theme_end = current_x
        theme_boundaries.append((theme_start, theme_end, theme.name, theme_color))

    # Add theme labels
    for start, end, name, color in theme_boundaries:
        mid = (start + end) / 2 - 0.5
        ax.text(
            mid,
            ax.get_ylim()[1] * 1.02,
            name,
            ha="center",
            fontsize=10,
            fontweight="bold",
            color=color,
        )

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Frequency")
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved theme frequency chart to {save_path}")

    return fig


def plot_pie_chart(
    frequencies: dict[str, int],
    title: str = "Code Distribution",
    top_n: int = 10,
    figsize: tuple[int, int] = (10, 8),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Create a pie chart of code frequencies.

    Args:
        frequencies: Dict mapping code name to count.
        title: Chart title.
        top_n: Number of top codes to show (rest grouped as "Other").
        figsize: Figure size.
        save_path: If set, save figure to this path.
        dpi: Resolution for saved figure.

    Returns:
        matplotlib Figure object.
    """
    sorted_items = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)

    if len(sorted_items) > top_n:
        top_items = sorted_items[:top_n]
        other_count = sum(item[1] for item in sorted_items[top_n:])
        top_items.append(("Other", other_count))
    else:
        top_items = sorted_items

    names = [item[0] for item in top_items]
    counts = [item[1] for item in top_items]

    fig, ax = plt.subplots(figsize=figsize)

    wedges, texts, autotexts = ax.pie(
        counts,
        labels=names,
        autopct="%1.1f%%",
        startangle=90,
    )

    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved pie chart to {save_path}")

    return fig
