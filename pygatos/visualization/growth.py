"""Codebook growth visualization over evaluation order."""

import logging
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from pygatos.core.codebook import Codebook

logger = logging.getLogger(__name__)


def plot_codebook_growth(
    codebook: Codebook,
    title: str = "Codebook Growth",
    figsize: tuple[int, int] = (12, 6),
    show_rejected: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Plot the growth of the codebook over evaluation order.

    Shows how the number of accepted codes grows as more codes are evaluated,
    which helps understand when the codebook reached saturation.

    Args:
        codebook: Codebook with evaluation_order metadata on codes.
        title: Chart title.
        figsize: Figure size.
        show_rejected: If True, also show cumulative rejected codes.
        save_path: If set, save figure to this path.
        dpi: Resolution for saved figure.

    Returns:
        matplotlib Figure object.
    """
    # Get codes sorted by evaluation order
    accepted_with_order = [
        (c.evaluation_order or 0, c)
        for c in codebook.accepted_codes
        if hasattr(c, "evaluation_order")
    ]
    accepted_with_order.sort(key=lambda x: x[0])

    rejected_with_order = [
        (c.evaluation_order or 0, c)
        for c in codebook.rejected_codes
        if hasattr(c, "evaluation_order")
    ]
    rejected_with_order.sort(key=lambda x: x[0])

    if not accepted_with_order and not rejected_with_order:
        logger.warning("No codes with evaluation_order found")
        # Fall back to simple index-based ordering
        accepted_with_order = [(i + 1, c) for i, c in enumerate(codebook.accepted_codes)]
        rejected_with_order = [(i + 1 + len(codebook.accepted_codes), c) for i, c in enumerate(codebook.rejected_codes)]

    # Combine and sort all codes by evaluation order
    all_codes = accepted_with_order + rejected_with_order
    all_codes.sort(key=lambda x: x[0])

    # Build cumulative counts
    eval_orders = []
    cumulative_accepted = []
    cumulative_rejected = []

    accepted_count = 0
    rejected_count = 0

    for order, code in all_codes:
        if code in [c for _, c in accepted_with_order]:
            accepted_count += 1
        else:
            rejected_count += 1

        eval_orders.append(order)
        cumulative_accepted.append(accepted_count)
        cumulative_rejected.append(rejected_count)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot accepted codes
    ax.plot(
        eval_orders,
        cumulative_accepted,
        color="green",
        linewidth=2,
        label=f"Accepted ({accepted_count})",
        marker="o",
        markersize=3,
    )

    if show_rejected:
        ax.plot(
            eval_orders,
            cumulative_rejected,
            color="red",
            linewidth=2,
            label=f"Rejected ({rejected_count})",
            marker="x",
            markersize=3,
            alpha=0.7,
        )

    ax.set_xlabel("Evaluation Order")
    ax.set_ylabel("Cumulative Count")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved growth chart to {save_path}")

    return fig


def plot_acceptance_rate(
    codebook: Codebook,
    title: str = "Rolling Acceptance Rate",
    window_size: int = 10,
    figsize: tuple[int, int] = (12, 6),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Plot the rolling acceptance rate over evaluation order.

    This shows how the acceptance rate changes over time, helping identify
    when the codebook starts to saturate (acceptance rate drops).

    Args:
        codebook: Codebook with evaluation_order metadata.
        title: Chart title.
        window_size: Size of rolling window for rate calculation.
        figsize: Figure size.
        save_path: If set, save figure to this path.
        dpi: Resolution for saved figure.

    Returns:
        matplotlib Figure object.
    """
    # Get all codes with evaluation order
    all_codes = []

    for c in codebook.accepted_codes:
        order = getattr(c, "evaluation_order", None) or 0
        all_codes.append((order, True))  # True = accepted

    for c in codebook.rejected_codes:
        order = getattr(c, "evaluation_order", None) or 0
        all_codes.append((order, False))  # False = rejected

    all_codes.sort(key=lambda x: x[0])

    if len(all_codes) < window_size:
        logger.warning(f"Not enough codes ({len(all_codes)}) for window size {window_size}")
        window_size = max(1, len(all_codes) // 3)

    # Calculate rolling acceptance rate
    eval_orders = []
    rolling_rates = []

    for i in range(window_size, len(all_codes) + 1):
        window = all_codes[i - window_size:i]
        accepted_in_window = sum(1 for _, is_accepted in window if is_accepted)
        rate = accepted_in_window / window_size * 100

        eval_orders.append(all_codes[i - 1][0])
        rolling_rates.append(rate)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        eval_orders,
        rolling_rates,
        color="blue",
        linewidth=2,
        label=f"Rolling Rate (window={window_size})",
    )

    # Add horizontal line at 50%
    ax.axhline(50, color="gray", linestyle="--", alpha=0.5, label="50% Rate")

    # Overall rate
    overall_rate = len(codebook.accepted_codes) / (len(codebook.accepted_codes) + len(codebook.rejected_codes)) * 100
    ax.axhline(
        overall_rate,
        color="green",
        linestyle=":",
        alpha=0.7,
        label=f"Overall Rate ({overall_rate:.1f}%)",
    )

    ax.set_xlabel("Evaluation Order")
    ax.set_ylabel("Acceptance Rate (%)")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved acceptance rate chart to {save_path}")

    return fig


def plot_saturation_analysis(
    codebook: Codebook,
    title: str = "Codebook Saturation Analysis",
    figsize: tuple[int, int] = (14, 5),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Create a multi-panel saturation analysis chart.

    Shows:
    1. Cumulative accepted codes
    2. Rolling acceptance rate
    3. Gap between suggested and accepted (deduplication)

    Args:
        codebook: Codebook with evaluation_order metadata.
        title: Overall title.
        figsize: Figure size.
        save_path: If set, save figure to this path.
        dpi: Resolution for saved figure.

    Returns:
        matplotlib Figure object.
    """
    # Get all codes sorted by evaluation order
    all_codes = []

    for c in codebook.accepted_codes:
        order = getattr(c, "evaluation_order", None) or 0
        all_codes.append((order, True, c))

    for c in codebook.rejected_codes:
        order = getattr(c, "evaluation_order", None) or 0
        all_codes.append((order, False, c))

    all_codes.sort(key=lambda x: x[0])

    if not all_codes:
        logger.warning("No codes found for saturation analysis")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        return fig

    # Calculate metrics
    eval_orders = []
    cumulative_accepted = []
    cumulative_total = []
    instant_rates = []

    accepted_count = 0
    total_count = 0
    window_size = 5

    for i, (order, is_accepted, code) in enumerate(all_codes):
        total_count += 1
        if is_accepted:
            accepted_count += 1

        eval_orders.append(order)
        cumulative_accepted.append(accepted_count)
        cumulative_total.append(total_count)

        # Rolling rate
        start_idx = max(0, i - window_size + 1)
        window = all_codes[start_idx:i + 1]
        rate = sum(1 for _, acc, _ in window if acc) / len(window) * 100
        instant_rates.append(rate)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Panel 1: Cumulative growth
    axes[0].fill_between(
        eval_orders,
        cumulative_accepted,
        alpha=0.3,
        color="green",
        label="Accepted",
    )
    axes[0].plot(eval_orders, cumulative_accepted, color="green", linewidth=2)
    axes[0].plot(
        eval_orders,
        cumulative_total,
        color="gray",
        linewidth=1,
        linestyle="--",
        label="Total Evaluated",
    )
    axes[0].set_xlabel("Evaluation Order")
    axes[0].set_ylabel("Cumulative Count")
    axes[0].set_title("Codebook Growth")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Rolling acceptance rate
    axes[1].plot(eval_orders, instant_rates, color="blue", linewidth=2)
    axes[1].axhline(50, color="gray", linestyle="--", alpha=0.5)
    axes[1].fill_between(
        eval_orders,
        instant_rates,
        50,
        where=[r > 50 for r in instant_rates],
        alpha=0.3,
        color="green",
        label="Above 50%",
    )
    axes[1].fill_between(
        eval_orders,
        instant_rates,
        50,
        where=[r <= 50 for r in instant_rates],
        alpha=0.3,
        color="red",
        label="Below 50%",
    )
    axes[1].set_xlabel("Evaluation Order")
    axes[1].set_ylabel("Acceptance Rate (%)")
    axes[1].set_title(f"Rolling Rate (window={window_size})")
    axes[1].set_ylim(0, 100)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Deduplication gap
    dedup_gap = [total - acc for total, acc in zip(cumulative_total, cumulative_accepted)]
    axes[2].stackplot(
        eval_orders,
        [cumulative_accepted, dedup_gap],
        labels=["Accepted", "Deduplicated"],
        colors=["green", "lightcoral"],
        alpha=0.7,
    )
    axes[2].set_xlabel("Evaluation Order")
    axes[2].set_ylabel("Cumulative Count")
    axes[2].set_title("Accepted vs Deduplicated")
    axes[2].legend(loc="upper left")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved saturation analysis to {save_path}")

    return fig
