"""Day-2 figures, exactly as predeclared in the preregistration's figure plan.

Six static figures. Which judges and families appear is fixed by the design,
never by how favourably they performed.

Colour follows the same validated categorical palette as Day 1, so a judge keeps
its identity across both experiments. Raw and corrected series are distinguished
by line style as well as colour, and every heatmap cell carries its exact value.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from formalcrrc import day2  # noqa: E402
from formalcrrc.config import (  # noqa: E402
    FAMILIES,
    MODEL_IDS,
    MODEL_SHORT_NAMES,
    N_STRICTNESS,
)
from formalcrrc.day2_config import FIGURE_DIR  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#86857f"
GRID = "#e3e2de"

#: Same fixed categorical order as Day 1; a judge keeps its colour.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "formalcrrc_blue",
    ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"],
)

FAMILY_LABELS = {
    "coverage": "A\ncoverage",
    "max_violation": "B\nmax viol.",
    "numeric_tolerance": "C\nnum. tol.",
}


def _color_for(model_id: str) -> str:
    return SERIES_COLORS[MODEL_IDS.index(model_id) % len(SERIES_COLORS)]


def _style_axes(ax: plt.Axes, *, x_grid: bool = False) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=3, width=1.0)
    ax.yaxis.grid(True, color=GRID, linewidth=1.0)
    if x_grid:
        ax.xaxis.grid(True, color=GRID, linewidth=1.0)
    ax.set_axisbelow(True)


def _new_figure(width: float, height: float) -> plt.Figure:
    return plt.figure(figsize=(width, height), facecolor=SURFACE, dpi=200)


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return path


def _present(container) -> list[str]:
    return [m for m in MODEL_IDS if m in container]


# --------------------------------------------------------------------------
# Figure 1 -- bias-vs-shape concept
# --------------------------------------------------------------------------


def figure1_bias_vs_shape(root: Path) -> Path:
    strictness = np.arange(N_STRICTNESS)
    # A curve with a deliberate local reversal at s = 2 -> 3.
    margin = np.array([3.2, 2.1, 0.9, 1.4, 0.2, -0.6, -1.5, -2.4, -3.1])
    alpha = -1.6

    fig = _new_figure(11.0, 4.3)
    axes = fig.subplots(1, 3)

    # -- margin space --------------------------------------------------------
    ax = axes[0]
    _style_axes(ax)
    ax.axhline(0.0, color=INK_MUTED, linewidth=1.4, linestyle=(0, (4, 2)))
    ax.plot(strictness, margin, color=SERIES_COLORS[0], linewidth=2.0, label="raw M(s)")
    ax.plot(
        strictness,
        margin + alpha,
        color=SERIES_COLORS[1],
        linewidth=2.0,
        linestyle=(0, (5, 2)),
        label=f"M(s) + α   (α = {alpha})",
    )
    for series, color in ((margin, SERIES_COLORS[0]), (margin + alpha, SERIES_COLORS[1])):
        ax.plot(
            strictness, series, "o", markersize=4.5, color=color,
            markeredgecolor=SURFACE, markeredgewidth=1.2,
        )
    ax.annotate(
        "reversal survives\nthe shift",
        xy=(3, margin[3]),
        xytext=(3.4, 3.0),
        fontsize=8,
        color=INK_SECONDARY,
        arrowprops=dict(arrowstyle="->", color=INK_MUTED, linewidth=1.0),
    )
    ax.set_xlabel("strictness index s", fontsize=9, color=INK_SECONDARY)
    ax.set_ylabel("decision margin  M = S$_A$ − S$_B$", fontsize=9, color=INK_SECONDARY)
    ax.set_xticks(strictness)
    ax.set_title("1. Margin space", fontsize=10.5, color=INK, loc="left", pad=8)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, loc="lower left")

    # -- adjacent differences ------------------------------------------------
    ax = axes[1]
    _style_axes(ax)
    steps = np.diff(margin)
    shifted_steps = np.diff(margin + alpha)
    width = 0.38
    positions = np.arange(len(steps))
    ax.bar(
        positions - width / 2, steps, width, color=SERIES_COLORS[0],
        edgecolor=SURFACE, linewidth=1.0, label="raw",
    )
    ax.bar(
        positions + width / 2, shifted_steps, width, color=SERIES_COLORS[1],
        edgecolor=SURFACE, linewidth=1.0, label="shifted",
    )
    ax.axhline(0.0, color=GRID, linewidth=1.2)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{i}→{i+1}" for i in positions], fontsize=7.5)
    ax.set_xlabel("adjacent step", fontsize=9, color=INK_SECONDARY)
    ax.set_ylabel("M(s+1) − M(s)", fontsize=9, color=INK_SECONDARY)
    ax.set_title(
        "2. Differences are identical", fontsize=10.5, color=INK, loc="left", pad=8
    )
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    ax.text(
        0.5,
        max(steps.max(), 0.1) * 0.95,
        "shape metrics unchanged:\nMMVR, MMVM, SPAN, Spearman",
        fontsize=8,
        color=INK_MUTED,
    )

    # -- probability space ---------------------------------------------------
    ax = axes[2]
    _style_axes(ax)
    p_raw = day2.sigmoid(margin)
    p_shift = day2.sigmoid(margin + alpha)
    ax.axhline(0.5, color=GRID, linewidth=1.4, linestyle=(0, (2, 2)))
    ax.plot(strictness, p_raw, color=SERIES_COLORS[0], linewidth=2.0, label="raw P")
    ax.plot(
        strictness, p_shift, color=SERIES_COLORS[1], linewidth=2.0,
        linestyle=(0, (5, 2)), label="corrected P",
    )
    j_raw = day2.predicted_first_fail_from_margin(margin)
    j_shift = day2.predicted_first_fail_from_margin(margin, alpha)
    ax.axvline(j_raw, color=SERIES_COLORS[0], linewidth=1.2, alpha=0.4)
    ax.axvline(j_shift, color=SERIES_COLORS[1], linewidth=1.2, alpha=0.4)
    ax.annotate(
        f"crossing {j_raw} → {j_shift}",
        xy=(0.5 * (j_raw + j_shift), 0.55),
        fontsize=8,
        color=INK_SECONDARY,
        ha="center",
    )
    ax.set_xlabel("strictness index s", fontsize=9, color=INK_SECONDARY)
    ax.set_ylabel("P(met)", fontsize=9, color=INK_SECONDARY)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(strictness)
    ax.set_title(
        "3. Only the boundary moves", fontsize=10.5, color=INK, loc="left", pad=8
    )
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, loc="lower left")

    fig.suptitle(
        "A global intercept moves a curve's location and cannot change its shape",
        fontsize=12.5, color=INK, x=0.01, ha="left", y=1.04,
    )
    fig.text(
        0.01, -0.04,
        "A vertical shift in margin space leaves every adjacent difference "
        "identical, so a local reversal cannot be calibrated away.",
        fontsize=8, color=INK_MUTED,
    )
    fig.tight_layout()
    return _save(fig, root / FIGURE_DIR / "figure1_bias_vs_shape.png")


# --------------------------------------------------------------------------
# Figure 2 -- estimated global offsets
# --------------------------------------------------------------------------


def figure2_global_offsets(root: Path, parameters: dict, samples: dict) -> Path:
    present = _present(samples)
    fig = _new_figure(8.6, 0.85 * len(present) + 2.6)
    ax = fig.add_subplot(111)
    _style_axes(ax, x_grid=True)
    ax.yaxis.grid(False)

    ax.axvline(0.0, color=INK_MUTED, linewidth=1.4, linestyle=(0, (4, 2)))
    ax.text(
        0.0, len(present) - 0.35, "  no global offset",
        fontsize=8, color=INK_MUTED, va="center",
    )

    for row, model_id in enumerate(present):
        color = _color_for(model_id)
        block = parameters["parameters"][model_id]
        alpha = block["global"]["alpha"]
        draws = samples[model_id]["alpha_global"]
        low, high = np.quantile(draws[np.isfinite(draws)], [0.025, 0.975])
        y = len(present) - 1 - row

        ax.plot([low, high], [y, y], color=color, linewidth=2.5, solid_capstyle="round")
        ax.plot(
            [alpha], [y], "o", markersize=9, color=color,
            markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=3,
        )
        for offset, family in zip((0.24, 0.30, 0.36), FAMILIES):
            ax.plot(
                [block["family"][family]["alpha"]], [y - offset], "|",
                markersize=9, color=color, alpha=0.75,
            )
        ax.annotate(
            f"{alpha:+.3f}  [{low:+.3f}, {high:+.3f}]",
            xy=(high, y), xytext=(8, 0), textcoords="offset points",
            fontsize=8.5, color=INK_SECONDARY, va="center",
        )

    ax.set_yticks(range(len(present)))
    ax.set_yticklabels(
        [MODEL_SHORT_NAMES[m] for m in reversed(present)],
        fontsize=9.5, color=INK_SECONDARY,
    )
    ax.set_ylim(-0.7, len(present) - 0.2)
    ax.set_xlabel(
        "fitted global intercept α  (margin units)", fontsize=10, color=INK_SECONDARY
    )
    ax.set_title(
        "Estimated global decision-margin offsets",
        fontsize=12, color=INK, loc="left", pad=10,
    )
    fig.text(
        0.02, -0.02,
        "Circle: point estimate fitted on CALIBRATION. Bar: 95% two-stage "
        "bootstrap interval. Ticks below each row: the three family-specific "
        "intercepts (secondary). α > 0 means the raw judge is globally too "
        "pessimistic about 'criterion met'.",
        fontsize=8, color=INK_MUTED, wrap=True,
    )
    return _save(fig, root / FIGURE_DIR / "figure2_global_offsets.png")


# --------------------------------------------------------------------------
# Figure 3 -- raw versus globally corrected TCE
# --------------------------------------------------------------------------


def figure3_paired_tce(root: Path, metrics: dict) -> Path:
    present = _present(metrics)
    fig = _new_figure(3.0 * len(present) + 0.8, 3.9)
    axes = fig.subplots(1, len(present), sharey=True, squeeze=False)[0]

    levels = np.arange(N_STRICTNESS + 1)
    for ax, model_id in zip(axes, present):
        _style_axes(ax)
        frame = metrics[model_id]
        color = _color_for(model_id)
        raw = frame["tce_raw"].to_numpy()
        corrected = frame["tce_global"].to_numpy()
        n = len(frame)

        raw_share = np.array([np.mean(raw == k) for k in levels])
        cor_share = np.array([np.mean(corrected == k) for k in levels])
        width = 0.4
        ax.bar(
            levels - width / 2, raw_share, width, color=color, alpha=0.45,
            edgecolor=SURFACE, linewidth=1.0, label="raw",
        )
        ax.bar(
            levels + width / 2, cor_share, width, color=color,
            edgecolor=SURFACE, linewidth=1.0, label="global-corrected",
        )
        improved = int(np.sum(corrected < raw))
        worsened = int(np.sum(corrected > raw))
        ax.set_title(
            f"{MODEL_SHORT_NAMES[model_id]}\n"
            f"mean {raw.mean():.2f} → {corrected.mean():.2f}\n"
            f"better {improved}/{n} · worse {worsened}/{n}",
            fontsize=9.5, color=INK, loc="left", pad=8,
        )
        ax.set_xticks(levels)
        ax.set_xticklabels([str(k) for k in levels], fontsize=7.5)
        ax.set_xlabel("TCE (levels)", fontsize=8.5, color=INK_SECONDARY)
        ax.set_ylim(0, 1.0)

    axes[0].set_ylabel("share of artifacts", fontsize=9, color=INK_SECONDARY)
    axes[0].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axes[0].set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    fig.suptitle(
        "Held-out threshold crossing error, raw versus globally corrected",
        fontsize=12, color=INK, x=0.02, ha="left", y=1.06,
    )
    fig.text(
        0.02, -0.08,
        "Paired at the artifact level on the TEST partition. The intercept was "
        "fitted on CALIBRATION only.",
        fontsize=8, color=INK_MUTED,
    )
    return _save(fig, root / FIGURE_DIR / "figure3_paired_tce.png")


# --------------------------------------------------------------------------
# Figure 4 -- boundary-aligned held-out curves
# --------------------------------------------------------------------------


def figure4_boundary_aligned(
    root: Path, test_scores: dict, parameters: dict
) -> Path:
    present = _present(test_scores)
    aligned = pd.concat(
        [
            day2.boundary_aligned_curves(
                test_scores[m], parameters["parameters"][m]["global"]["alpha"]
            )
            for m in present
        ],
        ignore_index=True,
    )

    fig = _new_figure(3.0 * len(present) + 0.8, 4.1)
    axes = fig.subplots(1, len(present), sharey=True, squeeze=False)[0]

    for ax, model_id in zip(axes, present):
        _style_axes(ax)
        subset = aligned[aligned["model_id"] == model_id].sort_values(
            "signed_distance"
        )
        d = subset["signed_distance"].to_numpy()
        color = _color_for(model_id)

        truth = (d < 0).astype(float)
        ax.step(
            d, truth, where="post", color=INK_MUTED, linewidth=2.0,
            linestyle=(0, (4, 2)),
        )
        for column, style, label in (
            ("mean_p_raw", "-", "raw"),
            ("mean_p_global", (0, (5, 2)), "corrected"),
        ):
            y = subset[column].to_numpy()
            se = subset[column.replace("mean", "se")].to_numpy()
            ax.plot(d, y, color=color, linewidth=2.0, linestyle=style, label=label)
            ax.fill_between(
                d, y - 1.96 * se, y + 1.96 * se, color=color, alpha=0.14, linewidth=0
            )
        ax.axvline(0, color=GRID, linewidth=1.4)
        ax.axhline(0.5, color=GRID, linewidth=1.2, linestyle=(0, (2, 2)))
        ax.set_title(MODEL_SHORT_NAMES[model_id], fontsize=10, color=INK, loc="left", pad=8)
        ax.set_xlabel("d = s − j*", fontsize=9, color=INK_SECONDARY)
        ax.set_ylim(-0.03, 1.05)
        ax.set_xticks(d[::2])

    axes[0].set_ylabel("mean P(met)", fontsize=9.5, color=INK_SECONDARY)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, loc="lower left")
    fig.suptitle(
        "Boundary-aligned held-out curves: raw, corrected, and formal truth",
        fontsize=12, color=INK, x=0.02, ha="left", y=1.04,
    )
    fig.text(
        0.02, -0.06,
        "Dashed grey step is the label known by construction. Bands are ±1.96 SE. "
        "The correction slides a curve vertically; it does not reshape it.",
        fontsize=8, color=INK_MUTED,
    )
    return _save(fig, root / FIGURE_DIR / "figure4_boundary_aligned.png")


# --------------------------------------------------------------------------
# Figure 5 -- residual localisation error
# --------------------------------------------------------------------------


def figure5_residual_tce(root: Path, metrics: dict) -> Path:
    present = _present(metrics)
    fig = _new_figure(3.0 * len(present) + 0.8, 3.6)
    axes = fig.subplots(1, len(present), sharey=True, squeeze=False)[0]
    levels = np.arange(N_STRICTNESS + 1)

    for ax, model_id in zip(axes, present):
        _style_axes(ax)
        frame = metrics[model_id]
        corrected = frame["tce_global"].to_numpy()
        share = np.array([np.mean(corrected == k) for k in levels])
        color = _color_for(model_id)
        ax.bar(levels, share, width=0.72, color=color, edgecolor=SURFACE, linewidth=1.0)

        labelled = sorted(
            (i for i, h in enumerate(share) if h >= 0.04),
            key=lambda i: share[i],
            reverse=True,
        )[:4]
        for index in sorted(labelled):
            ax.text(
                index, share[index] + 0.015, f"{share[index]:.0%}",
                ha="center", fontsize=7.5, color=INK_SECONDARY,
            )
        residual = float(np.mean(corrected > 1))
        ax.set_title(
            f"{MODEL_SHORT_NAMES[model_id]}\nmean {corrected.mean():.2f} · "
            f"residual >1: {residual:.0%}",
            fontsize=9.5, color=INK, loc="left", pad=8,
        )
        ax.set_xticks(levels)
        ax.set_xticklabels([str(k) for k in levels], fontsize=7.5)
        ax.set_xlabel("TCE after correction", fontsize=8.5, color=INK_SECONDARY)
        ax.set_ylim(0, 1.0)

    axes[0].set_ylabel("share of artifacts", fontsize=9, color=INK_SECONDARY)
    axes[0].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axes[0].set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    fig.suptitle(
        "Residual localisation error after global intercept correction",
        fontsize=12, color=INK, x=0.02, ha="left", y=1.05,
    )
    fig.text(
        0.02, -0.07,
        "What survives a shape-preserving correction of curve location. This "
        "residual is the scientific result, not a shortfall to be calibrated away.",
        fontsize=8, color=INK_MUTED,
    )
    return _save(fig, root / FIGURE_DIR / "figure5_residual_tce.png")


# --------------------------------------------------------------------------
# Figure 6 -- model x family decomposition
# --------------------------------------------------------------------------


def figure6_model_family(root: Path, metrics: dict) -> Path:
    present = _present(metrics)
    panels = [
        ("tce_raw", "raw TCE", "lower is better", "{:.2f}"),
        ("tce_global", "global TCE", "lower is better", "{:.2f}"),
        ("tce_family", "family TCE", "lower is better", "{:.2f}"),
        ("mmvr", "MMVR", "shape, invariant", "{:.3f}"),
    ]

    fig = _new_figure(2.5 * len(panels) + 1.4, 0.85 * len(present) + 2.3)
    axes = fig.subplots(1, len(panels), squeeze=False)[0]
    fig.subplots_adjust(wspace=0.28)

    for index, (ax, (column, label, direction, fmt)) in enumerate(zip(axes, panels)):
        grid = np.zeros((len(present), len(FAMILIES)))
        for row, model_id in enumerate(present):
            frame = metrics[model_id]
            for col, family in enumerate(FAMILIES):
                grid[row, col] = frame[frame["family"] == family][column].mean()

        span = grid.max() - grid.min()
        normed = (grid - grid.min()) / span if span > 0 else np.zeros_like(grid)
        ax.imshow(normed, cmap=SEQUENTIAL, vmin=0.0, vmax=1.0, aspect="auto")

        for row in range(len(present)):
            for col in range(len(FAMILIES)):
                ax.text(
                    col, row, fmt.format(grid[row, col]),
                    ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if normed[row, col] > 0.55 else INK,
                )

        ax.set_xticks(range(len(FAMILIES)))
        ax.set_xticklabels(
            [FAMILY_LABELS[f] for f in FAMILIES], fontsize=8, color=INK_SECONDARY
        )
        ax.set_yticks(range(len(present)))
        ax.set_yticklabels(
            [MODEL_SHORT_NAMES[m] for m in present] if index == 0 else [],
            fontsize=9, color=INK_SECONDARY,
        )
        ax.set_title(f"{label}\n{direction}", fontsize=9.5, color=INK, pad=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)

    fig.suptitle(
        "Held-out decomposition by judge and predicate family",
        fontsize=12, color=INK, x=0.02, ha="left", y=1.03,
    )
    fig.text(
        0.02, -0.03,
        "Shading is normalised within each panel; every cell carries its exact "
        "value. MMVR is computed on raw margins and is invariant to any "
        "intercept, so it is identical before and after correction.",
        fontsize=8, color=INK_MUTED,
    )
    return _save(fig, root / FIGURE_DIR / "figure6_model_family.png")


# --------------------------------------------------------------------------


def render_all(
    root: Path,
    test_scores: dict,
    metrics: dict,
    parameters: dict,
    samples: dict,
) -> list[Path]:
    """Render every predeclared Day-2 figure and return the paths written."""
    paths = [
        figure1_bias_vs_shape(root),
        figure2_global_offsets(root, parameters, samples),
        figure3_paired_tce(root, metrics),
        figure4_boundary_aligned(root, test_scores, parameters),
        figure5_residual_tce(root, metrics),
        figure6_model_family(root, metrics),
    ]
    for path in paths:
        print(f"wrote {path.relative_to(root)}", flush=True)
    return paths
