"""Day-1 figures, exactly as predeclared in the preregistration's figure plan.

Five static figures. Which models and families appear is fixed by the design,
never by how favourably they performed: every judge with a complete run and all
three predicate families are always drawn.

Colour follows a validated categorical palette; because three of its slots sit
below 3:1 contrast on the light surface, every series carries a direct label and
every heatmap cell carries its exact number.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from formalcrrc import metrics  # noqa: E402
from formalcrrc.config import (  # noqa: E402
    DECISION_THRESHOLD,
    FAMILIES,
    FIGURE_DIR,
    MODEL_IDS,
    MODEL_SHORT_NAMES,
    N_STRICTNESS,
)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#86857f"
GRID = "#e3e2de"

#: Fixed categorical order; a judge keeps its colour whoever else is present.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")

#: Single-hue sequential ramp for magnitude (heatmap cells).
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
    ax.xaxis.grid(x_grid, color=GRID, linewidth=1.0)
    ax.set_axisbelow(True)


def _new_figure(width: float, height: float):
    fig = plt.figure(figsize=(width, height), facecolor=SURFACE, dpi=200)
    return fig


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Figure 1 -- concept schematic
# --------------------------------------------------------------------------


def figure1_concept(root: Path) -> Path:
    fig = _new_figure(11.0, 4.6)
    ax = fig.add_axes([0.0, 0.0, 0.615, 1.0])
    ax.set_xlim(0, 12.3)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, w, h, text, *, fc, ec, fontsize=9, color=INK, pad=0.12):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle=f"round,pad={pad},rounding_size=0.16",
                facecolor=fc,
                edgecolor=ec,
                linewidth=1.2,
            )
        )
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=color,
        )

    def arrow(x1, y1, x2, y2):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.4,
                color=INK_MUTED,
                shrinkA=0,
                shrinkB=0,
            )
        )

    mid = 5.6
    ax.text(0.1, 9.3, "Fixed artifact", fontsize=10, color=INK, weight="bold")
    box(
        0.1,
        mid - 1.35,
        3.1,
        2.7,
        "candidate\nresponse x",
        fc="#ffffff",
        ec=GRID,
        fontsize=10,
    )
    ax.text(
        1.65,
        mid - 1.95,
        "byte-identical\nat every level",
        fontsize=8.5,
        color=INK_SECONDARY,
        ha="center",
        va="top",
    )

    arrow(3.45, mid, 4.55, mid)

    ax.text(4.75, 9.3, "Ordered thresholds", fontsize=10, color=INK, weight="bold")
    chip_h, gap = 0.62, 0.16
    top = mid + (N_STRICTNESS * (chip_h + gap) - gap) / 2 - chip_h
    for i in range(N_STRICTNESS):
        y = top - i * (chip_h + gap)
        shade = SEQUENTIAL(0.10 + 0.75 * i / (N_STRICTNESS - 1))
        box(
            4.75,
            y,
            2.5,
            chip_h,
            f"R(s = {i})",
            fc=shade,
            ec=SURFACE,
            fontsize=8.5,
            color="#ffffff" if i >= 4 else INK,
            pad=0.0,
        )
    ax.text(
        6.0,
        mid - 4.0,
        "s = 0 most permissive  →  s = 8 most strict",
        fontsize=8.5,
        color=INK_SECONDARY,
        ha="center",
    )

    arrow(7.45, mid, 8.55, mid)

    ax.text(8.7, 9.3, "Judge", fontsize=10, color=INK, weight="bold")
    box(
        8.7,
        mid - 1.35,
        3.3,
        2.7,
        "P(criterion met\n| x, R(s))",
        fc="#ffffff",
        ec=GRID,
        fontsize=10,
    )
    ax.text(
        10.35,
        mid - 1.95,
        "one probability\nper threshold",
        fontsize=8.5,
        color=INK_SECONDARY,
        ha="center",
        va="top",
    )

    ax.text(
        0.1,
        0.75,
        "Only the numeric threshold changes. The label at every level is known "
        "by construction, before any model is consulted.",
        fontsize=8.5,
        color=INK_MUTED,
    )

    curve_ax = fig.add_axes([0.70, 0.20, 0.28, 0.62])
    _style_axes(curve_ax)
    strictness = np.arange(N_STRICTNESS)
    truth = (strictness < 4).astype(float)
    judge = np.array([0.97, 0.94, 0.88, 0.71, 0.42, 0.22, 0.11, 0.06, 0.04])

    curve_ax.step(
        strictness,
        truth,
        where="post",
        color=INK_MUTED,
        linewidth=2.0,
        linestyle=(0, (4, 2)),
    )
    curve_ax.plot(strictness, judge, color=SERIES_COLORS[0], linewidth=2.0)
    curve_ax.plot(
        strictness, judge, "o", markersize=5, color=SERIES_COLORS[0],
        markeredgecolor=SURFACE, markeredgewidth=1.4,
    )
    curve_ax.axhline(
        DECISION_THRESHOLD, color=GRID, linewidth=1.4, linestyle=(0, (2, 2))
    )
    curve_ax.text(
        1.55, 1.03, "formal truth", fontsize=8, color=INK_MUTED, ha="left"
    )
    curve_ax.text(
        0.15, 0.72, "judge curve", fontsize=8, color=SERIES_COLORS[0]
    )
    curve_ax.text(
        8.2, DECISION_THRESHOLD + 0.04, "0.5", fontsize=8, color=INK_MUTED,
        ha="right",
    )
    curve_ax.set_xlabel("strictness index s", fontsize=9, color=INK_SECONDARY)
    curve_ax.set_ylabel("P(met)", fontsize=9, color=INK_SECONDARY)
    curve_ax.set_ylim(-0.05, 1.10)
    curve_ax.set_xticks(strictness)
    curve_ax.set_title(
        "Counterfactual rubric\nresponse curve",
        fontsize=10,
        color=INK,
        loc="left",
        pad=8,
    )

    fig.suptitle(
        "FormalCRRC: one fixed artifact, nine formally controlled thresholds",
        fontsize=12.5,
        color=INK,
        x=0.01,
        ha="left",
        y=1.05,
    )
    return _save(fig, root / FIGURE_DIR / "figure1_concept.png")


# --------------------------------------------------------------------------
# Figure 2 -- boundary-aligned response curves
# --------------------------------------------------------------------------


def figure2_boundary_aligned(root: Path, scores: dict[str, pd.DataFrame]) -> Path:
    aligned = pd.concat(
        [metrics.boundary_aligned_curve(frame) for frame in scores.values()],
        ignore_index=True,
    )

    fig = _new_figure(9.0, 5.2)
    ax = fig.add_axes([0.09, 0.14, 0.72, 0.74])
    _style_axes(ax)

    distances = np.sort(aligned["signed_distance"].unique())
    truth = (distances < 0).astype(float)
    ax.step(
        distances,
        truth,
        where="post",
        color=INK_MUTED,
        linewidth=2.0,
        linestyle=(0, (4, 2)),
        zorder=1,
    )

    end_labels: list[list] = []
    for model_id in MODEL_IDS:
        if model_id not in scores:
            continue
        subset = aligned[aligned["model_id"] == model_id].sort_values(
            "signed_distance"
        )
        color = _color_for(model_id)
        x = subset["signed_distance"].to_numpy()
        y = subset["mean_p_met"].to_numpy()
        se = subset["se_p_met"].to_numpy()
        ax.fill_between(
            x, y - 1.96 * se, y + 1.96 * se, color=color, alpha=0.16, linewidth=0
        )
        ax.plot(x, y, color=color, linewidth=2.0, zorder=3)
        ax.plot(
            x,
            y,
            "o",
            markersize=4.5,
            color=color,
            markeredgecolor=SURFACE,
            markeredgewidth=1.2,
            zorder=4,
        )
        end_labels.append([float(y[-1]), MODEL_SHORT_NAMES[model_id], color])

    # Dodge the direct labels apart so a converged right edge stays readable.
    end_labels.sort(key=lambda item: item[0])
    min_gap = 0.052
    for i in range(1, len(end_labels)):
        if end_labels[i][0] - end_labels[i - 1][0] < min_gap:
            end_labels[i][0] = end_labels[i - 1][0] + min_gap
    for y_label, name, color in end_labels:
        ax.annotate(
            name,
            xy=(distances[-1], y_label),
            xytext=(7, 0),
            textcoords="offset points",
            fontsize=8.5,
            color=color,
            va="center",
            annotation_clip=False,
        )

    ax.axvline(0, color=GRID, linewidth=1.4)
    ax.axhline(
        DECISION_THRESHOLD, color=GRID, linewidth=1.2, linestyle=(0, (2, 2))
    )
    ax.set_xlabel(
        "signed distance from the true crossing,  d = s − j*",
        fontsize=10,
        color=INK_SECONDARY,
    )
    ax.set_ylabel("mean P(met)", fontsize=10, color=INK_SECONDARY)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xticks(distances[::1] if len(distances) <= 20 else distances[::2])
    ax.text(
        -0.25,
        1.01,
        "formally satisfied",
        fontsize=8.5,
        color=INK_MUTED,
        ha="right",
    )
    ax.text(0.25, 1.01, "formally failed", fontsize=8.5, color=INK_MUTED)
    ax.set_title(
        "Boundary-aligned response curves",
        fontsize=12,
        color=INK,
        loc="left",
        pad=10,
    )
    ax.legend(
        handles=[
            plt.Line2D([], [], color=_color_for(m), linewidth=2.0,
                       label=MODEL_SHORT_NAMES[m])
            for m in MODEL_IDS
            if m in scores
        ]
        + [
            plt.Line2D(
                [], [], color=INK_MUTED, linewidth=2.0, linestyle=(0, (4, 2)),
                label="formal truth",
            )
        ],
        frameon=False,
        fontsize=8.5,
        labelcolor=INK_SECONDARY,
        loc="lower left",
        bbox_to_anchor=(0.0, 0.0),
    )
    fig.text(
        0.09,
        0.02,
        "Bands are ±1.96 SE of the mean across artifacts. Dashed step is the "
        "label known by construction.",
        fontsize=8,
        color=INK_MUTED,
    )
    return _save(fig, root / FIGURE_DIR / "figure2_boundary_aligned_curves.png")


# --------------------------------------------------------------------------
# Figures 3 and 4 -- metric distributions
# --------------------------------------------------------------------------


def _distribution_panel(
    root: Path,
    artifact_metrics: dict[str, pd.DataFrame],
    column: str,
    values: np.ndarray,
    tick_labels: list[str],
    title: str,
    xlabel: str,
    caption: str,
    filename: str,
) -> Path:
    present = [m for m in MODEL_IDS if m in artifact_metrics]
    fig = _new_figure(3.0 * len(present) + 0.8, 3.6)
    axes = fig.subplots(1, len(present), sharey=True, squeeze=False)[0]

    for ax, model_id in zip(axes, present):
        _style_axes(ax)
        frame = artifact_metrics[model_id]
        series = frame[column].to_numpy(dtype=float)
        counts = np.array(
            [np.sum(np.isclose(series, value)) for value in values], dtype=float
        )
        share = counts / counts.sum() if counts.sum() else counts
        color = _color_for(model_id)
        ax.bar(
            np.arange(len(values)),
            share,
            width=0.72,
            color=color,
            edgecolor=SURFACE,
            linewidth=1.0,
        )
        # Selective direct labels: the few tallest bars only, never every bar.
        labelled = sorted(
            (i for i, h in enumerate(share) if h >= 0.04),
            key=lambda i: share[i],
            reverse=True,
        )[:4]
        for index in sorted(labelled):
            ax.text(
                index,
                share[index] + 0.015,
                f"{share[index]:.0%}",
                ha="center",
                fontsize=7.5,
                color=INK_SECONDARY,
            )
        ax.set_xticks(np.arange(len(values)))
        ax.set_xticklabels(tick_labels, fontsize=7.5)
        ax.set_title(
            f"{MODEL_SHORT_NAMES[model_id]}\nmean {series.mean():.3f}",
            fontsize=9.5,
            color=INK,
            loc="left",
            pad=8,
        )
        ax.set_ylim(0, 1.08)
        ax.set_xlabel(xlabel, fontsize=8.5, color=INK_SECONDARY)

    axes[0].set_ylabel("share of artifacts", fontsize=9, color=INK_SECONDARY)
    axes[0].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axes[0].set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    fig.suptitle(title, fontsize=12, color=INK, x=0.02, ha="left", y=1.04)
    fig.text(0.02, -0.06, caption, fontsize=8, color=INK_MUTED)
    return _save(fig, root / FIGURE_DIR / filename)


def figure3_mvr(root: Path, artifact_metrics: dict[str, pd.DataFrame]) -> Path:
    values = np.arange(N_STRICTNESS - 1 + 1) / (N_STRICTNESS - 1)
    labels = [f"{i}/8" for i in range(N_STRICTNESS)]
    return _distribution_panel(
        root,
        artifact_metrics,
        "mvr",
        values,
        labels,
        "Monotonicity violation rate per artifact (tolerance 0.05)",
        "MVR",
        "MVR counts the eight adjacent steps on which P(met) rises by more than "
        "0.05. 0/8 is a curve with no practical violation.",
        "figure3_mvr_distributions.png",
    )


def figure4_tce(root: Path, artifact_metrics: dict[str, pd.DataFrame]) -> Path:
    values = np.arange(N_STRICTNESS + 1)
    labels = [str(i) for i in range(N_STRICTNESS + 1)]
    return _distribution_panel(
        root,
        artifact_metrics,
        "tce",
        values,
        labels,
        "Threshold crossing error per artifact",
        "TCE (levels)",
        "TCE is |ĵ − j*|, the distance between the judge's first "
        "sub-0.5 level and the formal first-fail index. 0 is an exact boundary.",
        "figure4_tce_distributions.png",
    )


# --------------------------------------------------------------------------
# Figure 5 -- model x predicate-family summary
# --------------------------------------------------------------------------


def figure5_model_family(
    root: Path, artifact_metrics: dict[str, pd.DataFrame]
) -> Path:
    present = [m for m in MODEL_IDS if m in artifact_metrics]
    panels = [
        ("mvr", "MVR", "lower is better", "{:.3f}"),
        ("mvm", "MVM", "lower is better", "{:.3f}"),
        ("tce", "mean TCE", "lower is better", "{:.2f}"),
        ("accuracy", "accuracy", "higher is better", "{:.3f}"),
    ]

    fig = _new_figure(2.5 * len(panels) + 1.4, 0.85 * len(present) + 2.2)
    axes = fig.subplots(1, len(panels), squeeze=False)[0]
    fig.subplots_adjust(wspace=0.28)

    for index, (ax, (column, label, direction, fmt)) in enumerate(
        zip(axes, panels)
    ):
        grid = np.zeros((len(present), len(FAMILIES)))
        for row, model_id in enumerate(present):
            frame = artifact_metrics[model_id]
            for col, family in enumerate(FAMILIES):
                grid[row, col] = frame[frame["family"] == family][column].mean()

        span = grid.max() - grid.min()
        normed = (grid - grid.min()) / span if span > 0 else np.zeros_like(grid)
        ax.imshow(normed, cmap=SEQUENTIAL, vmin=0.0, vmax=1.0, aspect="auto")

        for row in range(len(present)):
            for col in range(len(FAMILIES)):
                ax.text(
                    col,
                    row,
                    fmt.format(grid[row, col]),
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="#ffffff" if normed[row, col] > 0.55 else INK,
                )

        ax.set_xticks(range(len(FAMILIES)))
        ax.set_xticklabels(
            [FAMILY_LABELS[f] for f in FAMILIES], fontsize=8, color=INK_SECONDARY
        )
        ax.set_yticks(range(len(present)))
        ax.set_yticklabels(
            [MODEL_SHORT_NAMES[m] for m in present] if index == 0 else [],
            fontsize=9,
            color=INK_SECONDARY,
        )
        ax.set_title(f"{label}\n{direction}", fontsize=9.5, color=INK, pad=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)

    fig.suptitle(
        "Primary metrics by judge and predicate family",
        fontsize=12,
        color=INK,
        x=0.02,
        ha="left",
        y=1.02,
    )
    fig.text(
        0.02,
        -0.02,
        "Shading is normalised within each panel; every cell carries its exact "
        "value. All judges with a complete run and all three families are shown.",
        fontsize=8,
        color=INK_MUTED,
    )
    return _save(fig, root / FIGURE_DIR / "figure5_model_family_heatmap.png")


# --------------------------------------------------------------------------


def render_all(
    root: Path,
    scores: dict[str, pd.DataFrame],
    artifact_metrics: dict[str, pd.DataFrame],
    bootstrap_table: pd.DataFrame | None = None,
) -> list[Path]:
    """Render every predeclared figure and return the paths written."""
    paths = [
        figure1_concept(root),
        figure2_boundary_aligned(root, scores),
        figure3_mvr(root, artifact_metrics),
        figure4_tce(root, artifact_metrics),
        figure5_model_family(root, artifact_metrics),
    ]
    for path in paths:
        print(f"wrote {path.relative_to(root)}", flush=True)
    return paths
