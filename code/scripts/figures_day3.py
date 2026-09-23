"""Day-3 figures, exactly as predeclared in the preregistration's figure plan.

Six static figures. Which judges and families appear is fixed by the design,
never by how favourably they performed.

Colour follows the same validated categorical palette as Days 1 and 2, so a
judge keeps its identity across all three experiments.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from formalcrrc import day3  # noqa: E402
from formalcrrc.config import (  # noqa: E402
    FAMILIES,
    MODEL_IDS,
    MODEL_SHORT_NAMES,
    N_STRICTNESS,
)
from formalcrrc.day3_config import FIGURE_DIR  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#86857f"
GRID = "#e3e2de"

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
# Figure 1 -- translation reachability concept
# --------------------------------------------------------------------------


def figure1_reachability_concept(root: Path) -> Path:
    strictness = np.arange(N_STRICTNESS)
    curve = np.array([3.0, 3.6, 1.4, 2.0, 0.6, 1.1, 1.5, -0.9, -0.4])
    records = day3.prefix_record_lows(curve)
    reachable = day3.reachable_crossings(curve)
    target = 5  # deliberately unreachable

    fig = _new_figure(11.0, 4.6)
    axes = fig.subplots(1, 2, width_ratios=[1.35, 1.0])

    # -- margin curve with record lows --------------------------------------
    ax = axes[0]
    _style_axes(ax)
    running = np.minimum.accumulate(curve)
    ax.plot(
        strictness, running, color=INK_MUTED, linewidth=1.6,
        linestyle=(0, (2, 2)), label="running minimum",
    )
    ax.plot(strictness, curve, color=SERIES_COLORS[0], linewidth=2.2, label="M(s)")
    ax.plot(
        strictness, curve, "o", markersize=6, color=SERIES_COLORS[0],
        markeredgecolor=SURFACE, markeredgewidth=1.4,
    )
    ax.plot(
        list(records), curve[list(records)], "o", markersize=11,
        markerfacecolor="none", markeredgecolor=SERIES_COLORS[1], markeredgewidth=2.2,
        label="strict prefix record low",
    )
    ax.axhline(0.0, color=GRID, linewidth=1.4)
    ax.text(8.15, 0.06, "M = 0", fontsize=8, color=INK_MUTED, ha="right")

    ax.annotate(
        f"j* = {target} is NOT a record low,\nso no translation can cross here",
        xy=(target, curve[target]),
        xytext=(2.6, 3.4),
        fontsize=8.5,
        color=INK_SECONDARY,
        arrowprops=dict(arrowstyle="->", color=INK_MUTED, linewidth=1.1),
    )
    ax.set_xlabel("strictness index s", fontsize=9.5, color=INK_SECONDARY)
    ax.set_ylabel("decision margin  M = S$_A$ − S$_B$", fontsize=9.5, color=INK_SECONDARY)
    ax.set_xticks(strictness)
    ax.set_title(
        "A translation slides the curve; it cannot reorder it",
        fontsize=10.5, color=INK, loc="left", pad=8,
    )
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, loc="lower left")

    # -- reachable set ------------------------------------------------------
    ax = axes[1]
    _style_axes(ax, x_grid=True)
    ax.yaxis.grid(False)
    for j in range(N_STRICTNESS + 1):
        is_reachable = j in reachable
        ax.barh(
            j,
            1.0,
            height=0.62,
            color=SERIES_COLORS[0] if is_reachable else "#ffffff",
            edgecolor=SERIES_COLORS[0] if is_reachable else GRID,
            linewidth=1.4,
        )
        if j in (0, N_STRICTNESS):
            ax.text(
                1.06, j, "always reachable", fontsize=7.5, color=INK_MUTED,
                va="center",
            )
    ax.plot(
        [1.03], [target], marker="<", markersize=11, color=SERIES_COLORS[1],
        clip_on=False,
    )
    ax.text(
        1.09, target, "j* — unreachable", fontsize=9, color=SERIES_COLORS[1],
        va="center", ha="left", weight="bold",
    )
    ax.set_yticks(range(N_STRICTNESS + 1))
    ax.set_yticklabels([str(j) for j in range(N_STRICTNESS + 1)], fontsize=8.5)
    ax.set_xticks([])
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.7, N_STRICTNESS + 0.7)
    ax.set_ylabel("crossing position j", fontsize=9.5, color=INK_SECONDARY)
    ax.set_title(
        f"Reachable set  R = {{{', '.join(str(j) for j in reachable)}}}",
        fontsize=10.5, color=INK, loc="left", pad=8,
    )
    otce = min(abs(j - target) for j in reachable)
    ax.text(
        0.5, -0.55, f"OTCE = {otce}", fontsize=9.5, color=SERIES_COLORS[1],
        ha="center", weight="bold",
    )

    fig.suptitle(
        "Translation reachability: only strict prefix record lows can become the crossing",
        fontsize=12.5, color=INK, x=0.01, ha="left", y=1.03,
    )
    fig.text(
        0.01, -0.05,
        "Filled bars are crossing positions some additive translation can produce. "
        "Crossings 0 and 9 are always available; every other position must be a "
        "strict new minimum of the raw curve.",
        fontsize=8, color=INK_MUTED,
    )
    fig.tight_layout()
    return _save(fig, root / FIGURE_DIR / "figure1_reachability_concept.png")


# --------------------------------------------------------------------------
# Figure 2 -- raw versus oracle-artifact TCE
# --------------------------------------------------------------------------


def figure2_raw_vs_otce(root: Path, metrics: dict) -> Path:
    present = _present(metrics)
    fig = _new_figure(3.0 * len(present) + 0.8, 3.9)
    axes = fig.subplots(1, len(present), sharey=True, squeeze=False)[0]
    levels = np.arange(N_STRICTNESS + 1)

    for ax, model_id in zip(axes, present):
        _style_axes(ax)
        frame = metrics[model_id]
        color = _color_for(model_id)
        raw = frame["tce_raw"].to_numpy()
        otce = frame["otce"].to_numpy()

        width = 0.4
        ax.bar(
            levels - width / 2,
            [np.mean(raw == k) for k in levels],
            width, color=color, alpha=0.45, edgecolor=SURFACE, linewidth=1.0,
            label="raw TCE",
        )
        ax.bar(
            levels + width / 2,
            [np.mean(otce == k) for k in levels],
            width, color=color, edgecolor=SURFACE, linewidth=1.0,
            label="oracle OTCE",
        )
        ax.set_title(
            f"{MODEL_SHORT_NAMES[model_id]}\n"
            f"mean {raw.mean():.2f} → {otce.mean():.2f}\n"
            f"OTCE > 1: {np.mean(otce > 1):.0%}",
            fontsize=9.5, color=INK, loc="left", pad=8,
        )
        ax.set_xticks(levels)
        ax.set_xticklabels([str(k) for k in levels], fontsize=7.5)
        ax.set_xlabel("levels from the formal boundary", fontsize=8.5, color=INK_SECONDARY)
        ax.set_ylim(0, 1.0)

    axes[0].set_ylabel("share of artifacts", fontsize=9, color=INK_SECONDARY)
    axes[0].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axes[0].set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    fig.suptitle(
        "Raw localisation versus the artifact-level translation floor",
        fontsize=12, color=INK, x=0.02, ha="left", y=1.08,
    )
    fig.text(
        0.02, -0.09,
        "OTCE is the best any additive translation could do for each individual "
        "curve, chosen with knowledge of the formal truth. It is a diagnostic "
        "bound, not a deployable correction.",
        fontsize=8, color=INK_MUTED,
    )
    return _save(fig, root / FIGURE_DIR / "figure2_raw_vs_otce.png")


# --------------------------------------------------------------------------
# Figure 3 -- translation reachability
# --------------------------------------------------------------------------


def figure3_reachability(root: Path, metrics: dict) -> Path:
    present = _present(metrics)
    fig = _new_figure(9.6, 4.4)
    axes = fig.subplots(1, 2, width_ratios=[1.0, 1.25])

    ax = axes[0]
    _style_axes(ax)
    values, colors, labels = [], [], []
    for model_id in present:
        frame = metrics[model_id]
        nontrivial = frame[frame["nontrivial_boundary"]]
        values.append(float(nontrivial["translation_reachable"].mean()))
        colors.append(_color_for(model_id))
        labels.append(MODEL_SHORT_NAMES[model_id])
    positions = np.arange(len(present))
    ax.bar(positions, values, width=0.62, color=colors, edgecolor=SURFACE, linewidth=1.0)
    for x, v in zip(positions, values):
        ax.text(x, v + 0.015, f"{v:.1%}", ha="center", fontsize=8.5, color=INK_SECONDARY)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8.5, rotation=12, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("nontrivial TRR", fontsize=9.5, color=INK_SECONDARY)
    ax.set_title("By judge", fontsize=10.5, color=INK, loc="left", pad=8)

    ax = axes[1]
    _style_axes(ax)
    width = 0.8 / len(present)
    base = np.arange(len(FAMILIES))
    for index, model_id in enumerate(present):
        frame = metrics[model_id]
        nontrivial = frame[frame["nontrivial_boundary"]]
        heights = [
            float(
                nontrivial[nontrivial["family"] == family][
                    "translation_reachable"
                ].mean()
            )
            for family in FAMILIES
        ]
        ax.bar(
            base + index * width - 0.4 + width / 2,
            heights,
            width * 0.92,
            color=_color_for(model_id),
            edgecolor=SURFACE,
            linewidth=0.8,
            label=MODEL_SHORT_NAMES[model_id],
        )
    ax.set_xticks(base)
    ax.set_xticklabels([FAMILY_LABELS[f] for f in FAMILIES], fontsize=8.5)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_title("By predicate family", fontsize=10.5, color=INK, loc="left", pad=8)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, ncol=2)

    fig.suptitle(
        "Fraction of nontrivial formal boundaries reachable by translation",
        fontsize=12, color=INK, x=0.02, ha="left", y=1.02,
    )
    fig.text(
        0.02, -0.04,
        "Restricted to j* in 1..8. The boundary j* = 9 is always reachable and is "
        "excluded so it cannot inflate the headline number.",
        fontsize=8, color=INK_MUTED,
    )
    fig.tight_layout()
    return _save(fig, root / FIGURE_DIR / "figure3_reachability.png")


# --------------------------------------------------------------------------
# Figure 4 -- reachable crossing geometry
# --------------------------------------------------------------------------


def figure4_crossing_geometry(root: Path, metrics: dict) -> Path:
    present = _present(metrics)
    boundaries = list(range(1, N_STRICTNESS + 1))
    grid = np.zeros((len(present), len(boundaries)))
    for row, model_id in enumerate(present):
        frame = metrics[model_id]
        for col, j in enumerate(boundaries):
            subset = frame[frame["true_first_fail_index"] == j]
            grid[row, col] = (
                float(subset["translation_reachable"].mean()) if len(subset) else np.nan
            )

    fig = _new_figure(1.0 * len(boundaries) + 3.0, 0.85 * len(present) + 2.4)
    ax = fig.add_subplot(111)
    ax.imshow(grid, cmap=SEQUENTIAL, vmin=0.0, vmax=1.0, aspect="auto")

    for row in range(len(present)):
        for col in range(len(boundaries)):
            value = grid[row, col]
            if np.isnan(value):
                continue
            ax.text(
                col, row, f"{value:.0%}",
                ha="center", va="center", fontsize=8.5,
                color="#ffffff" if value > 0.55 else INK,
            )

    ax.set_xticks(range(len(boundaries)))
    ax.set_xticklabels([str(j) for j in boundaries], fontsize=9, color=INK_SECONDARY)
    ax.set_yticks(range(len(present)))
    ax.set_yticklabels(
        [MODEL_SHORT_NAMES[m] for m in present], fontsize=9.5, color=INK_SECONDARY
    )
    ax.set_xlabel("true formal boundary  j*", fontsize=10, color=INK_SECONDARY)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(
        "Which formal boundaries the curve shape can represent",
        fontsize=12, color=INK, loc="left", pad=10,
    )
    fig.text(
        0.02, -0.02,
        "Fraction of artifacts with that true boundary for which some additive "
        "translation reaches it. j* = 9 is always reachable by construction.",
        fontsize=8, color=INK_MUTED,
    )
    return _save(fig, root / FIGURE_DIR / "figure4_crossing_geometry.png")


# --------------------------------------------------------------------------
# Figure 5 -- oracle hierarchy
# --------------------------------------------------------------------------


def figure5_oracle_hierarchy(root: Path, metrics: dict, samples: dict) -> Path:
    present = _present(metrics)
    stages = [
        ("tce_raw", "RAW"),
        ("tce_oracle_global", "ORACLE-G"),
        ("tce_oracle_family", "ORACLE-F"),
        ("otce", "ORACLE-A"),
    ]

    fig = _new_figure(9.4, 5.0)
    ax = fig.add_subplot(111)
    _style_axes(ax)

    x = np.arange(len(stages))
    end_labels: list[list] = []
    for model_id in present:
        frame = metrics[model_id]
        colour = _color_for(model_id)
        points = [
            float(frame["tce_raw"].mean()),
            float(frame["tce_oracle_global"].mean()),
            float(frame["tce_oracle_family"].mean()),
            float(frame["otce"].mean()),
        ]
        lows, highs = [], []
        for key, _ in stages:
            draws = samples[model_id][key]
            finite = draws[np.isfinite(draws)]
            low, high = np.quantile(finite, [0.025, 0.975])
            lows.append(low)
            highs.append(high)
        ax.fill_between(x, lows, highs, color=colour, alpha=0.14, linewidth=0)
        ax.plot(x, points, color=colour, linewidth=2.2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.4)
        end_labels.append([points[-1], f"{MODEL_SHORT_NAMES[model_id]}  {points[-1]:.2f}", colour])

    # Dodge the direct labels apart; the oracle-artifact bounds can coincide.
    top = max(float(metrics[m]["tce_raw"].mean()) for m in present)
    end_labels.sort(key=lambda item: item[0])
    min_gap = 0.055 * top
    for i in range(1, len(end_labels)):
        if end_labels[i][0] - end_labels[i - 1][0] < min_gap:
            end_labels[i][0] = end_labels[i - 1][0] + min_gap
    for y_label, text, colour in end_labels:
        ax.annotate(
            text,
            xy=(x[-1] + 0.08, y_label),
            fontsize=9,
            color=colour,
            va="center",
            ha="left",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in stages], fontsize=10, color=INK_SECONDARY)
    ax.set_ylabel("mean threshold crossing error", fontsize=10, color=INK_SECONDARY)
    ax.set_xlim(-0.2, len(stages) - 0.05)
    ax.set_ylim(bottom=0)
    ax.set_title(
        "Nested oracle bounds: what location-only explanations can achieve",
        fontsize=12, color=INK, loc="left", pad=10,
    )
    fig.text(
        0.02, -0.02,
        "Left to right, translation gains freedom: none, one shift per judge, one "
        "per judge and family, one per artifact. All three oracles use formal "
        "truth and are diagnostic bounds, not deployment results. Bands are 95% "
        "bootstrap intervals with the group oracles refitted in each replicate.",
        fontsize=8, color=INK_MUTED,
    )
    return _save(fig, root / FIGURE_DIR / "figure5_oracle_hierarchy.png")


# --------------------------------------------------------------------------
# Figure 6 -- shape floor by family
# --------------------------------------------------------------------------


def figure6_shape_floor(root: Path, metrics: dict) -> Path:
    present = _present(metrics)
    panels = [
        ("otce", "mean OTCE", "lower is better", "{:.2f}"),
        ("translation_reachable", "nontrivial TRR", "higher is better", "{:.0%}"),
        ("reachable_count", "reachable count", "higher is richer", "{:.2f}"),
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
                subset = frame[frame["family"] == family]
                if column == "translation_reachable":
                    subset = subset[subset["nontrivial_boundary"]]
                grid[row, col] = subset[column].mean()

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
        "Translation floor and curve geometry by judge and predicate family",
        fontsize=12, color=INK, x=0.02, ha="left", y=1.03,
    )
    fig.text(
        0.02, -0.03,
        "Shading is normalised within each panel; every cell carries its exact "
        "value. MMVR and the reachable count are properties of the raw margins "
        "and are invariant to any translation.",
        fontsize=8, color=INK_MUTED,
    )
    return _save(fig, root / FIGURE_DIR / "figure6_shape_floor.png")


# --------------------------------------------------------------------------


def render_all(
    root: Path,
    metrics: dict,
    stacks: dict,
    oracle_global: dict,
    samples: dict,
) -> list[Path]:
    """Render every predeclared Day-3 figure and return the paths written."""
    paths = [
        figure1_reachability_concept(root),
        figure2_raw_vs_otce(root, metrics),
        figure3_reachability(root, metrics),
        figure4_crossing_geometry(root, metrics),
        figure5_oracle_hierarchy(root, metrics, samples),
        figure6_shape_floor(root, metrics),
    ]
    for path in paths:
        print(f"wrote {path.relative_to(root)}", flush=True)
    return paths
