"""Reasoning-anchor figures. Four, each answering the anchor question directly.

1. TCE: the anchor against the original four judges and the structural references.
2. Nontrivial unreachable fraction: the anchor against the original four.
3. Distribution of reasoning-trace length.
4. TCE and reachability by reasoning length -- descriptive, and labelled so.

Generated from the saved machine-readable analysis outputs, never from numbers
retyped by hand. Colour follows the same categorical palette as Days 1-3 so a
judge keeps its identity across every FormalCRRC figure; the anchor is drawn in
a distinct colour and the structural references in grey, because they are not
comparable objects and the figure should not suggest otherwise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from formalcrrc import provenance  # noqa: E402
from formalcrrc import reasoning_anchor_config as rc  # noqa: E402

ROOT = provenance.repo_root()

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#86857f"
GRID = "#e3e2de"
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
ANCHOR_COLOR = "#7b3fb5"
REFERENCE_COLOR = "#b9b8b3"


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)


def _figure(width: float = 7.2, height: float = 4.2):
    fig, ax = plt.subplots(figsize=(width, height), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    return fig, ax


def _save(fig, name: str) -> Path:
    out_dir = ROOT / rc.FIGURE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")
    return path


def figure_tce(summary: dict) -> None:
    """Where the anchor's localisation error sits among everything measured."""
    anchor_tce = summary["primary"]["mean_tce"]
    ci = summary["primary"]["mean_tce_ci"]

    names = list(rc.DAY3_TCE)
    values = [rc.DAY3_TCE[n] for n in names]
    colors = list(SERIES_COLORS[: len(names)])

    names.append(rc.MODEL_SHORT_NAME)
    values.append(anchor_tce)
    colors.append(ANCHOR_COLOR)

    fig, ax = _figure()
    positions = np.arange(len(names))
    ax.bar(positions, values, color=colors, width=0.62, zorder=3)
    ax.errorbar(
        [positions[-1]], [anchor_tce],
        yerr=[[anchor_tce - ci[0]], [ci[1] - anchor_tce]],
        fmt="none", ecolor=INK, elinewidth=1.4, capsize=5, zorder=4,
    )
    for label, y, style in (
        ("fixed centre 2.222", rc.BASELINE_FIXED_CENTER_TCE, "--"),
        ("random crossing 2.963", rc.BASELINE_RANDOM_CROSSING_TCE, ":"),
    ):
        ax.axhline(y, color=INK_MUTED, linewidth=1.1, linestyle=style, zorder=2)
        ax.text(len(names) - 0.4, y, label, fontsize=8, color=INK_MUTED,
                va="bottom", ha="right")

    ax.set_xticks(positions)
    ax.set_xticklabels(names, fontsize=9, rotation=12, ha="right")
    ax.set_ylabel("Mean threshold crossing error", fontsize=10, color=INK)
    ax.set_title(
        "Threshold localisation: reasoning anchor vs original panel",
        fontsize=11, color=INK, loc="left",
    )
    ax.text(
        0, -0.30,
        "Original judges scored at the immediate next token; the anchor uses "
        "reason-then-score.\nThe comparison is descriptive, not a controlled "
        "contrast.",
        transform=ax.transAxes, fontsize=8, color=INK_MUTED, va="top",
    )
    _save(fig, "ranchor_tce.png")


def figure_unreachable(summary: dict) -> None:
    """The principal ordering endpoint, side by side with the frozen panel."""
    anchor = summary["primary"]["unreachable_pct"]
    trr_ci = summary["primary"]["trr_ci"]
    ci = [100.0 * (1.0 - trr_ci[1]), 100.0 * (1.0 - trr_ci[0])]

    names = list(rc.DAY3_UNREACHABLE_PCT)
    values = [rc.DAY3_UNREACHABLE_PCT[n] for n in names]
    colors = list(SERIES_COLORS[: len(names)])
    names.append(rc.MODEL_SHORT_NAME)
    values.append(anchor)
    colors.append(ANCHOR_COLOR)

    fig, ax = _figure()
    positions = np.arange(len(names))
    ax.bar(positions, values, color=colors, width=0.62, zorder=3)
    ax.errorbar(
        [positions[-1]], [anchor],
        yerr=[[max(anchor - ci[0], 0)], [max(ci[1] - anchor, 0)]],
        fmt="none", ecolor=INK, elinewidth=1.4, capsize=5, zorder=4,
    )
    low, high = rc.DAY3_UNREACHABLE_RANGE
    ax.axhspan(low, high, color=REFERENCE_COLOR, alpha=0.28, zorder=1)
    ax.text(
        0.02, high + 1.5, f"original panel range {low:.2f}-{high:.2f}%",
        fontsize=8, color=INK_MUTED,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(names, fontsize=9, rotation=12, ha="right")
    ax.set_ylabel("Nontrivial boundaries unreachable (%)", fontsize=10, color=INK)
    ax.set_title(
        "Boundary reachability: fraction of formal boundaries absent from the "
        "score ordering",
        fontsize=11, color=INK, loc="left",
    )
    _save(fig, "ranchor_unreachable.png")


def figure_reasoning_length(rows: pd.DataFrame) -> None:
    """How long the anchor actually thought, and how often it ran out."""
    lengths = rows["reasoning_token_count"].to_numpy(dtype=float)
    fig, ax = _figure()
    ax.hist(lengths, bins=48, color=ANCHOR_COLOR, alpha=0.85, zorder=3)
    median = float(np.median(lengths))
    ax.axvline(median, color=INK, linewidth=1.3, linestyle="--", zorder=4)
    ax.text(median, ax.get_ylim()[1] * 0.94, f"  median {median:.0f}",
            fontsize=9, color=INK, va="top")
    truncated = int(rows["reasoning_truncated"].sum())
    ax.set_xlabel("Reasoning tokens generated", fontsize=10, color=INK)
    ax.set_ylabel("Rows", fontsize=10, color=INK)
    ax.set_title(
        f"Reasoning trace length (budget {rc.REASONING_TOKEN_BUDGET}; "
        f"{truncated} truncated)",
        fontsize=11, color=INK, loc="left",
    )
    _save(fig, "ranchor_reasoning_length.png")


def figure_length_vs_outcome(curves: pd.DataFrame) -> None:
    """Descriptive only: length is chosen by the model, not assigned."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        _style(ax)

    x = curves["mean_reasoning_tokens"].to_numpy(dtype=float)
    axes[0].scatter(x, curves["tce_raw"], s=16, alpha=0.55, color=ANCHOR_COLOR,
                    edgecolors="none", zorder=3)
    axes[0].set_xlabel("Mean reasoning tokens per curve", fontsize=9, color=INK)
    axes[0].set_ylabel("Curve TCE", fontsize=9, color=INK)
    axes[0].set_title("Localisation vs trace length", fontsize=10, color=INK, loc="left")

    nontrivial = curves[curves["nontrivial_boundary"]]
    groups = [
        nontrivial[~nontrivial["translation_reachable"]]["mean_reasoning_tokens"],
        nontrivial[nontrivial["translation_reachable"]]["mean_reasoning_tokens"],
    ]
    axes[1].boxplot(
        [g.to_numpy(dtype=float) for g in groups],
        tick_labels=["unreachable", "reachable"], patch_artist=True,
        boxprops=dict(facecolor="#e7dcf3", color=INK_SECONDARY),
        medianprops=dict(color=ANCHOR_COLOR, linewidth=1.6),
        whiskerprops=dict(color=INK_SECONDARY),
        capprops=dict(color=INK_SECONDARY),
        flierprops=dict(markersize=3, markerfacecolor=INK_MUTED, markeredgecolor="none"),
    )
    axes[1].set_ylabel("Mean reasoning tokens per curve", fontsize=9, color=INK)
    axes[1].set_title("Trace length by boundary reachability", fontsize=10,
                      color=INK, loc="left")

    fig.suptitle(
        "Descriptive associations only -- trace length is chosen by the model "
        "and confounded with item difficulty",
        fontsize=8.5, color=INK_MUTED, y=0.02,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out_dir = ROOT / rc.FIGURE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "ranchor_length_vs_outcome.png"
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def main() -> int:
    summary = json.loads((ROOT / rc.SUMMARY_PATH).read_text(encoding="utf-8"))
    rows = pd.read_parquet(ROOT / rc.RAW_ROWS_PATH)
    curves = pd.read_parquet(ROOT / rc.METRICS_PATH)

    figure_tce(summary)
    figure_unreachable(summary)
    figure_reasoning_length(rows)
    figure_length_vs_outcome(curves)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
