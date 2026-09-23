"""Label-swap figures. Three, each answering the robustness question directly.

1. Original vs swapped TCE by judge, with the paired change and its interval.
2. Original vs swapped nontrivial unreachable fraction (1 - TRR).
3. Distribution of per-curve original-vs-swapped Spearman rho.

Generated from the saved machine-readable analysis outputs, never from numbers
retyped by hand. Colour follows the same categorical palette as Days 1-3 so a
judge keeps its identity across every FormalCRRC figure.
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
from formalcrrc import labelswap_config as lsc  # noqa: E402
from formalcrrc.config import MODEL_IDS, MODEL_SHORT_NAMES  # noqa: E402

ROOT = provenance.repo_root()

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#86857f"
GRID = "#e3e2de"
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")

ORIGINAL_FILL = "#c8d9ef"
SWAPPED_FILL = "#2a78d6"


def _color_for(model_id: str) -> str:
    return SERIES_COLORS[MODEL_IDS.index(model_id) % len(SERIES_COLORS)]


def _style(ax, *, x_grid: bool = False) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.grid(axis="x" if x_grid else "y", color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def _save(fig, name: str) -> str:
    out_dir = ROOT / lsc.FIGURE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return str(path.relative_to(ROOT))


def figure_tce(summary: dict) -> str:
    """Paired original vs swapped TCE, with the delta and its 95% interval."""
    models = list(MODEL_IDS)
    labels = [MODEL_SHORT_NAMES[m] for m in models]
    original = [summary["by_model"][m]["point_estimates"]["tce_original"] for m in models]
    swapped = [summary["by_model"][m]["point_estimates"]["tce_swapped"] for m in models]
    deltas = [summary["by_model"][m]["point_estimates"]["delta_tce"] for m in models]
    lows = [
        summary["by_model"][m]["bootstrap"]["statistics"]["delta_tce"]["ci_low"]
        for m in models
    ]
    highs = [
        summary["by_model"][m]["bootstrap"]["statistics"]["delta_tce"]["ci_high"]
        for m in models
    ]

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.25, 1]}
    )
    fig.patch.set_facecolor(SURFACE)

    y = np.arange(len(models))
    height = 0.36
    left.barh(y + height / 2, original, height, color=ORIGINAL_FILL,
              edgecolor=SURFACE, linewidth=2, label="original (A = met)")
    left.barh(y - height / 2, swapped, height, color=SWAPPED_FILL,
              edgecolor=SURFACE, linewidth=2, label="swapped (B = met)")
    for i, (o, s) in enumerate(zip(original, swapped)):
        left.text(o + 0.06, i + height / 2, f"{o:.3f}", va="center",
                  fontsize=8.5, color=INK_SECONDARY)
        left.text(s + 0.06, i - height / 2, f"{s:.3f}", va="center",
                  fontsize=8.5, color=INK)
    left.set_yticks(y, labels)
    left.set_xlabel("mean TCE  (lower is better)", fontsize=9.5, color=INK_SECONDARY)
    left.set_title("Threshold crossing error", fontsize=11, color=INK, loc="left")
    left.legend(frameon=False, fontsize=8.5, loc="lower right")
    left.set_xlim(0, max(original + swapped) * 1.2)
    _style(left, x_grid=True)

    band = lsc.DELTA_TCE_BAND
    right.axvspan(band[0], band[1], color="#eaf2fc", zorder=0)
    right.axvline(0, color=INK_MUTED, linewidth=1, zorder=1)
    for i, model_id in enumerate(models):
        right.plot([lows[i], highs[i]], [i, i], color=_color_for(model_id),
                   linewidth=2.4, solid_capstyle="round", zorder=3)
        right.plot([deltas[i]], [i], "o", color=_color_for(model_id),
                   markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
        right.text(highs[i] + 0.02, i, f"[{lows[i]:+.3f}, {highs[i]:+.3f}]",
                   va="center", fontsize=8, color=INK_SECONDARY)
    right.set_yticks(y, labels)
    right.set_xlabel("Δ TCE (swapped − original), 95% CI", fontsize=9.5,
                     color=INK_SECONDARY)
    right.set_title(f"Paired change · band [{band[0]}, {band[1]}]",
                    fontsize=11, color=INK, loc="left")
    _style(right, x_grid=True)

    fig.suptitle("Label swap: threshold localization", fontsize=13, color=INK,
                 x=0.02, ha="left", y=1.02)
    return _save(fig, "labelswap_tce.png")


def figure_reachability(summary: dict) -> str:
    """Original vs swapped nontrivial unreachable fraction, plus Δ TRR."""
    models = list(MODEL_IDS)
    labels = [MODEL_SHORT_NAMES[m] for m in models]
    unreachable_o = [
        100 * summary["by_model"][m]["point_estimates"]["unreachable_original"]
        for m in models
    ]
    unreachable_s = [
        100 * summary["by_model"][m]["point_estimates"]["unreachable_swapped"]
        for m in models
    ]
    deltas = [summary["by_model"][m]["point_estimates"]["delta_trr"] for m in models]
    lows = [
        summary["by_model"][m]["bootstrap"]["statistics"]["delta_trr"]["ci_low"]
        for m in models
    ]
    highs = [
        summary["by_model"][m]["bootstrap"]["statistics"]["delta_trr"]["ci_high"]
        for m in models
    ]

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.25, 1]}
    )
    fig.patch.set_facecolor(SURFACE)

    y = np.arange(len(models))
    height = 0.36
    left.barh(y + height / 2, unreachable_o, height, color=ORIGINAL_FILL,
              edgecolor=SURFACE, linewidth=2, label="original (A = met)")
    left.barh(y - height / 2, unreachable_s, height, color=SWAPPED_FILL,
              edgecolor=SURFACE, linewidth=2, label="swapped (B = met)")
    for i, (o, s) in enumerate(zip(unreachable_o, unreachable_s)):
        left.text(o + 1.2, i + height / 2, f"{o:.1f}%", va="center",
                  fontsize=8.5, color=INK_SECONDARY)
        left.text(s + 1.2, i - height / 2, f"{s:.1f}%", va="center",
                  fontsize=8.5, color=INK)
    left.set_yticks(y, labels)
    left.set_xlabel("nontrivial boundaries unreachable by translation (%)",
                    fontsize=9.5, color=INK_SECONDARY)
    left.set_title("Ordering failure persists", fontsize=11, color=INK, loc="left")
    left.legend(frameon=False, fontsize=8.5, loc="lower right")
    left.set_xlim(0, 100)
    _style(left, x_grid=True)

    band = lsc.DELTA_TRR_BAND
    right.axvspan(band[0], band[1], color="#eaf2fc", zorder=0)
    right.axvline(0, color=INK_MUTED, linewidth=1, zorder=1)
    for i, model_id in enumerate(models):
        right.plot([lows[i], highs[i]], [i, i], color=_color_for(model_id),
                   linewidth=2.4, solid_capstyle="round", zorder=3)
        right.plot([deltas[i]], [i], "o", color=_color_for(model_id),
                   markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
        right.text(highs[i] + 0.004, i, f"[{lows[i]:+.3f}, {highs[i]:+.3f}]",
                   va="center", fontsize=8, color=INK_SECONDARY)
    right.set_yticks(y, labels)
    right.set_xlabel("Δ TRR (swapped − original), 95% CI", fontsize=9.5,
                     color=INK_SECONDARY)
    right.set_title(f"Paired change · band [{band[0]}, {band[1]}]",
                    fontsize=11, color=INK, loc="left")
    _style(right, x_grid=True)

    fig.suptitle("Label swap: boundary reachability", fontsize=13, color=INK,
                 x=0.02, ha="left", y=1.02)
    return _save(fig, "labelswap_reachability.png")


def figure_rank_correlation(metrics: pd.DataFrame) -> str:
    """Per-curve Spearman rho between original and swapped canonical margins."""
    models = list(MODEL_IDS)
    fig, axes = plt.subplots(1, len(models), figsize=(12.5, 3.4), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    bins = np.linspace(-1, 1, 41)

    for ax, model_id in zip(axes, models):
        values = metrics[metrics["model_id"] == model_id][
            "margin_rank_correlation"
        ].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        ax.hist(finite, bins=bins, color=_color_for(model_id), edgecolor=SURFACE,
                linewidth=0.5)
        median = float(np.median(finite))
        ax.axvline(median, color=INK, linewidth=1.4, linestyle="--")
        ax.text(0.03, 0.95, f"median {median:+.3f}", transform=ax.transAxes,
                fontsize=8.5, color=INK, va="top")
        ax.text(0.03, 0.85, f"n = {finite.size}", transform=ax.transAxes,
                fontsize=8, color=INK_MUTED, va="top")
        ax.set_title(MODEL_SHORT_NAMES[model_id], fontsize=10, color=INK, loc="left")
        ax.set_xlim(-1.05, 1.05)
        ax.set_xlabel("Spearman ρ", fontsize=9, color=INK_SECONDARY)
        _style(ax)
    axes[0].set_ylabel("curves", fontsize=9, color=INK_SECONDARY)

    fig.suptitle(
        "Per-curve rank correlation between original and swapped canonical margins",
        fontsize=12, color=INK, x=0.02, ha="left", y=1.06,
    )
    return _save(fig, "labelswap_rank_correlation.png")


def main() -> int:
    summary_path = ROOT / lsc.SUMMARY_PATH
    metrics_path = ROOT / lsc.METRICS_PATH
    if not (summary_path.is_file() and metrics_path.is_file()):
        print("run scripts/analyze_labelswap.py first")
        return 2

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = pd.read_parquet(metrics_path)

    written = [
        figure_tce(summary),
        figure_reachability(summary),
        figure_rank_correlation(metrics),
    ]
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
