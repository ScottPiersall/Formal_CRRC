"""Reason-then-score ablation figures. Four, each answering the ablation question.

1. Within-model reachability: the same Qwen2.5 under both protocols, with the
   Qwen3 anchor shown separately and greyed, because it is context and not a
   comparable arm.
2. Paired per-curve TCE change, which shows the distribution behind the mean.
3. Unreachable-failure mechanism: strict inversion versus tie-only, per protocol.
4. Generated-trace length against outcome -- descriptive, and labelled so.

Generated from the saved machine-readable outputs, never from numbers retyped by
hand. The two Qwen2.5 conditions share a colour family because they are the same
model; the anchor is visually separated so no reader mistakes it for a third arm
of the ablation.
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
from formalcrrc import rts_ablation_config as rc  # noqa: E402

ROOT = provenance.repo_root()

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#86857f"
GRID = "#e3e2de"

IMMEDIATE = "#c8d9ef"
REASONING = "#2a78d6"
ANCHOR = "#b9b8b3"
INVERSION = "#eb6834"
TIE = "#eda100"


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)


def _figure(width: float = 7.4, height: float = 4.3):
    fig, ax = plt.subplots(figsize=(width, height), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    return fig, ax


def _save(fig, name: str) -> None:
    out_dir = ROOT / rc.FIGURE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def figure_reachability(summary: dict) -> None:
    """The primary result: same model, two protocols."""
    primary = summary["primary"]
    u_o = primary["unreachable_pct_immediate"]
    u_r = primary["unreachable_pct_reason_then_score"]

    fig, ax = _figure()
    labels = [
        f"{rc.MODEL_SHORT_NAME}\nimmediate\n(frozen Day 3)",
        f"{rc.MODEL_SHORT_NAME}\nreason-then-score\n(this study)",
        "Qwen3-30B-Thinking\nnative reasoning\n(context only)",
    ]
    values = [u_o, u_r, rc.ANCHOR_UNREACHABLE_PCT]
    colors = [IMMEDIATE, REASONING, ANCHOR]
    positions = np.arange(3)
    bars = ax.bar(positions, values, color=colors, width=0.6, zorder=3)
    bars[2].set_hatch("//")
    bars[2].set_edgecolor("#9a9995")

    for x, value in zip(positions, values):
        ax.text(x, value + 1.2, f"{value:.2f}%", ha="center", fontsize=9.5, color=INK)

    ax.axvline(1.5, color=GRID, linewidth=1.4, zorder=2)
    ax.text(
        0.5, max(values) * 0.92, "within-model ablation",
        ha="center", fontsize=8.5, color=INK_SECONDARY,
    )
    ax.text(
        2.0, max(values) * 0.92, "external context",
        ha="center", fontsize=8.5, color=INK_MUTED,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Nontrivial boundaries unreachable (%)", fontsize=10, color=INK)
    ax.set_ylim(0, max(max(values) * 1.25, 10))
    ax.set_title(
        "Boundary reachability under two readout protocols for one model",
        fontsize=11, color=INK, loc="left",
    )
    _save(fig, "rts_reachability.png")


def figure_paired_tce(curves: pd.DataFrame) -> None:
    """Per-curve change, so the mean is not doing the talking alone."""
    delta = curves["delta_tce"].to_numpy(dtype=float)
    fig, ax = _figure()
    bins = np.arange(delta.min() - 0.5, delta.max() + 1.5, 1.0)
    ax.hist(delta, bins=bins, color=REASONING, alpha=0.85, zorder=3)
    ax.axvline(0, color=INK, linewidth=1.2, linestyle="--", zorder=4)
    mean = float(delta.mean())
    ax.axvline(mean, color=INVERSION, linewidth=1.6, zorder=5)
    ax.text(mean, ax.get_ylim()[1] * 0.93, f"  mean {mean:+.2f}",
            fontsize=9, color=INVERSION, va="top")
    improved = int((delta < 0).sum())
    worse = int((delta > 0).sum())
    same = int((delta == 0).sum())
    ax.set_xlabel(
        "Change in curve TCE  (reason-then-score − immediate)", fontsize=10, color=INK
    )
    ax.set_ylabel("Artifact curves", fontsize=10, color=INK)
    ax.set_title(
        f"Paired localisation change: {improved} improved, {same} unchanged, "
        f"{worse} worse",
        fontsize=11, color=INK, loc="left",
    )
    _save(fig, "rts_paired_tce.png")


def figure_failure_mechanism(decomp: pd.DataFrame) -> None:
    """Why boundaries were unreachable: real inversions or bare ties."""
    fig, ax = _figure()
    labels, inversions, ties = [], [], []
    for row in decomp.itertuples(index=False):
        labels.append(
            "immediate\n(frozen Day 3)"
            if row.condition == rc.CONDITION_ORIGINAL
            else "reason-then-score\n(this study)"
        )
        inversions.append(row.n_strict_inversion)
        ties.append(row.n_tie_only)

    positions = np.arange(len(labels))
    ax.bar(positions, inversions, width=0.55, color=INVERSION, zorder=3,
           label="strict inversion")
    ax.bar(positions, ties, width=0.55, bottom=inversions, color=TIE, zorder=3,
           label="tie only")
    for x, (inv, tie) in enumerate(zip(inversions, ties)):
        total = inv + tie
        if total:
            ax.text(x, total + 1.5, f"{total} unreachable", ha="center",
                    fontsize=9, color=INK)
        else:
            ax.text(x, 1.5, "0 unreachable", ha="center", fontsize=9, color=INK)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Nontrivial curves", fontsize=10, color=INK)
    ax.set_title(
        "Unreachable-boundary mechanism, same model under both protocols",
        fontsize=11, color=INK, loc="left",
    )
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    _save(fig, "rts_failure_mechanism.png")


def figure_length(curves: pd.DataFrame) -> None:
    """Descriptive: trace length is chosen by the model, not assigned."""
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        _style(ax)

    lengths = curves["mean_generated_tokens"].to_numpy(dtype=float)
    axes[0].hist(lengths, bins=40, color=REASONING, alpha=0.85, zorder=3)
    median = float(np.median(lengths))
    axes[0].axvline(median, color=INK, linewidth=1.3, linestyle="--", zorder=4)
    axes[0].text(median, axes[0].get_ylim()[1] * 0.94, f"  median {median:.0f}",
                 fontsize=9, color=INK, va="top")
    axes[0].set_xlabel("Mean generated tokens per curve", fontsize=9, color=INK)
    axes[0].set_ylabel("Artifact curves", fontsize=9, color=INK)
    axes[0].set_title("Elicited trace length", fontsize=10, color=INK, loc="left")

    axes[1].scatter(lengths, curves["tce_r"], s=16, alpha=0.55, color=REASONING,
                    edgecolors="none", zorder=3)
    axes[1].set_xlabel("Mean generated tokens per curve", fontsize=9, color=INK)
    axes[1].set_ylabel("Curve TCE (reason-then-score)", fontsize=9, color=INK)
    axes[1].set_title("Localisation vs trace length", fontsize=10, color=INK,
                      loc="left")

    fig.suptitle(
        "Descriptive associations only -- trace length is chosen by the model "
        "and confounded with item difficulty",
        fontsize=8.5, color=INK_MUTED, y=0.02,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out_dir = ROOT / rc.FIGURE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "rts_length.png"
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def main() -> int:
    summary = json.loads((ROOT / rc.SUMMARY_PATH).read_text(encoding="utf-8"))
    curves = pd.read_parquet(ROOT / rc.METRICS_PATH)
    decomp = pd.read_parquet(ROOT / rc.FAILURE_DECOMP_PATH)

    figure_reachability(summary)
    figure_paired_tce(curves)
    figure_failure_mechanism(decomp)
    figure_length(curves)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
