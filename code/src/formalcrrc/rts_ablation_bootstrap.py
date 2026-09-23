"""Paired artifact-clustered bootstrap for the within-model protocol ablation.

The two conditions are measured on the *same* artifacts, so the comparison is
paired and the resampling must respect that: a replicate draws an artifact once
and takes both its immediate and its reason-then-score curve together. Drawing
the conditions independently would discard the pairing that makes this a
within-model ablation and would inflate the interval on every difference.

The unit is the artifact curve, never the threshold row. All nine strictness
levels move together because they are nine measurements of one item.

One deliberate refusal: when a rate is identically zero across every curve, the
bootstrap returns ``[0, 0]``. That describes the resampling distribution, not
the population, and reporting it as uncertainty would overstate what the data
support. :func:`formalcrrc.rts_ablation.clopper_pearson` supplies the exact
interval for that case, and the analysis reports it instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from formalcrrc.config import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, FAMILIES


def percentile_ci(samples: np.ndarray, level: float = 95.0) -> tuple[float, float]:
    """Two-sided percentile interval over the finite replicates."""
    finite = np.asarray(samples, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (float("nan"), float("nan"))
    tail = (100.0 - level) / 2.0
    return (
        float(np.percentile(finite, tail)),
        float(np.percentile(finite, 100.0 - tail)),
    )


def summarise(
    samples: np.ndarray, point: float, level: float = 95.0, degenerate_note: str | None = None
) -> dict[str, Any]:
    low, high = percentile_ci(samples, level)
    out: dict[str, Any] = {
        "point": float(point),
        "ci_low": low,
        "ci_high": high,
        "ci_level": level,
        "n_replicates": int(np.sum(np.isfinite(np.asarray(samples, dtype=float)))),
    }
    if low == high:
        out["degenerate"] = True
        out["note"] = degenerate_note or (
            "the bootstrap interval collapsed to a point; this reflects the "
            "resampling distribution of a constant statistic, not population "
            "uncertainty"
        )
    return out


@dataclass(frozen=True)
class PairedPlan:
    """A family-stratified resampling of artifacts, shared by both conditions."""

    artifact_ids: tuple[str, ...]
    families: tuple[str, ...]
    indices: np.ndarray
    replicates: int
    seed: int


def build_plan(
    artifact_ids: Sequence[str],
    families: Sequence[str],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> PairedPlan:
    """Draw the index matrix once so every metric and both conditions share it.

    Sharing one plan is what makes the paired differences coherent: the same
    replicate universe underlies the immediate curve, the reason-then-score
    curve and their difference.
    """
    if len(artifact_ids) != len(families):
        raise ValueError("artifact_ids and families must be the same length")
    family_of = np.asarray(families)
    rng = np.random.default_rng(seed)
    columns: list[np.ndarray] = []
    for family in FAMILIES:
        positions = np.flatnonzero(family_of == family)
        if positions.size == 0:
            continue
        drawn = rng.integers(0, positions.size, size=(replicates, positions.size))
        columns.append(positions[drawn])
    indices = (
        np.concatenate(columns, axis=1) if columns else np.empty((replicates, 0), int)
    )
    return PairedPlan(
        artifact_ids=tuple(artifact_ids),
        families=tuple(families),
        indices=indices,
        replicates=replicates,
        seed=seed,
    )


def replicate_means(
    values: np.ndarray, plan: PairedPlan, mask: np.ndarray | None = None
) -> np.ndarray:
    """Mean of ``values`` under every replicate, restricted to ``mask``."""
    values = np.asarray(values, dtype=float)
    drawn = values[plan.indices]
    if mask is None:
        return drawn.mean(axis=1)
    eligible = np.asarray(mask, dtype=bool)[plan.indices]
    counts = eligible.sum(axis=1)
    totals = np.where(eligible, drawn, 0.0).sum(axis=1)
    out = np.full(plan.replicates, np.nan, dtype=float)
    nonzero = counts > 0
    out[nonzero] = totals[nonzero] / counts[nonzero]
    return out


def bootstrap_ablation(
    curves: pd.DataFrame,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Every preregistered interval, paired, from one shared plan.

    ``curves`` is one row per complete artifact curve carrying both conditions:
    ``tce_o``/``tce_r``, ``reachable_o``/``reachable_r``, ``accuracy_o``/
    ``accuracy_r``, ``brier_o``/``brier_r``, ``rc_o``/``rc_r``,
    ``nontrivial_boundary`` and ``family``.
    """
    frame = curves.reset_index(drop=True)
    plan = build_plan(
        frame["artifact_id"].tolist(),
        frame["family"].tolist(),
        replicates=replicates,
        seed=seed,
    )
    nontrivial = frame["nontrivial_boundary"].to_numpy(dtype=bool)

    def col(name: str) -> np.ndarray:
        return frame[name].to_numpy(dtype=float)

    tce_o, tce_r = col("tce_o"), col("tce_r")
    reach_o, reach_r = col("reachable_o"), col("reachable_r")
    acc_o, acc_r = col("accuracy_o"), col("accuracy_r")
    brier_o, brier_r = col("brier_o"), col("brier_r")
    rc_o, rc_r = col("rc_o"), col("rc_r")

    trr_o_s = replicate_means(reach_o, plan, mask=nontrivial)
    trr_r_s = replicate_means(reach_r, plan, mask=nontrivial)
    delta_trr_s = trr_r_s - trr_o_s
    delta_tce_s = replicate_means(tce_r - tce_o, plan)

    results: dict[str, Any] = {
        "bootstrap": {
            "replicates": replicates,
            "seed": seed,
            "unit": "artifact_curve",
            "pairing": "both conditions of one artifact drawn together",
            "stratified_by": "family",
            "ci": "percentile, 95%",
        },
        "immediate": {
            "mean_tce": summarise(replicate_means(tce_o, plan), float(tce_o.mean())),
            "trr_nontrivial": summarise(
                trr_o_s, float(reach_o[nontrivial].mean())
            ),
            "unreachable_fraction": summarise(
                1.0 - trr_o_s, float(1.0 - reach_o[nontrivial].mean())
            ),
            "accuracy": summarise(replicate_means(acc_o, plan), float(acc_o.mean())),
            "brier": summarise(replicate_means(brier_o, plan), float(brier_o.mean())),
            "mean_reachable_count": summarise(
                replicate_means(rc_o, plan), float(rc_o.mean())
            ),
        },
        "reason_then_score": {
            "mean_tce": summarise(replicate_means(tce_r, plan), float(tce_r.mean())),
            "trr_nontrivial": summarise(
                trr_r_s, float(reach_r[nontrivial].mean())
            ),
            "unreachable_fraction": summarise(
                1.0 - trr_r_s,
                float(1.0 - reach_r[nontrivial].mean()),
                degenerate_note=(
                    "every nontrivial boundary was reachable in every replicate; "
                    "use the exact Clopper-Pearson interval for the zero count "
                    "rather than this bootstrap"
                ),
            ),
            "accuracy": summarise(replicate_means(acc_r, plan), float(acc_r.mean())),
            "brier": summarise(replicate_means(brier_r, plan), float(brier_r.mean())),
            "mean_reachable_count": summarise(
                replicate_means(rc_r, plan), float(rc_r.mean())
            ),
        },
        "paired": {
            "delta_trr": summarise(
                delta_trr_s,
                float(reach_r[nontrivial].mean() - reach_o[nontrivial].mean()),
            ),
            "delta_tce": summarise(delta_tce_s, float((tce_r - tce_o).mean())),
            "delta_accuracy": summarise(
                replicate_means(acc_r - acc_o, plan), float((acc_r - acc_o).mean())
            ),
            "delta_brier": summarise(
                replicate_means(brier_r - brier_o, plan),
                float((brier_r - brier_o).mean()),
            ),
            "delta_reachable_count": summarise(
                replicate_means(rc_r - rc_o, plan), float((rc_r - rc_o).mean())
            ),
        },
        "by_family": {},
    }

    for family in FAMILIES:
        in_family = (frame["family"] == family).to_numpy(dtype=bool)
        if not in_family.any():
            continue
        fam_nontrivial = in_family & nontrivial
        f_trr_o = replicate_means(reach_o, plan, mask=fam_nontrivial)
        f_trr_r = replicate_means(reach_r, plan, mask=fam_nontrivial)
        results["by_family"][family] = {
            "n_curves": int(in_family.sum()),
            "n_nontrivial": int(fam_nontrivial.sum()),
            "tce_o": summarise(
                replicate_means(tce_o, plan, mask=in_family),
                float(tce_o[in_family].mean()),
            ),
            "tce_r": summarise(
                replicate_means(tce_r, plan, mask=in_family),
                float(tce_r[in_family].mean()),
            ),
            "delta_tce": summarise(
                replicate_means(tce_r - tce_o, plan, mask=in_family),
                float((tce_r - tce_o)[in_family].mean()),
            ),
            "trr_o": summarise(f_trr_o, float(reach_o[fam_nontrivial].mean())),
            "trr_r": summarise(f_trr_r, float(reach_r[fam_nontrivial].mean())),
            "delta_trr": summarise(
                f_trr_r - f_trr_o,
                float(
                    reach_r[fam_nontrivial].mean() - reach_o[fam_nontrivial].mean()
                ),
            ),
        }
    return results


def to_frame(results: dict[str, Any]) -> pd.DataFrame:
    """Flatten the nested interval record into one tidy table."""
    rows: list[dict[str, Any]] = []
    for scope in ("immediate", "reason_then_score", "paired"):
        for metric, block in results[scope].items():
            rows.append(
                {"scope": scope, "family": None, "metric": metric, **block}
            )
    for family, metrics in results["by_family"].items():
        for metric, block in metrics.items():
            if isinstance(block, dict):
                rows.append(
                    {"scope": "family", "family": family, "metric": metric, **block}
                )
    return pd.DataFrame(rows)
