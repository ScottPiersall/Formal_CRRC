"""Paired artifact-clustered bootstrap for the label-swap study.

Two things make this bootstrap *paired* without any extra machinery:

* the original and swapped conditions are indexed by the **same** artifact ids,
  so resampling an artifact draws both members of its pair together; and
* every statistic is reduced to a per-artifact value before resampling, so all
  nine strictness levels of a curve necessarily move as one unit.

The resampling plan itself is the tested Day-3 one
(:func:`formalcrrc.day3_bootstrap.build_plan`): family-stratified, artifact
clustered, ``BOOTSTRAP_REPLICATES`` replicates from ``BOOTSTRAP_SEED``. Nothing
here treats the 2,916 threshold rows as independent.

Nontrivial TRR is bootstrapped over its own 288-curve plan, as preregistered,
rather than as a subset mean of the 324-curve plan.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from formalcrrc.config import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED
from formalcrrc.day3_bootstrap import (
    ArtifactPlan,
    build_plan,
    multiplicity_matrix,
    replicate_means,
)

#: Sub-stream offset so the nontrivial plan is independent of the full plan
#: while staying deterministic from the recorded master seed.
NONTRIVIAL_SEED_OFFSET: int = 1


def percentile_ci(samples: np.ndarray, level: float = 95.0) -> tuple[float, float]:
    """Percentile interval. The preregistered interval type."""
    low = (100.0 - level) / 2.0
    return (
        float(np.percentile(samples, low)),
        float(np.percentile(samples, 100.0 - low)),
    )


def summarise(samples: np.ndarray, point: float, level: float = 95.0) -> dict:
    low, high = percentile_ci(samples, level)
    return {
        "point": float(point),
        "ci_low": low,
        "ci_high": high,
        "bootstrap_mean": float(np.mean(samples)),
        "bootstrap_sd": float(np.std(samples, ddof=1)),
        "replicates": int(samples.size),
    }


def build_plans(
    artifact_ids: Sequence[str],
    families: Sequence[str],
    nontrivial_mask: Sequence[bool],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[ArtifactPlan, ArtifactPlan]:
    """The full-curve plan and the nontrivial-curve plan.

    Both are family-stratified. The two are drawn from different seeds so the
    nontrivial analysis is not a deterministic function of the full one.
    """
    ids = list(artifact_ids)
    fams = list(families)
    mask = np.asarray(nontrivial_mask, dtype=bool)

    full = build_plan(ids, fams, replicates=replicates, seed=seed)
    nontrivial = build_plan(
        [i for i, keep in zip(ids, mask) if keep],
        [f for f, keep in zip(fams, mask) if keep],
        replicates=replicates,
        seed=seed + NONTRIVIAL_SEED_OFFSET,
    )
    return full, nontrivial


def bootstrap_statistic(
    values: Sequence[float], counts: np.ndarray, point: float | None = None
) -> dict:
    """Bootstrap the mean of one per-artifact quantity."""
    array = np.asarray(values, dtype=float)
    samples = replicate_means(array, counts)
    return summarise(samples, float(np.mean(array)) if point is None else point)


def bootstrap_judge(
    metrics: pd.DataFrame,
    plans: tuple[ArtifactPlan, ArtifactPlan] | None = None,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Every preregistered interval for one judge.

    ``metrics`` is one row per artifact carrying the paired per-curve
    quantities. Artifacts are assumed to be in a stable order.
    """
    metrics = metrics.sort_values("artifact_id").reset_index(drop=True)
    ids = metrics["artifact_id"].tolist()
    families = metrics["family"].tolist()
    nontrivial = metrics["nontrivial_boundary"].to_numpy(dtype=bool)

    if plans is None:
        plans = build_plans(
            ids, families, nontrivial, replicates=replicates, seed=seed
        )
    full_plan, nontrivial_plan = plans

    full_counts = multiplicity_matrix(full_plan)
    nontrivial_counts = multiplicity_matrix(nontrivial_plan)
    sub = metrics[nontrivial].reset_index(drop=True)

    results: dict[str, dict] = {}

    # ---- primary -----------------------------------------------------------
    results["agreement"] = bootstrap_statistic(
        metrics["semantic_agreement"], full_counts
    )
    results["tce_original"] = bootstrap_statistic(metrics["tce_original"], full_counts)
    results["tce_swapped"] = bootstrap_statistic(metrics["tce_swapped"], full_counts)
    results["delta_tce"] = bootstrap_statistic(metrics["delta_tce"], full_counts)

    trr_original = replicate_means(
        sub["tr_original"].to_numpy(dtype=float), nontrivial_counts
    )
    trr_swapped = replicate_means(
        sub["tr_swapped"].to_numpy(dtype=float), nontrivial_counts
    )
    results["trr_original"] = summarise(
        trr_original, float(sub["tr_original"].mean())
    )
    results["trr_swapped"] = summarise(trr_swapped, float(sub["tr_swapped"].mean()))
    results["delta_trr"] = summarise(
        trr_swapped - trr_original,
        float(sub["tr_swapped"].mean() - sub["tr_original"].mean()),
    )
    results["unreachable_original"] = summarise(
        1.0 - trr_original, float(1.0 - sub["tr_original"].mean())
    )
    results["unreachable_swapped"] = summarise(
        1.0 - trr_swapped, float(1.0 - sub["tr_swapped"].mean())
    )

    # ---- secondary ---------------------------------------------------------
    for column in (
        "accuracy_original",
        "accuracy_swapped",
        "delta_accuracy",
        "brier_original",
        "brier_swapped",
        "delta_brier",
        "rc_original",
        "rc_swapped",
        "delta_rc",
        "prr_original",
        "prr_swapped",
        "sign_agreement",
        "reachable_set_identical",
        "reachable_set_jaccard",
        "mmvr_original",
        "mmvr_swapped",
        "full_reachability_original",
        "full_reachability_swapped",
    ):
        if column in metrics.columns:
            results[column] = bootstrap_statistic(metrics[column], full_counts)

    # Spearman is undefined for flat curves; bootstrap the finite subset and
    # record how many curves were usable rather than silently dropping them.
    rho = metrics["margin_rank_correlation"].to_numpy(dtype=float)
    finite = np.isfinite(rho)
    if finite.any():
        finite_metrics = metrics[finite].reset_index(drop=True)
        finite_plan = build_plan(
            finite_metrics["artifact_id"].tolist(),
            finite_metrics["family"].tolist(),
            replicates=replicates,
            seed=seed + 2,
        )
        finite_counts = multiplicity_matrix(finite_plan)
        results["margin_rank_correlation"] = bootstrap_statistic(
            finite_metrics["margin_rank_correlation"], finite_counts
        )
        results["margin_rank_correlation"]["n_defined"] = int(finite.sum())
        results["margin_rank_correlation"]["n_undefined"] = int((~finite).sum())
        results["margin_rank_correlation"]["median"] = float(np.median(rho[finite]))

    return {
        "replicates": replicates,
        "seed": seed,
        "n_artifacts": int(len(metrics)),
        "n_nontrivial": int(nontrivial.sum()),
        "statistics": results,
    }


def bootstrap_family(
    metrics: pd.DataFrame,
    family: str,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """The same intervals restricted to one predicate family."""
    subset = metrics[metrics["family"] == family].reset_index(drop=True)
    return bootstrap_judge(subset, replicates=replicates, seed=seed)


def evaluate_robustness(
    statistics: dict,
    delta_tce_band: tuple[float, float],
    delta_trr_band: tuple[float, float],
    agreement_lower_bound: float,
) -> dict:
    """Apply the three preregistered practical bands. No threshold moves here."""
    tce = statistics["delta_tce"]
    trr = statistics["delta_trr"]
    agreement = statistics["agreement"]

    criteria = {
        "delta_tce_within_band": bool(
            delta_tce_band[0] <= tce["ci_low"] and tce["ci_high"] <= delta_tce_band[1]
        ),
        "delta_trr_within_band": bool(
            delta_trr_band[0] <= trr["ci_low"] and trr["ci_high"] <= delta_trr_band[1]
        ),
        "agreement_lower_bound_met": bool(
            agreement["ci_low"] >= agreement_lower_bound
        ),
    }
    failed = [name for name, ok in criteria.items() if not ok]
    return {
        "criteria": criteria,
        "failed_criteria": failed,
        "strong_robustness": not failed,
        "classification": "strong" if not failed else "not_strong",
    }
