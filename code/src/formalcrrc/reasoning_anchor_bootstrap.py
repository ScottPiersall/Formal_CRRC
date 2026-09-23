"""Artifact-clustered bootstrap for the reasoning-capable external anchor.

The resampling unit is the **artifact curve**, never the threshold row. All nine
strictness levels of an artifact are drawn together, because they are nine
measurements of one item and treating them as independent would understate every
interval. Family strata are preserved so a replicate keeps the design's 108
artifacts per family.

Row-level quantities -- accuracy and the Brier score -- are first reduced to a
per-artifact value and only then resampled, so the clustering applies to them
too.

Nontrivial reachability is bootstrapped over complete nontrivial curves only,
matching the endpoint's definition; a curve with an extreme true boundary
carries no ordering information and is not silently counted as a success.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from formalcrrc.config import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, FAMILIES


def percentile_ci(samples: np.ndarray, level: float = 95.0) -> tuple[float, float]:
    """Two-sided percentile interval, ignoring replicates that are undefined."""
    finite = np.asarray(samples, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (float("nan"), float("nan"))
    tail = (100.0 - level) / 2.0
    return (
        float(np.percentile(finite, tail)),
        float(np.percentile(finite, 100.0 - tail)),
    )


def summarise(samples: np.ndarray, point: float, level: float = 95.0) -> dict[str, Any]:
    low, high = percentile_ci(samples, level)
    return {
        "point": float(point),
        "ci_low": low,
        "ci_high": high,
        "ci_level": level,
        "n_replicates": int(np.sum(np.isfinite(np.asarray(samples, dtype=float)))),
    }


@dataclass(frozen=True)
class CurvePlan:
    """A family-stratified resampling of artifact curves."""

    artifact_ids: tuple[str, ...]
    families: tuple[str, ...]
    indices: np.ndarray  # (replicates, n_artifacts) positional draws
    replicates: int
    seed: int

    def positions(self, family: str | None = None) -> np.ndarray:
        if family is None:
            return np.arange(len(self.artifact_ids))
        return np.flatnonzero(np.asarray(self.families) == family)


def build_plan(
    artifact_ids: Sequence[str],
    families: Sequence[str],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> CurvePlan:
    """Draw the stratified index matrix once so every metric shares replicates.

    Sharing one plan across metrics keeps the intervals mutually consistent: the
    same replicate universe underlies TCE, TRR and every secondary quantity.
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
    indices = np.concatenate(columns, axis=1) if columns else np.empty((replicates, 0), int)
    return CurvePlan(
        artifact_ids=tuple(artifact_ids),
        families=tuple(families),
        indices=indices,
        replicates=replicates,
        seed=seed,
    )


def replicate_means(values: np.ndarray, plan: CurvePlan, mask: np.ndarray | None = None) -> np.ndarray:
    """Mean of ``values`` under every replicate, restricted to ``mask``.

    A replicate that happens to draw no eligible curve yields NaN rather than a
    fabricated zero; those replicates are dropped when the interval is taken.
    """
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


def bootstrap_anchor(
    metrics: pd.DataFrame,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Every preregistered interval, from one shared plan.

    ``metrics`` is one row per complete artifact curve, carrying ``tce_raw``,
    ``translation_reachable``, ``nontrivial_boundary``, ``reachable_count``,
    ``artifact_accuracy``, ``artifact_brier`` and ``family``.
    """
    frame = metrics.reset_index(drop=True)
    plan = build_plan(
        frame["artifact_id"].tolist(),
        frame["family"].tolist(),
        replicates=replicates,
        seed=seed,
    )

    tce = frame["tce_raw"].to_numpy(dtype=float)
    reachable = frame["translation_reachable"].to_numpy(dtype=float)
    nontrivial = frame["nontrivial_boundary"].to_numpy(dtype=bool)
    rc_count = frame["reachable_count"].to_numpy(dtype=float)
    accuracy = frame["artifact_accuracy"].to_numpy(dtype=float)
    brier = frame["artifact_brier"].to_numpy(dtype=float)

    trr_samples = replicate_means(reachable, plan, mask=nontrivial)
    results: dict[str, Any] = {
        "bootstrap": {
            "replicates": replicates,
            "seed": seed,
            "unit": "artifact_curve",
            "stratified_by": "family",
            "ci": "percentile, 95%",
        },
        "overall": {
            "mean_tce": summarise(replicate_means(tce, plan), float(np.mean(tce))),
            "trr_nontrivial": summarise(
                trr_samples, float(np.mean(reachable[nontrivial]))
            ),
            "unreachable_fraction": summarise(
                1.0 - trr_samples, float(1.0 - np.mean(reachable[nontrivial]))
            ),
            "accuracy": summarise(
                replicate_means(accuracy, plan), float(np.mean(accuracy))
            ),
            "brier": summarise(replicate_means(brier, plan), float(np.mean(brier))),
            "mean_reachable_count": summarise(
                replicate_means(rc_count, plan), float(np.mean(rc_count))
            ),
        },
        "by_family": {},
    }

    for family in FAMILIES:
        in_family = (frame["family"] == family).to_numpy(dtype=bool)
        if not in_family.any():
            continue
        family_nontrivial = in_family & nontrivial
        family_trr = replicate_means(reachable, plan, mask=family_nontrivial)
        results["by_family"][family] = {
            "n_curves": int(in_family.sum()),
            "n_nontrivial": int(family_nontrivial.sum()),
            "mean_tce": summarise(
                replicate_means(tce, plan, mask=in_family),
                float(np.mean(tce[in_family])),
            ),
            "trr_nontrivial": summarise(
                family_trr,
                float(np.mean(reachable[family_nontrivial]))
                if family_nontrivial.any()
                else float("nan"),
            ),
            "unreachable_fraction": summarise(
                1.0 - family_trr,
                float(1.0 - np.mean(reachable[family_nontrivial]))
                if family_nontrivial.any()
                else float("nan"),
            ),
            "accuracy": summarise(
                replicate_means(accuracy, plan, mask=in_family),
                float(np.mean(accuracy[in_family])),
            ),
            "brier": summarise(
                replicate_means(brier, plan, mask=in_family),
                float(np.mean(brier[in_family])),
            ),
        }
    return results


def to_frame(results: dict[str, Any]) -> pd.DataFrame:
    """Flatten the nested interval record into one tidy table."""
    rows: list[dict[str, Any]] = []
    for metric, block in results["overall"].items():
        rows.append({"scope": "overall", "family": None, "metric": metric, **block})
    for family, metrics in results["by_family"].items():
        for metric, block in metrics.items():
            if not isinstance(block, dict):
                continue
            rows.append(
                {"scope": "family", "family": family, "metric": metric, **block}
            )
    return pd.DataFrame(rows)
