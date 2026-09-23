"""Artifact-clustered bootstrap for the Day-3 oracle bounds.

Every replicate resamples artifacts within predicate family, carrying all nine
thresholds of an artifact together. The group oracles are **refitted inside each
replicate**: resampling the test set while holding a truth-optimised intercept
fixed would understate the uncertainty of a quantity that was itself chosen from
the data.

The artifact oracle needs no refitting -- OTCE is a property of a single curve
and its formal boundary, so a resampled artifact carries its own value.

Refitting an enumerated oracle 5,000 times would be prohibitive done naively, so
the per-artifact TCE at every candidate intercept is precomputed once into a
matrix. A replicate's mean TCE curve is then a single matrix product against the
replicate's artifact multiplicities, and the oracle is an argmin over that row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from formalcrrc import day3
from formalcrrc.bootstrap import percentile_interval
from formalcrrc.config import (
    BOOTSTRAP_CI,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    FAMILIES,
)

#: Replicates evaluated per matrix product, to bound peak memory.
CHUNK = 500


@dataclass(frozen=True)
class ArtifactPlan:
    """A family-stratified artifact resampling scheme shared by every judge."""

    artifact_ids: tuple[str, ...]
    families: tuple[str, ...]
    indices: np.ndarray
    column_families: tuple[str, ...]
    replicates: int
    seed: int

    def positions(self, family: str | None) -> np.ndarray:
        if family is None:
            return np.arange(len(self.artifact_ids))
        return np.flatnonzero(np.asarray(self.families) == family)

    def columns(self, family: str | None) -> np.ndarray:
        if family is None:
            return np.arange(self.indices.shape[1])
        return np.flatnonzero(np.asarray(self.column_families) == family)


def build_plan(
    artifact_ids: Sequence[str],
    families: Sequence[str],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> ArtifactPlan:
    """Draw the stratified index matrix once, for all judges."""
    if len(artifact_ids) != len(families):
        raise ValueError("artifact_ids and families must be the same length")
    family_of = np.asarray(families)
    rng = np.random.default_rng(seed)

    blocks: list[np.ndarray] = []
    column_families: list[str] = []
    for family in FAMILIES:
        positions = np.flatnonzero(family_of == family)
        if positions.size == 0:
            continue
        drawn = rng.integers(0, positions.size, size=(replicates, positions.size))
        blocks.append(positions[drawn])
        column_families.extend([family] * positions.size)
    if not blocks:
        raise ValueError("no artifacts to resample")

    return ArtifactPlan(
        artifact_ids=tuple(artifact_ids),
        families=tuple(families),
        indices=np.concatenate(blocks, axis=1),
        column_families=tuple(column_families),
        replicates=replicates,
        seed=seed,
    )


def multiplicity_matrix(
    plan: ArtifactPlan, family: str | None = None
) -> np.ndarray:
    """How many times each artifact appears in each replicate.

    Shape ``(replicates, n_artifacts_in_scope)``. Reduces a resampled mean to a
    matrix product, which is what makes refitting the oracle 5,000 times cheap.
    """
    positions = plan.positions(family)
    lookup = {p: i for i, p in enumerate(positions)}
    columns = plan.columns(family)
    counts = np.zeros((plan.replicates, positions.size), dtype=np.float64)
    for r in range(plan.replicates):
        drawn = plan.indices[r, columns]
        for value in drawn:
            counts[r, lookup[int(value)]] += 1.0
    return counts


def replicate_means(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Mean of a per-artifact quantity under every replicate."""
    values = np.asarray(values, dtype=float)
    return (counts @ values) / counts.sum(axis=1)


def replicate_rates(mask: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Rate of a per-artifact boolean under every replicate."""
    return replicate_means(np.asarray(mask, dtype=float), counts)


def refit_group_oracle_per_replicate(
    tce_matrix: np.ndarray, alphas: np.ndarray, counts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Refit the truth-optimised group intercept inside every replicate.

    ``tce_matrix`` is ``(n_artifacts, n_alphas)``. Returns the achieved mean TCE
    and the selected intercept per replicate, under the same preregistered
    tie-break as the point estimate.
    """
    tce_matrix = np.asarray(tce_matrix, dtype=np.float64)
    alphas = np.asarray(alphas, dtype=float)
    n_replicates = counts.shape[0]
    best_tce = np.empty(n_replicates)
    best_alpha = np.empty(n_replicates)

    # Preferred candidate order under the tie-break, computed once.
    preference = np.lexsort((alphas, np.abs(alphas)))
    ordered_alphas = alphas[preference]
    ordered_matrix = tce_matrix[:, preference]

    denominator = counts.sum(axis=1)
    for start in range(0, n_replicates, CHUNK):
        stop = min(start + CHUNK, n_replicates)
        block = (counts[start:stop] @ ordered_matrix) / denominator[start:stop, None]
        winners = block.argmin(axis=1)
        best_tce[start:stop] = block[np.arange(stop - start), winners]
        best_alpha[start:stop] = ordered_alphas[winners]
    return best_tce, best_alpha


def summarise(
    samples: np.ndarray, point: float, ci: float = BOOTSTRAP_CI
) -> dict[str, float]:
    """Point estimate with a percentile interval over the replicates."""
    finite = np.asarray(samples, dtype=float)
    finite = finite[np.isfinite(finite)]
    low, high = percentile_interval(finite, ci)
    return {
        "point": float(point),
        "ci_low": low,
        "ci_high": high,
        "bootstrap_mean": float(finite.mean()),
        "bootstrap_sd": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        "fraction_positive": float(np.mean(finite > 0.0)),
        "excludes_zero": bool(low > 0.0 or high < 0.0),
        "n_replicates": int(finite.size),
    }


def paired_difference(
    samples_a: np.ndarray, samples_b: np.ndarray, point: float, ci: float = BOOTSTRAP_CI
) -> dict[str, float]:
    """Judge-versus-judge difference on identical artifact resamples."""
    return summarise(np.asarray(samples_a) - np.asarray(samples_b), point, ci)


def bootstrap_judge(
    metrics: pd.DataFrame,
    margins_block: np.ndarray,
    j_star: np.ndarray,
    plan: ArtifactPlan,
    counts_overall: np.ndarray,
    counts_by_family: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """All Day-3 bootstrap quantities for one judge.

    ``metrics`` must be ordered to match ``plan.artifact_ids``; ``margins_block``
    and ``j_star`` likewise. Group oracles are refitted per replicate; the raw
    and artifact-oracle quantities are per-artifact values resampled directly.
    """
    ordered = metrics.set_index("artifact_id").loc[list(plan.artifact_ids)]
    out: dict[str, np.ndarray] = {}

    # -- quantities that are simple per-artifact means or rates --------------
    for name, values in (
        ("tce_raw", ordered["tce_raw"].to_numpy(float)),
        ("otce", ordered["otce"].to_numpy(float)),
        ("delta_tce_artifact", ordered["delta_tce_artifact"].to_numpy(float)),
        ("reachable_count", ordered["reachable_count"].to_numpy(float)),
        ("prefix_record_rate", ordered["prefix_record_rate"].to_numpy(float)),
        ("mmvr", ordered["mmvr"].to_numpy(float)),
    ):
        out[name] = replicate_means(values, counts_overall)

    for name, mask in (
        ("exact_rate_raw", ordered["tce_raw"].to_numpy() == 0),
        ("within_one_raw", ordered["tce_raw"].to_numpy() <= 1),
        ("exact_rate_otce", ordered["otce"].to_numpy() == 0),
        ("within_one_otce", ordered["otce"].to_numpy() <= 1),
        ("slr1", ordered["otce"].to_numpy() > 0),
        ("slr2", ordered["otce"].to_numpy() > 1),
        ("trr_overall", ordered["translation_reachable"].to_numpy(bool)),
    ):
        out[name] = replicate_rates(mask, counts_overall)

    # Nontrivial reachability is a rate over a subset, so it needs its own
    # numerator and denominator per replicate.
    nontrivial = ordered["nontrivial_boundary"].to_numpy(bool).astype(float)
    reachable_nontrivial = (
        ordered["translation_reachable"].to_numpy(bool)
        & ordered["nontrivial_boundary"].to_numpy(bool)
    ).astype(float)
    denominator = counts_overall @ nontrivial
    with np.errstate(invalid="ignore", divide="ignore"):
        out["trr_nontrivial"] = np.where(
            denominator > 0, (counts_overall @ reachable_nontrivial) / denominator, np.nan
        )

    # -- oracle global, refitted per replicate -------------------------------
    alphas_global = day3.group_candidate_intercepts(margins_block)
    matrix_global = day3.tce_matrix(margins_block, j_star, alphas_global)
    tce_global, alpha_global = refit_group_oracle_per_replicate(
        matrix_global, alphas_global, counts_overall
    )
    out["tce_oracle_global"] = tce_global
    out["alpha_oracle_global"] = alpha_global

    # -- oracle family, refitted per replicate within each family ------------
    family_of = np.asarray(plan.families)
    family_tce = np.zeros(plan.replicates)
    family_weight = np.zeros(plan.replicates)
    for family in FAMILIES:
        rows = np.flatnonzero(family_of == family)
        if rows.size == 0:
            continue
        counts_family = counts_by_family[family]
        alphas_f = day3.group_candidate_intercepts(margins_block[rows])
        matrix_f = day3.tce_matrix(margins_block[rows], j_star[rows], alphas_f)
        tce_f, alpha_f = refit_group_oracle_per_replicate(
            matrix_f, alphas_f, counts_family
        )
        out[f"tce_oracle_family_{family}"] = tce_f
        out[f"alpha_oracle_family_{family}"] = alpha_f
        weight = counts_family.sum(axis=1)
        family_tce += tce_f * weight
        family_weight += weight
    out["tce_oracle_family"] = family_tce / family_weight

    return out
