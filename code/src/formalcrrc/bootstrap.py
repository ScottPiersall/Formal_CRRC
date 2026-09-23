"""Paired, cluster-aware bootstrap over artifacts.

The resampling unit is the **artifact**, never the individual rubric-response
cell: all nine thresholds of an artifact are carried together, because the nine
probabilities of one curve are anything but independent.

A single index matrix of shape ``(replicates, n_artifacts)`` is drawn once and
reused for every judge. That makes every judge-vs-judge comparison paired by
construction: replicate ``r`` contains the same artifacts for both models.

Resampling is stratified by predicate family, so each replicate holds the
designed 108 artifacts per family. Intervals are percentile intervals.
No hyperparameter anywhere in Day 1 is selected from these outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from formalcrrc.config import (
    BOOTSTRAP_CI,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    FAMILIES,
)

#: Quantities for which 95% intervals are preregistered.
BOOTSTRAP_METRICS: tuple[str, ...] = ("mvr", "mvm", "tce", "accuracy", "brier")


@dataclass(frozen=True)
class BootstrapPlan:
    """A frozen resampling scheme shared by every judge.

    ``artifact_ids`` and ``artifact_families`` describe the canonical artifact
    order; ``indices`` holds positions into that order, and ``column_families``
    records which family stratum each column of ``indices`` was drawn from.
    """

    artifact_ids: tuple[str, ...]
    artifact_families: tuple[str, ...]
    indices: np.ndarray
    column_families: tuple[str, ...]
    seed: int
    replicates: int

    def artifact_positions(self, family: str | None) -> np.ndarray:
        """Canonical positions of the artifacts in scope."""
        if family is None:
            return np.arange(len(self.artifact_ids))
        return np.flatnonzero(np.asarray(self.artifact_families) == family)

    def index_columns(self, family: str | None) -> np.ndarray:
        """Columns of ``indices`` belonging to the scope."""
        if family is None:
            return np.arange(self.indices.shape[1])
        return np.flatnonzero(np.asarray(self.column_families) == family)


def build_plan(
    artifact_ids: Sequence[str],
    families: Sequence[str],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> BootstrapPlan:
    """Draw the family-stratified index matrix once, for all judges."""
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

    return BootstrapPlan(
        artifact_ids=tuple(artifact_ids),
        artifact_families=tuple(families),
        indices=np.concatenate(blocks, axis=1),
        column_families=tuple(column_families),
        seed=seed,
        replicates=replicates,
    )


def replicate_means(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Mean of ``values`` under every bootstrap replicate."""
    return np.asarray(values, dtype=float)[indices].mean(axis=1)


def percentile_interval(
    samples: np.ndarray, ci: float = BOOTSTRAP_CI
) -> tuple[float, float]:
    """Two-sided percentile interval at coverage ``ci``."""
    alpha = (1.0 - ci) / 2.0
    low, high = np.quantile(samples, [alpha, 1.0 - alpha])
    return float(low), float(high)


def aligned_values(
    metrics: pd.DataFrame, plan: BootstrapPlan, metric: str
) -> np.ndarray:
    """Return ``metric`` for one judge, ordered to match ``plan.artifact_ids``."""
    indexed = metrics.set_index("artifact_id")
    missing = set(plan.artifact_ids) - set(indexed.index)
    if missing:
        raise ValueError(f"metric frame is missing {len(missing)} artifacts")
    return indexed.loc[list(plan.artifact_ids), metric].to_numpy(dtype=float)


def bootstrap_metric(
    metrics: pd.DataFrame,
    plan: BootstrapPlan,
    metric: str,
    *,
    family: str | None = None,
    ci: float = BOOTSTRAP_CI,
) -> dict[str, float]:
    """Point estimate and percentile interval for one judge, metric and scope."""
    values = aligned_values(metrics, plan, metric)
    positions = plan.artifact_positions(family)
    samples = replicate_means(values, plan.indices[:, plan.index_columns(family)])
    low, high = percentile_interval(samples, ci)
    return {
        "point": float(values[positions].mean()),
        "ci_low": low,
        "ci_high": high,
        "bootstrap_mean": float(samples.mean()),
        "bootstrap_sd": float(samples.std(ddof=1)),
        "n_units": int(positions.size),
    }


def bootstrap_table(
    metrics_by_model: dict[str, pd.DataFrame],
    plan: BootstrapPlan,
    *,
    metrics: Iterable[str] = BOOTSTRAP_METRICS,
    ci: float = BOOTSTRAP_CI,
) -> pd.DataFrame:
    """Per-judge intervals for every metric, overall and per predicate family."""
    rows: list[dict[str, object]] = []
    for model_id, frame in metrics_by_model.items():
        for metric in metrics:
            for scope in [None, *FAMILIES]:
                result = bootstrap_metric(frame, plan, metric, family=scope, ci=ci)
                rows.append(
                    {
                        "model_id": model_id,
                        "scope": scope or "overall",
                        "metric": metric,
                        **result,
                    }
                )
    return pd.DataFrame(rows)


def paired_difference(
    metrics_a: pd.DataFrame,
    metrics_b: pd.DataFrame,
    plan: BootstrapPlan,
    metric: str,
    *,
    family: str | None = None,
    ci: float = BOOTSTRAP_CI,
) -> dict[str, float]:
    """Paired bootstrap for ``mean(metric_a) - mean(metric_b)``.

    Both judges are evaluated on the identical resampled artifact sets, so the
    difference is paired at the artifact level.
    """
    values_a = aligned_values(metrics_a, plan, metric)
    values_b = aligned_values(metrics_b, plan, metric)
    positions = plan.artifact_positions(family)
    columns = plan.indices[:, plan.index_columns(family)]
    samples = replicate_means(values_a, columns) - replicate_means(values_b, columns)
    low, high = percentile_interval(samples, ci)
    return {
        "point": float(values_a[positions].mean() - values_b[positions].mean()),
        "ci_low": low,
        "ci_high": high,
        "excludes_zero": bool(low > 0.0 or high < 0.0),
        "n_units": int(positions.size),
    }


def paired_comparison_table(
    metrics_by_model: dict[str, pd.DataFrame],
    plan: BootstrapPlan,
    *,
    metrics: Iterable[str] = BOOTSTRAP_METRICS,
    ci: float = BOOTSTRAP_CI,
) -> pd.DataFrame:
    """All unordered judge pairs x metrics, overall and per predicate family."""
    model_ids = list(metrics_by_model)
    rows: list[dict[str, object]] = []
    for i, model_a in enumerate(model_ids):
        for model_b in model_ids[i + 1 :]:
            for metric in metrics:
                for scope in [None, *FAMILIES]:
                    result = paired_difference(
                        metrics_by_model[model_a],
                        metrics_by_model[model_b],
                        plan,
                        metric,
                        family=scope,
                        ci=ci,
                    )
                    rows.append(
                        {
                            "model_a": model_a,
                            "model_b": model_b,
                            "scope": scope or "overall",
                            "metric": metric,
                            **result,
                        }
                    )
    return pd.DataFrame(rows)
