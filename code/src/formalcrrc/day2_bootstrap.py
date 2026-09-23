"""Two-stage cluster bootstrap for the Day-2 held-out calibration design.

Day 2 has two independent sources of sampling error: which artifacts happened to
land in the calibration set, and which landed in the test set. Resampling only
the test set would understate the uncertainty, because it would treat the fitted
intercept as if it were known exactly.

Each replicate therefore:

1. resamples CALIBRATION artifacts within predicate family,
2. **refits** the intercept on that resampled calibration data,
3. independently resamples TEST artifacts within family,
4. applies the refitted intercept to that test sample,
5. computes the paired raw-versus-corrected statistics.

All nine thresholds of an artifact move together in both stages. The same
calibration and test resamples are reused across judges, so judge-versus-judge
differences are paired by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from formalcrrc import day2
from formalcrrc.bootstrap import percentile_interval
from formalcrrc.config import (
    BOOTSTRAP_CI,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    FAMILIES,
    N_STRICTNESS,
)

#: Sub-stream indices so the two stages never share a random stream.
CALIBRATION_STREAM: int = 0
TEST_STREAM: int = 1


@dataclass(frozen=True)
class PartitionPlan:
    """A family-stratified artifact resampling scheme for one partition."""

    name: str
    artifact_ids: tuple[str, ...]
    artifact_families: tuple[str, ...]
    indices: np.ndarray
    column_families: tuple[str, ...]

    def artifact_positions(self, family: str | None) -> np.ndarray:
        if family is None:
            return np.arange(len(self.artifact_ids))
        return np.flatnonzero(np.asarray(self.artifact_families) == family)

    def index_columns(self, family: str | None) -> np.ndarray:
        if family is None:
            return np.arange(self.indices.shape[1])
        return np.flatnonzero(np.asarray(self.column_families) == family)


@dataclass(frozen=True)
class TwoStagePlan:
    """The frozen calibration and test resampling schemes, shared by all judges."""

    calibration: PartitionPlan
    test: PartitionPlan
    replicates: int
    seed: int


def _build_partition_plan(
    name: str,
    artifact_ids: Sequence[str],
    families: Sequence[str],
    replicates: int,
    rng: np.random.Generator,
) -> PartitionPlan:
    family_of = np.asarray(families)
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
        raise ValueError(f"no artifacts to resample in partition {name!r}")
    return PartitionPlan(
        name=name,
        artifact_ids=tuple(artifact_ids),
        artifact_families=tuple(families),
        indices=np.concatenate(blocks, axis=1),
        column_families=tuple(column_families),
    )


def build_two_stage_plan(
    calibration_artifacts: pd.DataFrame,
    test_artifacts: pd.DataFrame,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> TwoStagePlan:
    """Draw both resampling schemes once, from independent sub-streams.

    ``calibration_artifacts`` and ``test_artifacts`` each need one row per
    artifact with ``artifact_id`` and ``family``, in a fixed canonical order.
    """
    cal_rng = np.random.default_rng([seed, CALIBRATION_STREAM])
    test_rng = np.random.default_rng([seed, TEST_STREAM])
    return TwoStagePlan(
        calibration=_build_partition_plan(
            "calibration",
            calibration_artifacts["artifact_id"].tolist(),
            calibration_artifacts["family"].tolist(),
            replicates,
            cal_rng,
        ),
        test=_build_partition_plan(
            "test",
            test_artifacts["artifact_id"].tolist(),
            test_artifacts["family"].tolist(),
            replicates,
            test_rng,
        ),
        replicates=replicates,
        seed=seed,
    )


# --------------------------------------------------------------------------
# Per-artifact stacks
# --------------------------------------------------------------------------


def stack_calibration(
    scores: pd.DataFrame, plan: PartitionPlan, margin_column: str = "margin"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Arrange calibration rows as ``(n_artifacts, 9)`` margin and truth blocks.

    Returns the margins, the formal labels and the family index of each
    artifact, all ordered to match ``plan.artifact_ids`` so a bootstrap index
    selects whole artifacts.
    """
    ordered = scores.sort_values(["artifact_id", "strictness_index"])
    grouped = ordered.groupby("artifact_id", sort=True)

    lookup_margin = {}
    lookup_truth = {}
    for artifact_id, group in grouped:
        if len(group) != N_STRICTNESS:
            raise ValueError(f"{artifact_id}: expected {N_STRICTNESS} rows")
        lookup_margin[artifact_id] = group[margin_column].to_numpy(dtype=float)
        lookup_truth[artifact_id] = group["formal_truth"].to_numpy(dtype=float)

    missing = set(plan.artifact_ids) - set(lookup_margin)
    if missing:
        raise ValueError(f"calibration scores are missing {len(missing)} artifacts")

    margins = np.stack([lookup_margin[a] for a in plan.artifact_ids])
    truths = np.stack([lookup_truth[a] for a in plan.artifact_ids])
    family_index = np.asarray(
        [FAMILIES.index(f) for f in plan.artifact_families], dtype=int
    )
    return margins, truths, family_index


def refit_alpha_for_replicate(
    margins: np.ndarray,
    truths: np.ndarray,
    replicate_indices: np.ndarray,
) -> float:
    """Refit the global intercept on one bootstrap calibration sample."""
    selected_margins = margins[replicate_indices].ravel()
    selected_truths = truths[replicate_indices].ravel()
    return day2.fit_intercept(selected_margins, selected_truths).alpha


def refit_family_alphas_for_replicate(
    margins: np.ndarray,
    truths: np.ndarray,
    family_index: np.ndarray,
    replicate_indices: np.ndarray,
) -> dict[str, float]:
    """Refit one intercept per family on one bootstrap calibration sample."""
    selected = replicate_indices
    families_selected = family_index[selected]
    alphas: dict[str, float] = {}
    for position, family in enumerate(FAMILIES):
        mask = families_selected == position
        if not np.any(mask):
            alphas[family] = float("nan")
            continue
        rows = selected[mask]
        alphas[family] = day2.fit_intercept(
            margins[rows].ravel(), truths[rows].ravel()
        ).alpha
    return alphas


# --------------------------------------------------------------------------
# Test-side evaluation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TestStack:
    """TEST margins, labels and boundaries arranged per artifact."""

    artifact_ids: tuple[str, ...]
    margins: np.ndarray  # (n_artifacts, 9)
    truths: np.ndarray  # (n_artifacts, 9)
    j_star: np.ndarray  # (n_artifacts,)
    family_index: np.ndarray  # (n_artifacts,)


def stack_test(
    scores: pd.DataFrame, plan: PartitionPlan, margin_column: str = "margin"
) -> TestStack:
    """Arrange test rows per artifact, ordered to match ``plan.artifact_ids``."""
    ordered = scores.sort_values(["artifact_id", "strictness_index"])
    margin_lookup, truth_lookup, boundary_lookup = {}, {}, {}
    for artifact_id, group in ordered.groupby("artifact_id", sort=True):
        if len(group) != N_STRICTNESS:
            raise ValueError(f"{artifact_id}: expected {N_STRICTNESS} rows")
        margin_lookup[artifact_id] = group[margin_column].to_numpy(dtype=float)
        truth_lookup[artifact_id] = group["formal_truth"].to_numpy(dtype=float)
        boundary_lookup[artifact_id] = int(group["true_first_fail_index"].iloc[0])

    missing = set(plan.artifact_ids) - set(margin_lookup)
    if missing:
        raise ValueError(f"test scores are missing {len(missing)} artifacts")

    return TestStack(
        artifact_ids=plan.artifact_ids,
        margins=np.stack([margin_lookup[a] for a in plan.artifact_ids]),
        truths=np.stack([truth_lookup[a] for a in plan.artifact_ids]),
        j_star=np.asarray(
            [boundary_lookup[a] for a in plan.artifact_ids], dtype=int
        ),
        family_index=np.asarray(
            [FAMILIES.index(f) for f in plan.artifact_families], dtype=int
        ),
    )


def crossing_index(shifted_margins: np.ndarray) -> np.ndarray:
    """Vectorised first-fail index over an ``(n, 9)`` block of shifted margins.

    ``sigma(M) < 0.5`` exactly when ``M < 0``, so the crossing is the first
    negative entry; artifacts that never go negative take the sentinel 9.
    """
    below = shifted_margins < 0.0
    any_below = below.any(axis=1)
    first = np.where(any_below, below.argmax(axis=1), N_STRICTNESS)
    return first.astype(int)


def tce_for_shift(stack: TestStack, alpha: float | np.ndarray) -> np.ndarray:
    """Threshold crossing error for every test artifact under a shift."""
    if np.isscalar(alpha):
        shifted = stack.margins + float(alpha)
    else:
        shifted = stack.margins + np.asarray(alpha, dtype=float)[:, None]
    return np.abs(crossing_index(shifted) - stack.j_star)


def family_shift_vector(
    stack: TestStack, alphas: dict[str, float]
) -> np.ndarray:
    """Per-artifact shift implied by a family-specific intercept mapping."""
    lookup = np.asarray([alphas[f] for f in FAMILIES], dtype=float)
    return lookup[stack.family_index]


# --------------------------------------------------------------------------
# Replicate loop
# --------------------------------------------------------------------------


def bootstrap_judge(
    calibration_scores: pd.DataFrame,
    test_scores: pd.DataFrame,
    plan: TwoStagePlan,
    *,
    margin_column: str = "margin",
    progress: Callable[[int], None] | None = None,
) -> dict[str, np.ndarray]:
    """Run the full two-stage bootstrap for one judge.

    Returns per-replicate arrays for the fitted intercepts and for every paired
    held-out quantity. Nothing is summarised here.
    """
    cal_margins, cal_truths, cal_family = stack_calibration(
        calibration_scores, plan.calibration, margin_column
    )
    stack = stack_test(test_scores, plan.test, margin_column)

    replicates = plan.replicates
    out: dict[str, np.ndarray] = {
        "alpha_global": np.empty(replicates),
        "tce_raw": np.empty(replicates),
        "tce_global": np.empty(replicates),
        "tce_family": np.empty(replicates),
        "delta_tce_global": np.empty(replicates),
        "delta_tce_family": np.empty(replicates),
        "exact_rate_raw": np.empty(replicates),
        "exact_rate_global": np.empty(replicates),
        "within_one_raw": np.empty(replicates),
        "within_one_global": np.empty(replicates),
    }
    for family in FAMILIES:
        out[f"alpha_{family}"] = np.empty(replicates)
        out[f"tce_raw_{family}"] = np.empty(replicates)
        out[f"tce_global_{family}"] = np.empty(replicates)
        out[f"tce_family_{family}"] = np.empty(replicates)
        out[f"delta_tce_global_{family}"] = np.empty(replicates)

    for r in range(replicates):
        cal_rows = plan.calibration.indices[r]
        alpha = refit_alpha_for_replicate(cal_margins, cal_truths, cal_rows)
        family_alphas = refit_family_alphas_for_replicate(
            cal_margins, cal_truths, cal_family, cal_rows
        )

        test_rows = plan.test.indices[r]
        sub = TestStack(
            artifact_ids=tuple(stack.artifact_ids[i] for i in test_rows),
            margins=stack.margins[test_rows],
            truths=stack.truths[test_rows],
            j_star=stack.j_star[test_rows],
            family_index=stack.family_index[test_rows],
        )

        tce_raw = tce_for_shift(sub, 0.0)
        tce_global = tce_for_shift(sub, alpha)
        tce_family = tce_for_shift(sub, family_shift_vector(sub, family_alphas))

        out["alpha_global"][r] = alpha
        out["tce_raw"][r] = tce_raw.mean()
        out["tce_global"][r] = tce_global.mean()
        out["tce_family"][r] = tce_family.mean()
        out["delta_tce_global"][r] = (tce_raw - tce_global).mean()
        out["delta_tce_family"][r] = (tce_global - tce_family).mean()
        out["exact_rate_raw"][r] = np.mean(tce_raw == 0)
        out["exact_rate_global"][r] = np.mean(tce_global == 0)
        out["within_one_raw"][r] = np.mean(tce_raw <= 1)
        out["within_one_global"][r] = np.mean(tce_global <= 1)

        for position, family in enumerate(FAMILIES):
            mask = sub.family_index == position
            out[f"alpha_{family}"][r] = family_alphas[family]
            if np.any(mask):
                out[f"tce_raw_{family}"][r] = tce_raw[mask].mean()
                out[f"tce_global_{family}"][r] = tce_global[mask].mean()
                out[f"tce_family_{family}"][r] = tce_family[mask].mean()
                out[f"delta_tce_global_{family}"][r] = (
                    tce_raw[mask] - tce_global[mask]
                ).mean()
            else:
                for key in (
                    f"tce_raw_{family}",
                    f"tce_global_{family}",
                    f"tce_family_{family}",
                    f"delta_tce_global_{family}",
                ):
                    out[key][r] = np.nan

        if progress is not None and (r + 1) % 500 == 0:
            progress(r + 1)

    return out


def summarise(
    samples: np.ndarray, point: float, ci: float = BOOTSTRAP_CI
) -> dict[str, float]:
    """Point estimate, percentile interval, and the fraction of replicates > 0."""
    finite = samples[np.isfinite(samples)]
    low, high = percentile_interval(finite, ci)
    return {
        "point": float(point),
        "ci_low": low,
        "ci_high": high,
        "bootstrap_mean": float(finite.mean()),
        "bootstrap_sd": float(finite.std(ddof=1)),
        "fraction_positive": float(np.mean(finite > 0.0)),
        "excludes_zero": bool(low > 0.0 or high < 0.0),
        "n_replicates": int(finite.size),
    }


def paired_difference(
    samples_a: np.ndarray, samples_b: np.ndarray, point: float, ci: float = BOOTSTRAP_CI
) -> dict[str, float]:
    """Judge-versus-judge difference on identical calibration and test resamples."""
    difference = samples_a - samples_b
    return summarise(difference, point, ci)
