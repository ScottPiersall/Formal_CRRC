"""Preregistered Day-1 metrics.

Primary
    * **MVR**  -- adjacent monotonicity violation rate at tolerance 0.05
    * **MVM**  -- adjacent monotonicity violation magnitude
    * **TCE**  -- threshold crossing error, ``|j_hat - j*|``

Secondary
    * zero-tolerance MVR, binary accuracy, Brier score,
      boundary-aligned response curve, within-artifact Spearman association.

All curve-level metrics take the nine probabilities of one artifact ordered by
strictness ``s = 0..8``. Nothing here recalibrates, smooths, or monotonises a
curve: the judge's raw normalised probabilities are used exactly as scored.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from formalcrrc.config import (
    DECISION_THRESHOLD,
    MVR_TOLERANCE,
    MVR_TOLERANCE_STRICT,
    N_STRICTNESS,
    NO_CROSSING_INDEX,
)


def _as_curve(probabilities: Sequence[float]) -> np.ndarray:
    curve = np.asarray(probabilities, dtype=float)
    if curve.shape != (N_STRICTNESS,):
        raise ValueError(
            f"expected {N_STRICTNESS} probabilities, got shape {curve.shape}"
        )
    if not np.all(np.isfinite(curve)):
        raise ValueError("curve contains non-finite probabilities")
    return curve


# --------------------------------------------------------------------------
# Primary metrics
# --------------------------------------------------------------------------


def monotonicity_violation_rate(
    probabilities: Sequence[float], tolerance: float = MVR_TOLERANCE
) -> float:
    """Fraction of the eight adjacent steps that rise by more than ``tolerance``.

    A well-behaved judge satisfies ``P(s+1) <= P(s)``; a step counts as a
    violation when ``P(s+1) > P(s) + tolerance``.

    The comparison is written in exactly that form rather than as
    ``diff > tolerance``: the two differ in floating point at the boundary, and
    a rise of precisely the tolerance must not count as a violation.
    """
    curve = _as_curve(probabilities)
    violations = curve[1:] > curve[:-1] + tolerance
    return float(np.mean(violations))


def monotonicity_violation_magnitude(probabilities: Sequence[float]) -> float:
    """Mean positive part of the eight adjacent steps.

    Measures how large the upward moves are, not merely how often they occur.
    """
    curve = _as_curve(probabilities)
    steps = np.diff(curve)
    return float(np.mean(np.maximum(0.0, steps)))


def predicted_first_fail_index(
    probabilities: Sequence[float], decision_threshold: float = DECISION_THRESHOLD
) -> int:
    """Return ``j_hat``, the first strictness at which ``P_met`` drops below 0.5.

    Returns :data:`~formalcrrc.config.NO_CROSSING_INDEX` (9) when the judge never
    falls below the boundary.
    """
    curve = _as_curve(probabilities)
    below = np.flatnonzero(curve < decision_threshold)
    return int(below[0]) if below.size else NO_CROSSING_INDEX


def threshold_crossing_error(
    probabilities: Sequence[float],
    true_first_fail_index: int,
    decision_threshold: float = DECISION_THRESHOLD,
) -> int:
    """Absolute distance between the judged and the formal pass/fail boundary."""
    predicted = predicted_first_fail_index(probabilities, decision_threshold)
    return int(abs(predicted - int(true_first_fail_index)))


# --------------------------------------------------------------------------
# Secondary metrics
# --------------------------------------------------------------------------


def binary_predictions(
    probabilities: Sequence[float], decision_threshold: float = DECISION_THRESHOLD
) -> np.ndarray:
    """Hard pass/fail decisions at the preregistered 0.5 boundary."""
    return (_as_curve(probabilities) >= decision_threshold).astype(int)


def accuracy(probabilities: Sequence[float], truth: Sequence[int]) -> float:
    """Agreement between the thresholded judge decision and the formal label."""
    predictions = binary_predictions(probabilities)
    labels = np.asarray(truth, dtype=int)
    return float(np.mean(predictions == labels))


def brier_score(probabilities: Sequence[float], truth: Sequence[int]) -> float:
    """Mean squared difference between ``P_met`` and the formal label."""
    curve = _as_curve(probabilities)
    labels = np.asarray(truth, dtype=float)
    return float(np.mean((curve - labels) ** 2))


def spearman_strictness(probabilities: Sequence[float]) -> float:
    """Within-artifact Spearman association between strictness and ``P_met``.

    Returns ``nan`` for a constant curve, where the statistic is undefined. The
    expected direction is non-positive.
    """
    curve = _as_curve(probabilities)
    if np.all(curve == curve[0]):
        return float("nan")
    result = stats.spearmanr(np.arange(N_STRICTNESS), curve)
    return float(result.statistic)


# --------------------------------------------------------------------------
# Frame-level assembly
# --------------------------------------------------------------------------

ARTIFACT_METRIC_COLUMNS: tuple[str, ...] = (
    "mvr",
    "mvr_strict",
    "mvm",
    "tce",
    "accuracy",
    "brier",
    "spearman",
)


def compute_artifact_metrics(scores: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-threshold scores into one metric row per (model, artifact).

    ``scores`` must carry ``model_id``, ``artifact_id``, ``family``,
    ``latent_level``, ``strictness_index``, ``p_met``, ``formal_truth`` and
    ``true_first_fail_index``, with all nine strictness levels present for every
    artifact.
    """
    required = {
        "model_id",
        "artifact_id",
        "family",
        "latent_level",
        "strictness_index",
        "p_met",
        "formal_truth",
        "true_first_fail_index",
    }
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"scores frame is missing columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    ordered = scores.sort_values(["model_id", "artifact_id", "strictness_index"])
    for (model_id, artifact_id), group in ordered.groupby(
        ["model_id", "artifact_id"], sort=True
    ):
        levels = group["strictness_index"].to_numpy()
        if not np.array_equal(levels, np.arange(N_STRICTNESS)):
            raise ValueError(
                f"{model_id}/{artifact_id}: expected strictness 0..8 exactly once, "
                f"got {levels.tolist()}"
            )
        curve = group["p_met"].to_numpy(dtype=float)
        truth = group["formal_truth"].to_numpy(dtype=int)
        j_star = int(group["true_first_fail_index"].iloc[0])
        rows.append(
            {
                "model_id": model_id,
                "artifact_id": artifact_id,
                "family": group["family"].iloc[0],
                "latent_level": int(group["latent_level"].iloc[0]),
                "true_first_fail_index": j_star,
                "predicted_first_fail_index": predicted_first_fail_index(curve),
                "mvr": monotonicity_violation_rate(curve, MVR_TOLERANCE),
                "mvr_strict": monotonicity_violation_rate(
                    curve, MVR_TOLERANCE_STRICT
                ),
                "mvm": monotonicity_violation_magnitude(curve),
                "tce": threshold_crossing_error(curve, j_star),
                "accuracy": accuracy(curve, truth),
                "brier": brier_score(curve, truth),
                "spearman": spearman_strictness(curve),
            }
        )
    return pd.DataFrame(rows)


def boundary_aligned_curve(scores: pd.DataFrame) -> pd.DataFrame:
    """Mean ``P_met`` as a function of signed distance from the true boundary.

    The signed distance is ``d = s - j*``: ``d < 0`` is a threshold the artifact
    formally satisfies, ``d >= 0`` one it formally fails.
    """
    frame = scores.copy()
    frame["signed_distance"] = (
        frame["strictness_index"] - frame["true_first_fail_index"]
    )
    grouped = frame.groupby(["model_id", "signed_distance"], sort=True)["p_met"]
    summary = grouped.agg(["mean", "std", "count"]).reset_index()
    summary = summary.rename(
        columns={"mean": "mean_p_met", "std": "sd_p_met", "count": "n"}
    )
    summary["se_p_met"] = summary["sd_p_met"] / np.sqrt(summary["n"])
    return summary
