"""Day-2 bias-vs-shape decomposition: margins, intercept calibration, metrics.

The Day-2 intervention is deliberately the weakest correction that can move a
curve's location: a single additive constant in decision-margin space,

    M'(s) = M(s) + alpha

with the slope fixed at exactly 1. Because the same constant is added at every
strictness level, adjacent differences are untouched:

    M'(s+1) - M'(s) = M(s+1) - M(s)

so the correction can move a curve relative to the decision boundary but cannot
change its shape -- not its reversals, not its span, not its rank ordering. That
invariance is the experimental control, and it is asserted by the test suite
rather than assumed.

Nothing here fits a slope, a temperature, an isotonic map, or anything else with
more than one free parameter per fitted unit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from formalcrrc.config import (
    DECISION_THRESHOLD,
    FAMILIES,
    N_STRICTNESS,
    NO_CROSSING_INDEX,
)

#: CLUSTER iterations allowed when solving the one-dimensional score equation.
MAX_CLUSTER_STEPS: int = 100

#: Convergence tolerance on the score equation residual.
CLUSTER_TOLERANCE: float = 1e-10

#: Bracket half-width used by the bisection safeguard, in margin units.
BRACKET_LIMIT: float = 200.0


# --------------------------------------------------------------------------
# Margins
# --------------------------------------------------------------------------


def decision_margin(score_met: np.ndarray, score_not_met: np.ndarray) -> np.ndarray:
    """The two-label decision margin ``M = S_A - S_B``.

    ``S_A`` is the raw score for answer A (criterion met) and ``S_B`` for answer
    B. Positive margins favour "met". Works for both Day-1 scoring rules: raw
    next-token logits and exact continuation log-likelihoods.
    """
    return np.asarray(score_met, dtype=float) - np.asarray(
        score_not_met, dtype=float
    )


def sigmoid(margin: np.ndarray | float) -> np.ndarray:
    """Numerically stable logistic function."""
    margin = np.asarray(margin, dtype=float)
    out = np.empty_like(margin)
    positive = margin >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-margin[positive]))
    exp_margin = np.exp(margin[~positive])
    out[~positive] = exp_margin / (1.0 + exp_margin)
    return out


def add_margin_column(scores: pd.DataFrame) -> pd.DataFrame:
    """Attach the decision margin to a raw-score frame."""
    frame = scores.copy()
    frame["margin"] = decision_margin(
        frame["raw_score_met"].to_numpy(), frame["raw_score_not_met"].to_numpy()
    )
    return frame


# --------------------------------------------------------------------------
# Intercept-only calibration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class InterceptFit:
    """The single fitted location parameter and its convergence record."""

    alpha: float
    n_rows: int
    n_positive: int
    converged: bool
    iterations: int
    residual: float
    degenerate: bool
    method: str = "CLUSTER_with_bisection_safeguard"

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "n_rows": self.n_rows,
            "n_positive": self.n_positive,
            "converged": self.converged,
            "iterations": self.iterations,
            "residual": self.residual,
            "degenerate": self.degenerate,
            "method": self.method,
        }


def _score_equation(margins: np.ndarray, truths: np.ndarray, alpha: float) -> float:
    """``sum_i [sigma(M_i + alpha) - Y_i]``; zero at the log-loss minimum."""
    return float(np.sum(sigmoid(margins + alpha) - truths))


def fit_intercept(
    margins: Sequence[float], truths: Sequence[int]
) -> InterceptFit:
    """Solve for the intercept that minimises calibration log-loss.

    The objective is convex in ``alpha`` with derivative
    ``sum_i [sigma(M_i + alpha) - Y_i]``, which is strictly increasing, so the
    root is unique whenever the labels are not all identical. CLUSTER's method is
    used with a bisection safeguard so that a bad step can never diverge.

    An all-zero or all-one label vector has no finite minimiser; that case is
    reported as ``degenerate`` rather than silently clipped.
    """
    margins = np.asarray(margins, dtype=float)
    truths = np.asarray(truths, dtype=float)
    if margins.shape != truths.shape:
        raise ValueError("margins and truths must have the same shape")
    if margins.size == 0:
        raise ValueError("cannot fit an intercept on an empty sample")

    n_positive = int(truths.sum())
    if n_positive == 0 or n_positive == margins.size:
        return InterceptFit(
            alpha=float("nan"),
            n_rows=int(margins.size),
            n_positive=n_positive,
            converged=False,
            iterations=0,
            residual=float("nan"),
            degenerate=True,
        )

    low, high = -BRACKET_LIMIT, BRACKET_LIMIT
    alpha = 0.0
    iterations = 0
    for iterations in range(1, MAX_CLUSTER_STEPS + 1):
        probabilities = sigmoid(margins + alpha)
        residual = float(np.sum(probabilities - truths))
        if residual > 0:
            high = alpha
        else:
            low = alpha
        if abs(residual) <= CLUSTER_TOLERANCE:
            return InterceptFit(
                alpha=float(alpha),
                n_rows=int(margins.size),
                n_positive=n_positive,
                converged=True,
                iterations=iterations,
                residual=residual,
                degenerate=False,
            )
        derivative = float(np.sum(probabilities * (1.0 - probabilities)))
        step = alpha - residual / derivative if derivative > 0 else alpha
        alpha = step if low < step < high else 0.5 * (low + high)

    return InterceptFit(
        alpha=float(alpha),
        n_rows=int(margins.size),
        n_positive=n_positive,
        converged=False,
        iterations=iterations,
        residual=_score_equation(margins, truths, alpha),
        degenerate=False,
    )


def fit_family_intercepts(
    frame: pd.DataFrame, margin_column: str = "margin"
) -> dict[str, InterceptFit]:
    """Fit one intercept per predicate family on the calibration partition."""
    fits: dict[str, InterceptFit] = {}
    for family in FAMILIES:
        subset = frame[frame["family"] == family]
        fits[family] = fit_intercept(
            subset[margin_column].to_numpy(), subset["formal_truth"].to_numpy()
        )
    return fits


def apply_intercept(margins: np.ndarray, alpha: float) -> np.ndarray:
    """Shift margins by the fitted location parameter."""
    return np.asarray(margins, dtype=float) + float(alpha)


def apply_family_intercepts(
    frame: pd.DataFrame, alphas: dict[str, float], margin_column: str = "margin"
) -> np.ndarray:
    """Shift each row by its family's location parameter."""
    offsets = frame["family"].map(alphas).to_numpy(dtype=float)
    return frame[margin_column].to_numpy(dtype=float) + offsets


# --------------------------------------------------------------------------
# Shape metrics -- computed on margins, invariant to any intercept
# --------------------------------------------------------------------------


def _as_margin_curve(margins: Sequence[float]) -> np.ndarray:
    curve = np.asarray(margins, dtype=float)
    if curve.shape != (N_STRICTNESS,):
        raise ValueError(f"expected {N_STRICTNESS} margins, got shape {curve.shape}")
    if not np.all(np.isfinite(curve)):
        raise ValueError("margin curve contains non-finite values")
    return curve


def margin_monotonicity_violation_rate(margins: Sequence[float]) -> float:
    """Fraction of the eight adjacent steps on which the margin rises."""
    curve = _as_margin_curve(margins)
    return float(np.mean(curve[1:] > curve[:-1]))


def margin_reversal_magnitude(margins: Sequence[float]) -> float:
    """Mean positive part of the eight adjacent margin steps."""
    curve = _as_margin_curve(margins)
    return float(np.mean(np.maximum(0.0, np.diff(curve))))


def response_span(margins: Sequence[float]) -> float:
    """Total response span ``M(0) - M(8)``; positive when the judge responds."""
    curve = _as_margin_curve(margins)
    return float(curve[0] - curve[-1])


def margin_spearman(margins: Sequence[float]) -> float:
    """Within-artifact Spearman association between strictness and margin."""
    curve = _as_margin_curve(margins)
    if np.all(curve == curve[0]):
        return float("nan")
    return float(stats.spearmanr(np.arange(N_STRICTNESS), curve).statistic)


# --------------------------------------------------------------------------
# Localisation
# --------------------------------------------------------------------------


def predicted_first_fail_from_margin(
    margins: Sequence[float], alpha: float = 0.0
) -> int:
    """First strictness at which the shifted margin implies P(met) < 0.5.

    ``sigma(M + alpha) < 0.5`` exactly when ``M + alpha < 0``, so the decision
    boundary in margin space is ``-alpha``. The 0.5 probability boundary is the
    preregistered one and is never moved.
    """
    curve = _as_margin_curve(margins) + float(alpha)
    below = np.flatnonzero(curve < 0.0)
    return int(below[0]) if below.size else NO_CROSSING_INDEX


def threshold_crossing_error_from_margin(
    margins: Sequence[float], true_first_fail_index: int, alpha: float = 0.0
) -> int:
    """``|j_hat - j*|`` for a margin curve under an intercept shift."""
    predicted = predicted_first_fail_from_margin(margins, alpha)
    return int(abs(predicted - int(true_first_fail_index)))


# --------------------------------------------------------------------------
# Per-artifact assembly
# --------------------------------------------------------------------------

#: Descriptive taxonomy predeclared in the Day-2 preregistration.
TAXONOMY_COLUMNS: tuple[str, ...] = (
    "mislocated_but_orderly",
    "locally_unstable",
    "correctly_localized",
    "correctable_by_global_bias",
    "residual_localization_failure",
)


def compute_artifact_metrics(
    scores: pd.DataFrame,
    alpha_global: float,
    alphas_family: dict[str, float],
    margin_column: str = "margin",
) -> pd.DataFrame:
    """Collapse per-threshold TEST scores into one row per artifact.

    Produces raw, globally corrected and family-corrected localisation together
    with the intercept-invariant shape metrics.
    """
    required = {
        "model_id",
        "artifact_id",
        "family",
        "latent_level",
        "strictness_index",
        margin_column,
        "formal_truth",
        "true_first_fail_index",
    }
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"scores frame is missing columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    ordered = scores.sort_values(["model_id", "artifact_id", "strictness_index"])
    for (model_id, artifact_id), group in ordered.groupby(
        ["model_id", "artifact_id"], sort=True
    ):
        levels = group["strictness_index"].to_numpy()
        if not np.array_equal(levels, np.arange(N_STRICTNESS)):
            raise ValueError(
                f"{model_id}/{artifact_id}: expected strictness 0..8 exactly once"
            )
        margins = group[margin_column].to_numpy(dtype=float)
        truth = group["formal_truth"].to_numpy(dtype=int)
        family = group["family"].iloc[0]
        j_star = int(group["true_first_fail_index"].iloc[0])
        alpha_f = float(alphas_family[family])

        tce_raw = threshold_crossing_error_from_margin(margins, j_star, 0.0)
        tce_global = threshold_crossing_error_from_margin(margins, j_star, alpha_global)
        tce_family = threshold_crossing_error_from_margin(margins, j_star, alpha_f)
        mmvr = margin_monotonicity_violation_rate(margins)

        p_raw = sigmoid(margins)
        p_global = sigmoid(margins + alpha_global)
        p_family = sigmoid(margins + alpha_f)

        rows.append(
            {
                "model_id": model_id,
                "artifact_id": artifact_id,
                "family": family,
                "latent_level": int(group["latent_level"].iloc[0]),
                "true_first_fail_index": j_star,
                # localisation
                "j_hat_raw": predicted_first_fail_from_margin(margins, 0.0),
                "j_hat_global": predicted_first_fail_from_margin(margins, alpha_global),
                "j_hat_family": predicted_first_fail_from_margin(margins, alpha_f),
                "tce_raw": tce_raw,
                "tce_global": tce_global,
                "tce_family": tce_family,
                "delta_tce_global": tce_raw - tce_global,
                "delta_tce_family": tce_global - tce_family,
                # shape, invariant to any intercept
                "mmvr": mmvr,
                "mmvm": margin_reversal_magnitude(margins),
                "span": response_span(margins),
                "margin_spearman": margin_spearman(margins),
                # pointwise performance
                "accuracy_raw": float(np.mean((p_raw >= DECISION_THRESHOLD) == truth)),
                "accuracy_global": float(
                    np.mean((p_global >= DECISION_THRESHOLD) == truth)
                ),
                "accuracy_family": float(
                    np.mean((p_family >= DECISION_THRESHOLD) == truth)
                ),
                "brier_raw": float(np.mean((p_raw - truth) ** 2)),
                "brier_global": float(np.mean((p_global - truth) ** 2)),
                "brier_family": float(np.mean((p_family - truth) ** 2)),
                # failure-mode descriptors
                "never_crosses_raw": bool(np.all(p_raw >= DECISION_THRESHOLD)),
                "never_crosses_global": bool(np.all(p_global >= DECISION_THRESHOLD)),
                "always_below_raw": bool(p_raw[0] < DECISION_THRESHOLD),
                "always_below_global": bool(p_global[0] < DECISION_THRESHOLD),
                # predeclared taxonomy
                "mislocated_but_orderly": bool(mmvr == 0.0 and tce_raw > 1),
                "locally_unstable": bool(mmvr > 0.0),
                "correctly_localized": bool(tce_raw == 0),
                "correctable_by_global_bias": bool(tce_raw > 1 and tce_global <= 1),
                "residual_localization_failure": bool(tce_global > 1),
            }
        )
    return pd.DataFrame(rows)


def boundary_aligned_curves(
    scores: pd.DataFrame, alpha_global: float, margin_column: str = "margin"
) -> pd.DataFrame:
    """Mean raw and globally corrected P(met) by signed distance from ``j*``."""
    frame = scores.copy()
    frame["signed_distance"] = (
        frame["strictness_index"] - frame["true_first_fail_index"]
    )
    margins = frame[margin_column].to_numpy(dtype=float)
    frame["p_raw"] = sigmoid(margins)
    frame["p_global"] = sigmoid(margins + alpha_global)

    grouped = frame.groupby(["model_id", "signed_distance"], sort=True)
    summary = grouped.agg(
        mean_p_raw=("p_raw", "mean"),
        sd_p_raw=("p_raw", "std"),
        mean_p_global=("p_global", "mean"),
        sd_p_global=("p_global", "std"),
        n=("p_raw", "size"),
    ).reset_index()
    summary["se_p_raw"] = summary["sd_p_raw"] / np.sqrt(summary["n"])
    summary["se_p_global"] = summary["sd_p_global"] / np.sqrt(summary["n"])
    return summary


def shape_invariance_report(
    scores: pd.DataFrame,
    alphas: Sequence[float],
    margin_column: str = "margin",
) -> dict[str, Any]:
    """Empirically confirm that intercept shifts leave every shape metric alone.

    H3 is a mathematical identity, so any material deviation here is an
    implementation defect rather than a finding.
    """
    ordered = scores.sort_values(["model_id", "artifact_id", "strictness_index"])
    deviations = {"mmvr": 0.0, "mmvm": 0.0, "span": 0.0, "spearman": 0.0}

    for _, group in ordered.groupby(["model_id", "artifact_id"], sort=True):
        margins = group[margin_column].to_numpy(dtype=float)
        base = (
            margin_monotonicity_violation_rate(margins),
            margin_reversal_magnitude(margins),
            response_span(margins),
            margin_spearman(margins),
        )
        for alpha in alphas:
            shifted = margins + float(alpha)
            values = (
                margin_monotonicity_violation_rate(shifted),
                margin_reversal_magnitude(shifted),
                response_span(shifted),
                margin_spearman(shifted),
            )
            for key, a, b in zip(deviations, base, values):
                if np.isnan(a) and np.isnan(b):
                    continue
                deviations[key] = max(deviations[key], abs(float(a) - float(b)))

    return {
        "alphas_tested": [float(a) for a in alphas],
        "max_abs_deviation": deviations,
        "mmvr_unchanged": deviations["mmvr"] == 0.0,
        "mmvm_unchanged": deviations["mmvm"] <= 1e-9,
        "span_unchanged": deviations["span"] <= 1e-9,
        "spearman_unchanged": deviations["spearman"] <= 1e-12,
    }
