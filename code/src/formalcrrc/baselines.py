"""Post-study analytic reference baselines for TCE and row-level accuracy.

Added **after** the three preregistered FormalCRRC studies were completed. These
are structural reference strategies, not preregistered endpoints, not learned or
calibrated models, and not competitive judges. They exist so a reader can answer
one question:

    How large is an observed TCE relative to strategies that use no
    artifact-specific information at all?

Because the FormalCRRC true first-fail boundary is exactly uniform over
``j* in 1..9`` by construction, every strategy here has a closed-form expectation
and needs no simulation. All quantities are computed as exact
:class:`fractions.Fraction` values and only converted to float for display.

The step identity
-----------------
A *step predictor* with crossing ``j_hat`` answers "met" at threshold ``s`` iff
``s < j_hat``. The formal label is "met" iff ``s < j*``. The two therefore
disagree on exactly the levels in ``[min(j_hat, j*), max(j_hat, j*))``, so

    row errors = |j_hat - j*|            accuracy = 1 - |j_hat - j*| / 9

This identity holds **only** for single-step predictors. A judge whose
thresholded decisions are not a single step -- which is the normal case, since
nothing forces a judge's probability curve to be monotone -- can and does depart
from it, which is why a judge's accuracy is not recoverable from its TCE.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Mapping, Sequence

import numpy as np

from formalcrrc.config import N_STRICTNESS, NO_CROSSING_INDEX

#: The support of the formal first-fail boundary in the FormalCRRC design.
#: ``j* = 0`` cannot occur: strictness ``s = 0`` is satisfied by every artifact
#: in every predicate family by construction.
BOUNDARY_SUPPORT: tuple[int, ...] = tuple(range(1, NO_CROSSING_INDEX + 1))

#: Crossings a predictor may emit: 0 (never met) through 9 (always met).
PREDICTION_SUPPORT: tuple[int, ...] = tuple(range(0, NO_CROSSING_INDEX + 1))


# --------------------------------------------------------------------------
# The step identity
# --------------------------------------------------------------------------


def step_predictions(j_hat: int, n_thresholds: int = N_STRICTNESS) -> np.ndarray:
    """Labels a step predictor with crossing ``j_hat`` emits, over ``s = 0..8``.

    Returns 1 for "criterion met". ``j_hat = 0`` never says met; ``j_hat = 9``
    always does.
    """
    return (np.arange(n_thresholds) < int(j_hat)).astype(int)


def row_errors(j_hat: int, j_star: int, n_thresholds: int = N_STRICTNESS) -> int:
    """Row-level disagreements between a step predictor and the formal labels."""
    predicted = step_predictions(j_hat, n_thresholds)
    truth = step_predictions(j_star, n_thresholds)
    return int(np.sum(predicted != truth))


def crossing_error(j_hat: int, j_star: int) -> int:
    """``TCE`` for a step predictor: ``|j_hat - j*|``."""
    return abs(int(j_hat) - int(j_star))


# --------------------------------------------------------------------------
# Exact expectations
# --------------------------------------------------------------------------


def constant_crossing_tce(
    j_hat: int, boundaries: Mapping[int, int] | Sequence[int]
) -> Fraction:
    """Exact mean TCE of always predicting ``j_hat``, over a boundary distribution.

    ``boundaries`` may be a sequence of observed ``j*`` values or a mapping from
    ``j*`` to its count. Computed exactly; nothing is sampled.
    """
    counts = _as_counts(boundaries)
    total = sum(counts.values())
    if total == 0:
        raise ValueError("boundary distribution is empty")
    return Fraction(
        sum(count * abs(int(j_hat) - j) for j, count in counts.items()), total
    )


def uniform_random_crossing_tce(
    boundaries: Mapping[int, int] | Sequence[int],
    prediction_support: Iterable[int] = BOUNDARY_SUPPORT,
) -> Fraction:
    """Exact expected TCE of drawing ``j_hat`` uniformly, independent of the artifact.

    Averages the deterministic constant-crossing result over every value the
    random predictor could take. **No Monte Carlo**: the expectation is exact and
    a sampled estimate would only add noise to a known quantity.
    """
    support = list(prediction_support)
    if not support:
        raise ValueError("prediction support is empty")
    return Fraction(
        sum(constant_crossing_tce(j, boundaries) for j in support), len(support)
    )


def accuracy_from_tce(tce: Fraction, n_thresholds: int = N_STRICTNESS) -> Fraction:
    """``1 - TCE / 9``. Valid for step predictors only -- see the module docstring."""
    return Fraction(1) - Fraction(tce) / n_thresholds


def minimum_tce_constant_crossing(
    boundaries: Mapping[int, int] | Sequence[int],
    prediction_support: Iterable[int] = PREDICTION_SUPPORT,
) -> tuple[int, Fraction]:
    """The constant crossing minimising expected TCE, and its value.

    For a uniform boundary this is the median of the support, ``j_hat = 5``.
    Ties break toward the smaller crossing, deterministically.
    """
    scored = [
        (constant_crossing_tce(j, boundaries), j) for j in prediction_support
    ]
    best_tce, best_j = min(scored)
    return best_j, best_tce


def majority_label(
    boundaries: Mapping[int, int] | Sequence[int], n_thresholds: int = N_STRICTNESS
) -> tuple[int, Fraction]:
    """The majority row label and its share.

    The met-rate is ``E[j*] / 9``; for a uniform boundary that is ``5/9``, so the
    majority label is "met" and the majority-class accuracy is ``5/9``.
    """
    counts = _as_counts(boundaries)
    total = sum(counts.values())
    met_rows = sum(count * min(j, n_thresholds) for j, count in counts.items())
    met_rate = Fraction(met_rows, total * n_thresholds)
    return (1, met_rate) if met_rate >= Fraction(1, 2) else (0, 1 - met_rate)


# --------------------------------------------------------------------------
# The baseline registry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Baseline:
    """One post-study reference strategy."""

    key: str
    name: str
    description: str
    #: Fixed crossing, or ``None`` for the randomised strategy.
    j_hat: int | None
    uses_artifact_input: bool = False

    def tce(self, boundaries: Mapping[int, int] | Sequence[int]) -> Fraction:
        if self.j_hat is None:
            return uniform_random_crossing_tce(boundaries)
        return constant_crossing_tce(self.j_hat, boundaries)

    def accuracy(self, boundaries: Mapping[int, int] | Sequence[int]) -> Fraction:
        return accuracy_from_tce(self.tce(boundaries))


BASELINES: tuple[Baseline, ...] = (
    Baseline(
        key="fixed_center",
        name="Fixed center / no input",
        description=(
            "Always predict the first-fail boundary at j_hat = 5. Design-aware: "
            "it exploits the known uniform boundary distribution but never "
            "examines the artifact."
        ),
        j_hat=5,
    ),
    Baseline(
        key="uniform_random_crossing",
        name="Uniform random crossing",
        description=(
            "Draw j_hat uniformly from 1..9, independent of the artifact and of "
            "the true boundary. Reported as an exact expectation, never sampled."
        ),
        j_hat=None,
    ),
    Baseline(
        key="always_met",
        name="Always met / majority class",
        description=(
            "Always answer 'criterion met', i.e. j_hat = 9. Since the balanced "
            "threshold grid holds more met rows than not-met rows, this is also "
            "the majority-class accuracy baseline."
        ),
        j_hat=NO_CROSSING_INDEX,
    ),
    Baseline(
        key="always_not_met",
        name="Always not met",
        description="Always answer 'criterion not met', i.e. j_hat = 0.",
        j_hat=0,
    ),
)

BASELINES_BY_KEY: dict[str, Baseline] = {b.key: b for b in BASELINES}


def analytic_table(
    boundaries: Mapping[int, int] | Sequence[int],
) -> list[dict[str, object]]:
    """Every baseline's exact TCE and accuracy for a given boundary distribution."""
    rows: list[dict[str, object]] = []
    for baseline in BASELINES:
        tce = baseline.tce(boundaries)
        accuracy = accuracy_from_tce(tce)
        rows.append(
            {
                "key": baseline.key,
                "name": baseline.name,
                "description": baseline.description,
                "j_hat": baseline.j_hat,
                "uses_artifact_input": baseline.uses_artifact_input,
                "tce_exact": str(tce),
                "tce": float(tce),
                "accuracy_exact": str(accuracy),
                "accuracy": float(accuracy),
                "is_expectation": baseline.j_hat is None,
            }
        )
    return rows


# --------------------------------------------------------------------------
# Empirical evaluation against frozen labels
# --------------------------------------------------------------------------


def empirical_baseline(
    j_hat: int, truth_block: np.ndarray, boundaries: np.ndarray
) -> dict[str, float]:
    """Apply a constant-crossing predictor to real frozen labels.

    ``truth_block`` is ``(n_artifacts, 9)`` of formal labels and ``boundaries``
    the matching ``j*`` values. Read-only with respect to the dataset.
    """
    truth_block = np.asarray(truth_block, dtype=int)
    boundaries = np.asarray(boundaries, dtype=int)
    predictions = step_predictions(j_hat)[None, :]
    errors = np.abs(int(j_hat) - boundaries)
    return {
        "j_hat": int(j_hat),
        "mean_tce": float(errors.mean()),
        "accuracy": float(np.mean(predictions == truth_block)),
        "row_errors_total": int(np.sum(predictions != truth_block)),
        "crossing_errors_total": int(errors.sum()),
        "n_artifacts": int(truth_block.shape[0]),
        "n_rows": int(truth_block.size),
    }


def empirical_random_crossing(
    truth_block: np.ndarray,
    boundaries: np.ndarray,
    prediction_support: Iterable[int] = BOUNDARY_SUPPORT,
) -> dict[str, float]:
    """Exact expectation of the random-crossing baseline on frozen labels.

    Averages the deterministic result over every value the predictor could take.
    No sampling: the expectation is exact.
    """
    support = list(prediction_support)
    results = [empirical_baseline(j, truth_block, boundaries) for j in support]
    return {
        "j_hat": None,
        "mean_tce": float(np.mean([r["mean_tce"] for r in results])),
        "accuracy": float(np.mean([r["accuracy"] for r in results])),
        "n_artifacts": results[0]["n_artifacts"],
        "n_rows": results[0]["n_rows"],
        "averaged_over": support,
        "method": "exact expectation over the prediction support; not sampled",
    }


def _as_counts(
    boundaries: Mapping[int, int] | Sequence[int],
) -> dict[int, int]:
    if isinstance(boundaries, Mapping):
        return {int(k): int(v) for k, v in boundaries.items()}
    counts: dict[int, int] = {}
    for value in boundaries:
        counts[int(value)] = counts.get(int(value), 0) + 1
    return counts
