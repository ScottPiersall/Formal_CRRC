"""Hand-checked metric behaviour on constructed toy curves.

Every expected value below is worked out by hand from the preregistered
definitions, not read off an implementation.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from formalcrrc import metrics
from formalcrrc.config import (
    DECISION_THRESHOLD,
    MVR_TOLERANCE,
    N_STRICTNESS,
    NO_CROSSING_INDEX,
)

# A perfectly behaved curve that crosses 0.5 between s = 3 and s = 4.
PERFECT = [0.99, 0.97, 0.95, 0.90, 0.10, 0.05, 0.03, 0.02, 0.01]

# Identical to PERFECT except for one +0.02 rise at s = 5 -> 6 (within tolerance).
SMALL_VIOLATION = [0.99, 0.97, 0.95, 0.90, 0.10, 0.05, 0.07, 0.02, 0.01]

# One +0.20 rise at s = 5 -> 6 (beyond tolerance).
LARGE_VIOLATION = [0.99, 0.97, 0.95, 0.90, 0.10, 0.05, 0.25, 0.02, 0.01]

# Three rises beyond tolerance.
MANY_VIOLATIONS = [0.10, 0.40, 0.20, 0.60, 0.30, 0.80, 0.10, 0.05, 0.01]

# Never drops below 0.5.
NEVER_CROSSES = [0.99] * N_STRICTNESS

# Drops below 0.5 immediately at s = 0.
ALWAYS_BELOW = [0.10] * N_STRICTNESS


class TestMonotonicityViolationRate:
    def test_perfect_curve_has_no_violations(self):
        assert metrics.monotonicity_violation_rate(PERFECT) == 0.0

    def test_small_violation_is_absorbed_by_the_tolerance(self):
        # +0.02 <= 0.05, so the primary MVR sees nothing.
        assert metrics.monotonicity_violation_rate(SMALL_VIOLATION) == 0.0

    def test_small_violation_is_visible_at_zero_tolerance(self):
        assert metrics.monotonicity_violation_rate(
            SMALL_VIOLATION, tolerance=0.0
        ) == pytest.approx(1 / 8)

    def test_single_large_violation(self):
        assert metrics.monotonicity_violation_rate(LARGE_VIOLATION) == pytest.approx(
            1 / 8
        )

    def test_multiple_violations(self):
        # Rises at 0->1 (+0.30), 2->3 (+0.40) and 4->5 (+0.50): three of eight.
        assert metrics.monotonicity_violation_rate(MANY_VIOLATIONS) == pytest.approx(
            3 / 8
        )

    def test_flat_curve_has_no_violations_at_either_tolerance(self):
        assert metrics.monotonicity_violation_rate(NEVER_CROSSES) == 0.0
        assert (
            metrics.monotonicity_violation_rate(NEVER_CROSSES, tolerance=0.0) == 0.0
        )

    def test_a_rise_of_exactly_the_tolerance_is_not_a_violation(self):
        curve = [0.5] * N_STRICTNESS
        curve[4] = 0.5 + MVR_TOLERANCE  # step 3->4 is exactly +0.05
        assert metrics.monotonicity_violation_rate(curve) == 0.0
        # ... but a hair more is.
        curve[4] = 0.5 + MVR_TOLERANCE + 1e-9
        assert metrics.monotonicity_violation_rate(curve) == pytest.approx(1 / 8)

    def test_monotonically_rising_curve(self):
        rising = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0]
        assert metrics.monotonicity_violation_rate(rising) == pytest.approx(5 / 8)

    def test_every_step_can_violate(self):
        rising = [i / 8 for i in range(N_STRICTNESS)]
        assert metrics.monotonicity_violation_rate(rising) == 1.0


class TestMonotonicityViolationMagnitude:
    def test_perfect_curve_has_zero_magnitude(self):
        assert metrics.monotonicity_violation_magnitude(PERFECT) == 0.0

    def test_small_violation_still_registers_magnitude(self):
        # MVM has no tolerance: +0.02 spread over eight steps.
        assert metrics.monotonicity_violation_magnitude(
            SMALL_VIOLATION
        ) == pytest.approx(0.02 / 8)

    def test_large_violation_magnitude(self):
        assert metrics.monotonicity_violation_magnitude(
            LARGE_VIOLATION
        ) == pytest.approx(0.20 / 8)

    def test_multiple_violation_magnitude(self):
        # +0.30, +0.40 and +0.50 summed over eight steps.
        assert metrics.monotonicity_violation_magnitude(
            MANY_VIOLATIONS
        ) == pytest.approx((0.30 + 0.40 + 0.50) / 8)

    def test_magnitude_ignores_downward_steps(self):
        assert metrics.monotonicity_violation_magnitude(ALWAYS_BELOW) == 0.0


class TestThresholdCrossing:
    def test_exact_crossing(self):
        assert metrics.predicted_first_fail_index(PERFECT) == 4
        assert metrics.threshold_crossing_error(PERFECT, 4) == 0

    def test_crossing_one_level_early(self):
        early = [0.99, 0.97, 0.95, 0.20, 0.10, 0.05, 0.03, 0.02, 0.01]
        assert metrics.predicted_first_fail_index(early) == 3
        assert metrics.threshold_crossing_error(early, 4) == 1

    def test_crossing_one_level_late(self):
        late = [0.99, 0.97, 0.95, 0.90, 0.80, 0.05, 0.03, 0.02, 0.01]
        assert metrics.predicted_first_fail_index(late) == 5
        assert metrics.threshold_crossing_error(late, 4) == 1

    def test_no_predicted_crossing(self):
        assert metrics.predicted_first_fail_index(NEVER_CROSSES) == NO_CROSSING_INDEX
        assert metrics.threshold_crossing_error(NEVER_CROSSES, 4) == 5

    def test_no_crossing_against_a_true_no_crossing_is_exact(self):
        assert (
            metrics.threshold_crossing_error(NEVER_CROSSES, NO_CROSSING_INDEX) == 0
        )

    def test_immediate_crossing(self):
        assert metrics.predicted_first_fail_index(ALWAYS_BELOW) == 0
        assert metrics.threshold_crossing_error(ALWAYS_BELOW, 4) == 4

    def test_probability_exactly_at_the_boundary_counts_as_met(self):
        curve = [DECISION_THRESHOLD] * N_STRICTNESS
        assert metrics.predicted_first_fail_index(curve) == NO_CROSSING_INDEX

    def test_first_crossing_wins_even_if_the_curve_recovers(self):
        wobbly = [0.99, 0.40, 0.95, 0.90, 0.10, 0.05, 0.03, 0.02, 0.01]
        assert metrics.predicted_first_fail_index(wobbly) == 1


class TestAccuracyAndBrier:
    TRUTH = [1, 1, 1, 1, 0, 0, 0, 0, 0]

    def test_perfect_curve_is_perfectly_accurate(self):
        assert metrics.accuracy(PERFECT, self.TRUTH) == 1.0

    def test_accuracy_counts_thresholded_agreements(self):
        curve = [0.99, 0.97, 0.95, 0.20, 0.10, 0.05, 0.03, 0.02, 0.01]
        assert metrics.accuracy(curve, self.TRUTH) == pytest.approx(8 / 9)

    def test_brier_of_a_certain_correct_curve_is_zero(self):
        assert metrics.brier_score(
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], self.TRUTH
        ) == pytest.approx(0.0)

    def test_brier_of_a_certain_wrong_curve_is_one(self):
        assert metrics.brier_score(
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0], self.TRUTH
        ) == pytest.approx(1.0)

    def test_brier_of_an_uninformative_curve(self):
        assert metrics.brier_score([0.5] * N_STRICTNESS, self.TRUTH) == pytest.approx(
            0.25
        )


class TestSpearman:
    def test_strictly_decreasing_curve_gives_minus_one(self):
        assert metrics.spearman_strictness(PERFECT) == pytest.approx(-1.0)

    def test_strictly_increasing_curve_gives_plus_one(self):
        rising = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        assert metrics.spearman_strictness(rising) == pytest.approx(1.0)

    def test_constant_curve_is_undefined(self):
        assert math.isnan(metrics.spearman_strictness(NEVER_CROSSES))


class TestInputValidation:
    def test_wrong_length_is_rejected(self):
        with pytest.raises(ValueError):
            metrics.monotonicity_violation_rate([0.5, 0.5])

    def test_non_finite_probability_is_rejected(self):
        curve = list(PERFECT)
        curve[2] = float("nan")
        with pytest.raises(ValueError):
            metrics.monotonicity_violation_rate(curve)


def _scores_frame(model_id: str, curves: dict[str, list[float]]) -> pd.DataFrame:
    rows = []
    for artifact_id, curve in curves.items():
        j_star = 4
        for s, p in enumerate(curve):
            rows.append(
                {
                    "model_id": model_id,
                    "artifact_id": artifact_id,
                    "family": "coverage",
                    "latent_level": 3,
                    "strictness_index": s,
                    "p_met": p,
                    "formal_truth": int(s < j_star),
                    "true_first_fail_index": j_star,
                }
            )
    return pd.DataFrame(rows)


class TestFrameAssembly:
    def test_one_row_per_model_artifact(self):
        frame = _scores_frame("m", {"a": PERFECT, "b": LARGE_VIOLATION})
        result = metrics.compute_artifact_metrics(frame)
        assert len(result) == 2
        assert set(result["artifact_id"]) == {"a", "b"}

    def test_values_match_the_curve_level_functions(self):
        frame = _scores_frame("m", {"a": LARGE_VIOLATION})
        row = metrics.compute_artifact_metrics(frame).iloc[0]
        assert row["mvr"] == pytest.approx(1 / 8)
        assert row["mvm"] == pytest.approx(0.20 / 8)
        assert row["tce"] == 0
        assert row["predicted_first_fail_index"] == 4

    def test_missing_threshold_is_rejected(self):
        frame = _scores_frame("m", {"a": PERFECT})
        frame = frame[frame["strictness_index"] != 5]
        with pytest.raises(ValueError):
            metrics.compute_artifact_metrics(frame)

    def test_duplicated_threshold_is_rejected(self):
        frame = _scores_frame("m", {"a": PERFECT})
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        with pytest.raises(ValueError):
            metrics.compute_artifact_metrics(frame)

    def test_boundary_alignment_uses_signed_distance(self):
        frame = _scores_frame("m", {"a": PERFECT})
        aligned = metrics.boundary_aligned_curve(frame)
        assert set(aligned["signed_distance"]) == set(range(-4, 5))
        at_zero = aligned[aligned["signed_distance"] == 0]["mean_p_met"].iloc[0]
        assert at_zero == pytest.approx(PERFECT[4])

    def test_boundary_alignment_averages_across_artifacts(self):
        frame = _scores_frame("m", {"a": PERFECT, "b": PERFECT})
        aligned = metrics.boundary_aligned_curve(frame)
        assert set(aligned["n"]) == {2}
        assert np.allclose(
            aligned.sort_values("signed_distance")["mean_p_met"].to_numpy(),
            np.asarray(PERFECT),
        )
