"""Margins, intercept fitting, shape invariance and localisation.

The invariance tests are the heart of Day 2: the whole design rests on the claim
that adding a constant in margin space moves a curve without deforming it.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from formalcrrc import day2, scoring
from formalcrrc.config import N_STRICTNESS, NO_CROSSING_INDEX


class TestMarginConstruction:
    def test_margin_is_met_minus_not_met(self):
        assert day2.decision_margin(np.array([3.0]), np.array([1.0]))[0] == 2.0

    def test_orientation_favours_met_when_positive(self):
        assert day2.sigmoid(day2.decision_margin(np.array([2.0]), np.array([0.0]))) > 0.5
        assert day2.sigmoid(day2.decision_margin(np.array([0.0]), np.array([2.0]))) < 0.5

    def test_equal_scores_give_a_zero_margin(self):
        assert day2.decision_margin(np.array([7.5]), np.array([7.5]))[0] == 0.0

    def test_hand_computed_margins(self):
        met = np.array([37.0, 10.9375, 21.5, 15.5625])
        not_met = np.array([13.75, 6.15625, 15.3125, 4.03125])
        expected = np.array([23.25, 4.78125, 6.1875, 11.53125])
        assert np.allclose(day2.decision_margin(met, not_met), expected)

    def test_margin_column_added_to_a_frame(self):
        frame = pd.DataFrame(
            {"raw_score_met": [2.0, -1.0], "raw_score_not_met": [1.0, 1.0]}
        )
        assert day2.add_margin_column(frame)["margin"].tolist() == [1.0, -2.0]


class TestSigmoidRelation:
    def test_matches_the_day1_two_label_normalisation(self):
        """sigma(M) must equal the Day-1 normalised probability exactly."""
        for met, not_met in [
            (3.0, 1.0),
            (-2.5, 4.0),
            (0.0, 0.0),
            (37.0, 13.75),
            (-40.0, 12.0),
        ]:
            margin = day2.decision_margin(np.array([met]), np.array([not_met]))
            assert float(day2.sigmoid(margin)[0]) == pytest.approx(
                scoring.normalized_probability(met, not_met), abs=1e-12
            )

    def test_stable_at_extremes(self):
        assert float(day2.sigmoid(np.array([1000.0]))[0]) == pytest.approx(1.0)
        assert float(day2.sigmoid(np.array([-1000.0]))[0]) == pytest.approx(0.0)

    def test_half_at_zero(self):
        assert float(day2.sigmoid(np.array([0.0]))[0]) == 0.5


class TestInterceptFitting:
    @staticmethod
    def _sample(shift: float, n: int = 4000, seed: int = 0):
        rng = np.random.default_rng(seed)
        margins = rng.normal(0.0, 2.0, n)
        truths = (rng.random(n) < day2.sigmoid(margins)).astype(int)
        return margins + shift, truths

    def test_unbiased_margins_give_alpha_near_zero(self):
        margins, truths = self._sample(0.0)
        fit = day2.fit_intercept(margins, truths)
        assert fit.converged
        assert abs(fit.alpha) < 0.1

    def test_pessimistic_shift_is_corrected_upward(self):
        """Margins shifted down need a positive alpha to come back."""
        margins, truths = self._sample(-3.0)
        fit = day2.fit_intercept(margins, truths)
        assert fit.converged
        assert fit.alpha == pytest.approx(3.0, abs=0.15)

    def test_permissive_shift_is_corrected_downward(self):
        margins, truths = self._sample(+3.0)
        fit = day2.fit_intercept(margins, truths)
        assert fit.converged
        assert fit.alpha == pytest.approx(-3.0, abs=0.15)

    def test_solution_satisfies_the_score_equation(self):
        margins, truths = self._sample(-2.0)
        fit = day2.fit_intercept(margins, truths)
        residual = np.sum(day2.sigmoid(margins + fit.alpha) - truths)
        assert abs(residual) < 1e-8

    def test_objective_is_minimised_at_the_fit(self):
        margins, truths = self._sample(1.5)
        fit = day2.fit_intercept(margins, truths)

        def loss(alpha: float) -> float:
            p = np.clip(day2.sigmoid(margins + alpha), 1e-12, 1 - 1e-12)
            return float(-np.sum(truths * np.log(p) + (1 - truths) * np.log(1 - p)))

        best = loss(fit.alpha)
        assert best <= loss(fit.alpha - 0.05)
        assert best <= loss(fit.alpha + 0.05)

    def test_exactly_one_free_parameter(self):
        margins, truths = self._sample(-1.0)
        fit = day2.fit_intercept(margins, truths)
        assert isinstance(fit.alpha, float)
        assert fit.to_dict()["method"] == "CLUSTER_with_bisection_safeguard"

    def test_all_positive_labels_are_reported_degenerate(self):
        margins = np.array([0.5, -0.5, 1.0])
        fit = day2.fit_intercept(margins, np.ones(3, dtype=int))
        assert fit.degenerate
        assert math.isnan(fit.alpha)

    def test_all_negative_labels_are_reported_degenerate(self):
        margins = np.array([0.5, -0.5, 1.0])
        fit = day2.fit_intercept(margins, np.zeros(3, dtype=int))
        assert fit.degenerate

    def test_empty_sample_is_rejected(self):
        with pytest.raises(ValueError):
            day2.fit_intercept(np.array([]), np.array([]))

    def test_mismatched_shapes_are_rejected(self):
        with pytest.raises(ValueError):
            day2.fit_intercept(np.zeros(3), np.zeros(4))

    def test_fitting_is_deterministic(self):
        margins, truths = self._sample(-1.25)
        assert (
            day2.fit_intercept(margins, truths).alpha
            == day2.fit_intercept(margins, truths).alpha
        )


class TestShapeInvariance:
    """M -> M + alpha may move a curve but must never deform it."""

    CURVES = [
        np.array([4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0]),
        np.array([1.0, 1.5, 0.5, 0.9, -0.2, 0.3, -1.0, -1.2, -3.0]),
        np.array([0.0] * N_STRICTNESS),
        np.array([-5.0, -5.0, -4.9, -6.0, -6.1, -6.2, -6.3, -6.4, -6.5]),
        np.array([12.0, 11.0, 10.5, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0]),
    ]
    ALPHAS = [-25.0, -3.7, -1e-6, 0.0, 1e-6, 2.5, 40.0]

    @pytest.mark.parametrize("curve", CURVES)
    @pytest.mark.parametrize("alpha", ALPHAS)
    def test_mmvr_unchanged(self, curve, alpha):
        assert day2.margin_monotonicity_violation_rate(
            curve + alpha
        ) == day2.margin_monotonicity_violation_rate(curve)

    @pytest.mark.parametrize("curve", CURVES)
    @pytest.mark.parametrize("alpha", ALPHAS)
    def test_mmvm_unchanged(self, curve, alpha):
        assert day2.margin_reversal_magnitude(curve + alpha) == pytest.approx(
            day2.margin_reversal_magnitude(curve), abs=1e-9
        )

    @pytest.mark.parametrize("curve", CURVES)
    @pytest.mark.parametrize("alpha", ALPHAS)
    def test_span_unchanged(self, curve, alpha):
        assert day2.response_span(curve + alpha) == pytest.approx(
            day2.response_span(curve), abs=1e-9
        )

    @pytest.mark.parametrize("curve", CURVES)
    @pytest.mark.parametrize("alpha", ALPHAS)
    def test_spearman_unchanged(self, curve, alpha):
        base = day2.margin_spearman(curve)
        shifted = day2.margin_spearman(curve + alpha)
        if math.isnan(base):
            assert math.isnan(shifted)
        else:
            assert shifted == pytest.approx(base, abs=1e-12)

    def test_adjacent_differences_are_preserved_exactly(self):
        curve = self.CURVES[1]
        for alpha in self.ALPHAS:
            assert np.allclose(np.diff(curve + alpha), np.diff(curve), atol=1e-12)

    def test_an_intercept_cannot_remove_a_reversal(self):
        wobbly = np.array([1.0, 2.0, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -1.0])
        assert day2.margin_monotonicity_violation_rate(wobbly) > 0
        for alpha in np.linspace(-50, 50, 41):
            assert day2.margin_monotonicity_violation_rate(wobbly + alpha) > 0


class TestShapeMetricValues:
    def test_mmvr_counts_rising_steps(self):
        curve = np.array([1.0, 2.0, 0.0, 1.0, -1.0, -2.0, -3.0, -4.0, -5.0])
        assert day2.margin_monotonicity_violation_rate(curve) == pytest.approx(2 / 8)

    def test_mmvr_zero_for_a_decreasing_curve(self):
        assert day2.margin_monotonicity_violation_rate(np.arange(8, -1, -1.0)) == 0.0

    def test_mmvr_uses_strict_inequality(self):
        assert day2.margin_monotonicity_violation_rate(np.zeros(N_STRICTNESS)) == 0.0

    def test_mmvm_sums_positive_parts(self):
        curve = np.array([0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert day2.margin_reversal_magnitude(curve) == pytest.approx(3.0 / 8)

    def test_span_is_first_minus_last(self):
        assert day2.response_span(np.arange(8, -1, -1.0)) == pytest.approx(8.0)

    def test_span_negative_when_the_judge_runs_backwards(self):
        assert day2.response_span(np.arange(0, 9, 1.0)) == pytest.approx(-8.0)

    def test_spearman_undefined_for_a_flat_curve(self):
        assert math.isnan(day2.margin_spearman(np.zeros(N_STRICTNESS)))

    def test_wrong_length_rejected(self):
        with pytest.raises(ValueError):
            day2.margin_monotonicity_violation_rate(np.zeros(3))


class TestLocalisation:
    def test_crossing_is_where_the_margin_turns_negative(self):
        curve = np.array([3.0, 2.0, 1.0, 0.5, -0.5, -1.0, -2.0, -3.0, -4.0])
        assert day2.predicted_first_fail_from_margin(curve) == 4

    def test_never_crossing(self):
        curve = np.full(N_STRICTNESS, 2.0)
        assert day2.predicted_first_fail_from_margin(curve) == NO_CROSSING_INDEX

    def test_below_from_the_first_level(self):
        curve = np.full(N_STRICTNESS, -2.0)
        assert day2.predicted_first_fail_from_margin(curve) == 0

    def test_exact_crossing_gives_zero_error(self):
        curve = np.array([3.0, 2.0, 1.0, 0.5, -0.5, -1.0, -2.0, -3.0, -4.0])
        assert day2.threshold_crossing_error_from_margin(curve, 4) == 0

    def test_early_and_late_crossing(self):
        early = np.array([3.0, 2.0, 1.0, -0.5, -1.0, -2.0, -3.0, -4.0, -5.0])
        late = np.array([3.0, 2.0, 1.0, 0.5, 0.2, -1.0, -2.0, -3.0, -4.0])
        assert day2.threshold_crossing_error_from_margin(early, 4) == 1
        assert day2.threshold_crossing_error_from_margin(late, 4) == 1

    def test_no_crossing_against_true_no_crossing(self):
        curve = np.full(N_STRICTNESS, 2.0)
        assert day2.threshold_crossing_error_from_margin(curve, NO_CROSSING_INDEX) == 0

    def test_zero_margin_counts_as_met(self):
        """sigma(0) = 0.5, and the boundary is 'met' at exactly 0.5."""
        assert day2.predicted_first_fail_from_margin(np.zeros(N_STRICTNESS)) == (
            NO_CROSSING_INDEX
        )

    def test_intercept_moves_the_boundary_not_the_curve(self):
        # Unit steps, so a unit intercept moves the crossing by exactly one level.
        curve = np.array([4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0])
        assert day2.predicted_first_fail_from_margin(curve, alpha=0.0) == 5
        assert day2.predicted_first_fail_from_margin(curve, alpha=1.0) == 6
        assert day2.predicted_first_fail_from_margin(curve, alpha=-1.0) == 4

    def test_a_margin_of_exactly_zero_counts_as_met(self):
        """sigma(0) = 0.5, and the boundary is met at exactly 0.5."""
        curve = np.array([4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0])
        assert day2.predicted_first_fail_from_margin(curve) == 5

    def test_a_large_positive_intercept_removes_the_crossing(self):
        curve = np.array([3.0, 2.0, 1.0, 0.5, -0.5, -1.0, -2.0, -3.0, -4.0])
        assert day2.predicted_first_fail_from_margin(curve, alpha=100.0) == (
            NO_CROSSING_INDEX
        )


def _frame(
    model_id: str,
    curves: dict[str, np.ndarray],
    family: str = "coverage",
    j_star: int = 4,
):
    rows = []
    for artifact_id, margins in curves.items():
        for s, m in enumerate(margins):
            rows.append(
                {
                    "model_id": model_id,
                    "artifact_id": artifact_id,
                    "family": family,
                    "latent_level": 3,
                    "strictness_index": s,
                    "margin": float(m),
                    "formal_truth": int(s < j_star),
                    "true_first_fail_index": j_star,
                }
            )
    return pd.DataFrame(rows)


class TestArtifactMetrics:
    ALPHAS = {"coverage": 0.5, "max_violation": 0.0, "numeric_tolerance": 0.0}

    def test_one_row_per_artifact(self):
        frame = _frame("m", {"a": np.arange(4, -5, -1.0), "b": np.arange(4, -5, -1.0)})
        result = day2.compute_artifact_metrics(frame, 0.0, self.ALPHAS)
        assert len(result) == 2

    def test_raw_and_corrected_localisation(self):
        # Unit steps and j* = 5: raw crossing is exact, a +1 intercept overshoots
        # by exactly one level, so the paired improvement is -1.
        curve = np.array([4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0])
        frame = _frame("m", {"a": curve}, j_star=5)
        row = day2.compute_artifact_metrics(frame, 1.0, self.ALPHAS).iloc[0]
        assert row["j_hat_raw"] == 5
        assert row["tce_raw"] == 0
        assert row["j_hat_global"] == 6
        assert row["tce_global"] == 1
        assert row["delta_tce_global"] == -1

    def test_a_correcting_intercept_reduces_tce(self):
        # Curve sits one level late; a -1 intercept pulls the crossing onto j*.
        curve = np.array([4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0])
        frame = _frame("m", {"a": curve}, j_star=4)
        row = day2.compute_artifact_metrics(frame, -1.0, self.ALPHAS).iloc[0]
        assert row["tce_raw"] == 1
        assert row["tce_global"] == 0
        assert row["delta_tce_global"] == 1

    def test_shape_metrics_do_not_depend_on_alpha(self):
        curve = np.array([1.0, 2.0, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -1.0])
        frame = _frame("m", {"a": curve})
        first = day2.compute_artifact_metrics(frame, 0.0, self.ALPHAS).iloc[0]
        second = day2.compute_artifact_metrics(frame, 12.5, self.ALPHAS).iloc[0]
        for column in ("mmvr", "mmvm", "span", "margin_spearman"):
            assert first[column] == pytest.approx(second[column], abs=1e-12)

    def test_missing_threshold_rejected(self):
        frame = _frame("m", {"a": np.arange(4, -5, -1.0)})
        with pytest.raises(ValueError):
            day2.compute_artifact_metrics(
                frame[frame["strictness_index"] != 5], 0.0, self.ALPHAS
            )

    def test_taxonomy_flags(self):
        orderly = np.array([3.0, 2.0, 1.0, 0.5, 0.4, 0.3, 0.2, 0.1, -1.0])
        frame = _frame("m", {"a": orderly})
        row = day2.compute_artifact_metrics(frame, 0.0, self.ALPHAS).iloc[0]
        assert row["mmvr"] == 0.0
        assert row["tce_raw"] == 4
        assert bool(row["mislocated_but_orderly"]) is True
        assert bool(row["locally_unstable"]) is False
        assert bool(row["correctly_localized"]) is False

    def test_correctable_by_global_bias_flag(self):
        curve = np.array([3.0, 2.0, 1.0, 0.5, 0.4, 0.3, 0.2, 0.1, -1.0])
        frame = _frame("m", {"a": curve})
        row = day2.compute_artifact_metrics(frame, -0.35, self.ALPHAS).iloc[0]
        assert row["tce_raw"] > 1
        assert row["tce_global"] <= 1
        assert bool(row["correctable_by_global_bias"]) is True

    def test_shape_invariance_report_passes(self):
        frame = _frame(
            "m",
            {
                "a": np.array([1.0, 2.0, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -1.0]),
                "b": np.arange(4, -5, -1.0),
            },
        )
        report = day2.shape_invariance_report(frame, [-10.0, 0.0, 3.3])
        assert report["mmvr_unchanged"]
        assert report["mmvm_unchanged"]
        assert report["span_unchanged"]
        assert report["spearman_unchanged"]
