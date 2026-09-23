"""The reachability theorem, the oracle bounds, and their cross-checks.

The theorem is the load-bearing claim of Day 3, so it is tested against an
independent enumeration implementation on canonical curves, on hand-worked
cases, and on thousands of random ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from formalcrrc import day3
from formalcrrc.config import FAMILIES, N_STRICTNESS, NO_CROSSING_INDEX

DECREASING = np.array([8.0, 7, 6, 5, 4, 3, 2, 1, 0])
INCREASING = np.array([0.0, 1, 2, 3, 4, 5, 6, 7, 8])
FLAT = np.array([1.0] * N_STRICTNESS)
SPARSE = np.array([8.0, 9, 6, 7, 5, 6, 7, 3, 4])


class TestPrefixRecordLows:
    def test_strictly_decreasing_records_every_level(self):
        assert day3.prefix_record_lows(DECREASING) == tuple(range(1, 9))

    def test_strictly_increasing_records_nothing(self):
        assert day3.prefix_record_lows(INCREASING) == ()

    def test_flat_records_nothing(self):
        assert day3.prefix_record_lows(FLAT) == ()

    def test_sparse_case_hand_worked(self):
        # 8, then 9 (no), 6 < 8 yes, 7 no, 5 < 6 yes, 6 no, 7 no, 3 < 5 yes, 4 no
        assert day3.prefix_record_lows(SPARSE) == (2, 4, 7)

    def test_equality_is_not_a_record(self):
        curve = np.array([5.0, 5, 5, 4, 4, 3, 3, 3, 3])
        assert day3.prefix_record_lows(curve) == (3, 5)

    def test_wrong_length_rejected(self):
        with pytest.raises(ValueError):
            day3.prefix_record_lows(np.zeros(3))

    def test_non_finite_rejected(self):
        curve = DECREASING.copy()
        curve[2] = np.nan
        with pytest.raises(ValueError):
            day3.prefix_record_lows(curve)


class TestReachableSet:
    def test_extremes_always_present(self):
        for curve in (DECREASING, INCREASING, FLAT, SPARSE):
            reachable = day3.reachable_crossings(curve)
            assert 0 in reachable
            assert NO_CROSSING_INDEX in reachable

    def test_strictly_decreasing_reaches_everything(self):
        assert day3.reachable_crossings(DECREASING) == tuple(range(10))
        assert day3.reachable_count(DECREASING) == 10

    def test_strictly_increasing_reaches_only_extremes(self):
        assert day3.reachable_crossings(INCREASING) == (0, 9)
        assert day3.reachable_count(INCREASING) == 2

    def test_flat_reaches_only_extremes(self):
        assert day3.reachable_crossings(FLAT) == (0, 9)

    def test_sparse_case(self):
        assert day3.reachable_crossings(SPARSE) == (0, 2, 4, 7, 9)

    def test_reachable_count_bounds(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            curve = rng.normal(0, 3, N_STRICTNESS)
            assert 2 <= day3.reachable_count(curve) <= 10

    def test_prefix_record_rate(self):
        assert day3.prefix_record_rate(DECREASING) == 1.0
        assert day3.prefix_record_rate(INCREASING) == 0.0
        assert day3.prefix_record_rate(SPARSE) == pytest.approx(3 / 8)


class TestTheoremVersusEnumeration:
    """Method A and Method B must agree exactly, always."""

    @pytest.mark.parametrize(
        "curve", [DECREASING, INCREASING, FLAT, SPARSE]
    )
    def test_canonical_curves_agree(self, curve):
        assert day3.reachable_crossings(curve) == (
            day3.reachable_crossings_by_enumeration(curve)
        )

    def test_thousands_of_random_curves_agree(self):
        rng = np.random.default_rng(20260907)
        for _ in range(3000):
            curve = rng.normal(0.0, 5.0, N_STRICTNESS)
            assert day3.reachable_crossings(curve) == (
                day3.reachable_crossings_by_enumeration(curve)
            )

    def test_curves_with_many_ties_agree(self):
        rng = np.random.default_rng(11)
        for _ in range(1000):
            curve = rng.integers(-3, 4, N_STRICTNESS).astype(float)
            assert day3.reachable_crossings(curve) == (
                day3.reachable_crossings_by_enumeration(curve)
            )

    def test_near_identical_margins_agree(self):
        """Breakpoints closer than a midpoint is representable."""
        base = 1.0
        curve = np.array([base + i * np.finfo(float).eps for i in range(9)])
        assert day3.reachable_crossings(curve) == (
            day3.reachable_crossings_by_enumeration(curve)
        )

    def test_otce_agrees_on_random_curves(self):
        rng = np.random.default_rng(7)
        for _ in range(2000):
            curve = rng.normal(0.0, 4.0, N_STRICTNESS)
            j_star = int(rng.integers(1, 10))
            assert day3.oracle_translation_tce(curve, j_star) == (
                day3.oracle_translation_tce_by_enumeration(curve, j_star)
            )


class TestOTCE:
    def test_exact_target_reachable(self):
        assert day3.oracle_translation_tce(DECREASING, 4) == 0

    def test_one_level_from_reachable(self):
        # reachable {0,2,4,7,9}; target 3 is one away from 2 and from 4
        assert day3.oracle_translation_tce(SPARSE, 3) == 1

    def test_multiple_levels_away(self):
        # reachable {0,9} only; target 5 is 4 from 9 and 5 from 0
        assert day3.oracle_translation_tce(INCREASING, 5) == 4

    def test_target_nine_always_free(self):
        for curve in (DECREASING, INCREASING, FLAT, SPARSE):
            assert day3.oracle_translation_tce(curve, NO_CROSSING_INDEX) == 0

    def test_flat_curve_worst_case_target(self):
        # reachable {0,9}; the hardest target is 4 or 5
        assert day3.oracle_translation_tce(FLAT, 4) == 4
        assert day3.oracle_translation_tce(FLAT, 5) == 4

    def test_otce_never_exceeds_raw_tce(self):
        rng = np.random.default_rng(3)
        from formalcrrc.day2_bootstrap import crossing_index

        for _ in range(2000):
            curve = rng.normal(0.0, 3.0, N_STRICTNESS)
            j_star = int(rng.integers(1, 10))
            raw = abs(int(crossing_index(curve[None, :])[0]) - j_star)
            assert day3.oracle_translation_tce(curve, j_star) <= raw

    def test_nearest_reachable_is_deterministic_and_closest(self):
        assert day3.nearest_reachable_crossing(SPARSE, 3) == 2
        assert day3.nearest_reachable_crossing(SPARSE, 5) == 4

    def test_translation_reachable_flag(self):
        assert day3.translation_reachable(SPARSE, 4) is True
        assert day3.translation_reachable(SPARSE, 3) is False
        assert day3.translation_reachable(INCREASING, 9) is True


class TestTies:
    def test_equal_prefix_minimum_creates_no_crossing(self):
        curve = np.array([3.0, 3, 3, 3, 3, 3, 3, 3, 3])
        assert day3.reachable_crossings(curve) == (0, 9)

    def test_target_blocked_specifically_by_equality(self):
        # level 3 equals the running minimum, so it is not reachable
        curve = np.array([5.0, 4, 4, 4, 2, 1, 0, -1, -2])
        assert 3 not in day3.reachable_crossings(curve)
        diagnostics = day3.tie_diagnostics(curve, 3)
        assert diagnostics["target_blocked_by_equality"] is True

    def test_tie_counts_are_descriptive(self):
        curve = np.array([2.0, 2, 2, 1, 1, 0, 0, 0, 0])
        # pairs equal: (0,1) (1,2) (3,4) (5,6) (6,7) (7,8) = 6
        # prefix ties: levels 1, 2, 4, 6, 7, 8 = 6
        diagnostics = day3.tie_diagnostics(curve, 5)
        assert diagnostics["adjacent_equal_margins"] == 6
        assert diagnostics["prefix_tie_levels"] == 6

    def test_no_tolerance_is_applied(self):
        """A margin lower by one ULP is a genuine record low."""
        curve = np.array([1.0] * N_STRICTNESS)
        curve[4] = np.nextafter(1.0, -np.inf)
        assert 4 in day3.reachable_crossings(curve)


class TestCandidateIntercepts:
    def test_candidates_cover_every_reachable_state(self):
        rng = np.random.default_rng(5)
        for _ in range(300):
            curve = rng.normal(0, 2, N_STRICTNESS)
            alphas = day3.candidate_intercepts(curve)
            states = {
                int(j)
                for j in day3.crossing_index(curve[None, :] + alphas[:, None])
            }
            assert states == set(day3.reachable_crossings(curve))

    def test_candidates_are_sorted_and_unique(self):
        alphas = day3.candidate_intercepts(SPARSE)
        assert np.all(np.diff(alphas) > 0)


def _block(curves: list[np.ndarray]) -> np.ndarray:
    return np.stack(curves)


class TestGroupOracle:
    def test_matches_brute_force_on_a_toy_group(self):
        rng = np.random.default_rng(13)
        block = _block([rng.normal(0, 2, N_STRICTNESS) for _ in range(12)])
        j_star = rng.integers(1, 10, 12)
        oracle = day3.oracle_group_intercept(block, j_star)

        grid = np.linspace(-30, 30, 24001)
        best = min(
            float(
                np.mean(np.abs(day3.crossing_index(block + a) - j_star))
            )
            for a in grid
        )
        assert oracle.mean_tce <= best + 1e-12

    def test_tie_break_prefers_smallest_absolute_alpha(self):
        alphas = np.array([-2.0, 0.0, 2.0, 5.0])
        mean_tce = np.array([1.0, 1.0, 1.0, 3.0])
        chosen = day3.select_group_oracle(alphas, mean_tce, 10)
        assert chosen.alpha == 0.0

    def test_tie_break_then_prefers_numerically_smallest(self):
        alphas = np.array([-2.0, 2.0])
        mean_tce = np.array([1.0, 1.0])
        assert day3.select_group_oracle(alphas, mean_tce, 10).alpha == -2.0

    def test_oracle_is_never_worse_than_zero_shift(self):
        rng = np.random.default_rng(17)
        for _ in range(50):
            block = _block([rng.normal(0, 3, N_STRICTNESS) for _ in range(20)])
            j_star = rng.integers(1, 10, 20)
            raw = float(np.mean(np.abs(day3.crossing_index(block) - j_star)))
            assert day3.oracle_group_intercept(block, j_star).mean_tce <= raw + 1e-12

    def test_deterministic(self):
        rng = np.random.default_rng(19)
        block = _block([rng.normal(0, 2, N_STRICTNESS) for _ in range(15)])
        j_star = rng.integers(1, 10, 15)
        first = day3.oracle_group_intercept(block, j_star)
        second = day3.oracle_group_intercept(block, j_star)
        assert first == second

    def test_tce_matrix_shape_and_values(self):
        block = _block([DECREASING, INCREASING])
        j_star = np.array([4, 4])
        alphas = np.array([0.0, -4.5])
        matrix = day3.tce_matrix(block, j_star, alphas)
        assert matrix.shape == (2, 2)
        assert matrix[0, 0] == abs(int(day3.crossing_index(DECREASING[None, :])[0]) - 4)


class TestOracleNesting:
    """G subset F subset A, so the optimised bounds must be ordered."""

    def test_nesting_holds_on_synthetic_data(self):
        rng = np.random.default_rng(23)
        n_per_family = 20
        block, j_star, families = [], [], []
        for offset, family in zip((-4.0, 0.0, 4.0), FAMILIES):
            for _ in range(n_per_family):
                block.append(np.sort(rng.normal(offset, 2, N_STRICTNESS))[::-1])
                j_star.append(int(rng.integers(1, 10)))
                families.append(family)
        block = np.stack(block)
        j_star = np.asarray(j_star)
        families = np.asarray(families)

        raw = float(np.mean(np.abs(day3.crossing_index(block) - j_star)))
        global_oracle = day3.oracle_group_intercept(block, j_star)

        family_total, family_n = 0.0, 0
        for family in FAMILIES:
            rows = np.flatnonzero(families == family)
            oracle = day3.oracle_group_intercept(block[rows], j_star[rows])
            family_total += oracle.mean_tce * rows.size
            family_n += rows.size
        family_mean = family_total / family_n

        artifact_mean = float(
            np.mean(
                [
                    day3.oracle_translation_tce(block[i], int(j_star[i]))
                    for i in range(block.shape[0])
                ]
            )
        )

        report = day3.assert_oracle_nesting(
            raw, global_oracle.mean_tce, family_mean, artifact_mean
        )
        assert report["status"] == "PASS", report

    def test_nesting_detects_a_violation(self):
        report = day3.assert_oracle_nesting(3.0, 2.0, 2.5, 1.0)
        assert report["status"] == "FAIL"
        assert report["checks"]["family_le_global"] is False


def _scores_frame(curves: dict[str, np.ndarray], j_stars: dict[str, int]):
    rows = []
    for artifact_id, margins in curves.items():
        for s, m in enumerate(margins):
            rows.append(
                {
                    "model_id": "m",
                    "artifact_id": artifact_id,
                    "family": FAMILIES[0],
                    "latent_level": 3,
                    "strictness_index": s,
                    "margin": float(m),
                    "formal_truth": int(s < j_stars[artifact_id]),
                    "true_first_fail_index": j_stars[artifact_id],
                }
            )
    return pd.DataFrame(rows)


class TestArtifactMetrics:
    def test_one_row_per_artifact_with_expected_fields(self):
        frame = _scores_frame({"a": DECREASING, "b": SPARSE}, {"a": 4, "b": 3})
        alphas = {f: 0.0 for f in FAMILIES}
        result = day3.compute_artifact_metrics(frame, 0.0, alphas)
        assert len(result) == 2
        row = result.set_index("artifact_id").loc["b"]
        assert row["otce"] == 1
        assert row["reachable_count"] == 5
        assert bool(row["translation_reachable"]) is False
        assert bool(row["translation_near_solvable"]) is True

    def test_otce_bounded_by_raw(self):
        frame = _scores_frame({"a": SPARSE}, {"a": 6})
        alphas = {f: 0.0 for f in FAMILIES}
        row = day3.compute_artifact_metrics(frame, 0.0, alphas).iloc[0]
        assert row["otce"] <= row["tce_raw"]
        assert row["delta_tce_artifact"] == row["tce_raw"] - row["otce"]

    def test_taxonomy_flags(self):
        frame = _scores_frame({"a": DECREASING, "b": FLAT}, {"a": 4, "b": 4})
        alphas = {f: 0.0 for f in FAMILIES}
        result = day3.compute_artifact_metrics(frame, 0.0, alphas).set_index(
            "artifact_id"
        )
        assert bool(result.loc["a", "translation_solvable"]) is True
        assert bool(result.loc["a", "order_rich"]) is True
        assert bool(result.loc["b", "translation_limited"]) is True
        assert bool(result.loc["b", "order_poor"]) is True

    def test_reachability_table_cross_checks_both_methods(self):
        frame = _scores_frame({"a": DECREASING, "b": SPARSE}, {"a": 4, "b": 3})
        table = day3.reachability_table(frame)
        assert table["methods_agree"].all()
        assert (table["otce_theorem"] == table["otce_enumeration"]).all()

    def test_missing_threshold_rejected(self):
        frame = _scores_frame({"a": DECREASING}, {"a": 4})
        with pytest.raises(ValueError):
            day3.compute_artifact_metrics(
                frame[frame["strictness_index"] != 5], 0.0, {f: 0.0 for f in FAMILIES}
            )


class TestTranslationInvariance:
    """Translation must not disturb any shape descriptor Day 3 records."""

    @pytest.mark.parametrize("alpha", [-30.0, -1.0, 0.0, 1.0, 30.0])
    def test_reachable_set_is_translation_invariant(self, alpha):
        rng = np.random.default_rng(29)
        for _ in range(200):
            curve = rng.normal(0, 3, N_STRICTNESS)
            assert day3.reachable_crossings(curve + alpha) == (
                day3.reachable_crossings(curve)
            )

    @pytest.mark.parametrize("alpha", [-12.5, 0.0, 7.25])
    def test_shape_metrics_are_translation_invariant(self, alpha):
        curve = SPARSE
        assert day3.margin_monotonicity_violation_rate(
            curve + alpha
        ) == day3.margin_monotonicity_violation_rate(curve)
        assert day3.response_span(curve + alpha) == pytest.approx(
            day3.response_span(curve), abs=1e-9
        )
