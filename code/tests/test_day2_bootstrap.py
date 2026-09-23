"""Two-stage bootstrap: independent resampling, refitting, clustering, pairing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from formalcrrc import day2, day2_bootstrap
from formalcrrc.config import BOOTSTRAP_SEED, FAMILIES, N_STRICTNESS

N_PER_FAMILY = 8


def _artifact_table(prefix: str) -> pd.DataFrame:
    rows = []
    for family in FAMILIES:
        for i in range(N_PER_FAMILY):
            rows.append({"artifact_id": f"{prefix}-{family[:3]}-{i:02d}", "family": family})
    return pd.DataFrame(rows)


def _scores(table: pd.DataFrame, shift: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for entry in table.itertuples(index=False):
        j_star = int(rng.integers(1, N_STRICTNESS + 1))
        base = np.linspace(3.0, -3.0, N_STRICTNESS) + shift
        for s in range(N_STRICTNESS):
            rows.append(
                {
                    "model_id": "m",
                    "artifact_id": entry.artifact_id,
                    "family": entry.family,
                    "strictness_index": s,
                    "margin": float(base[s]),
                    "formal_truth": int(s < j_star),
                    "true_first_fail_index": j_star,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def tables():
    return _artifact_table("CAL"), _artifact_table("TST")


@pytest.fixture(scope="module")
def plan(tables):
    cal, test = tables
    return day2_bootstrap.build_two_stage_plan(cal, test, replicates=200)


class TestPlanConstruction:
    def test_same_seed_reproduces_both_stages(self, tables):
        cal, test = tables
        a = day2_bootstrap.build_two_stage_plan(cal, test, replicates=50)
        b = day2_bootstrap.build_two_stage_plan(cal, test, replicates=50)
        assert np.array_equal(a.calibration.indices, b.calibration.indices)
        assert np.array_equal(a.test.indices, b.test.indices)

    def test_calibration_and_test_are_independent_streams(self, plan):
        assert not np.array_equal(plan.calibration.indices, plan.test.indices)

    def test_seed_is_the_preregistered_one(self, plan):
        assert plan.seed == BOOTSTRAP_SEED == 42

    def test_stratification_holds_in_both_stages(self, plan):
        for partition in (plan.calibration, plan.test):
            for family in FAMILIES:
                columns = partition.index_columns(family)
                assert columns.size == N_PER_FAMILY
                drawn = np.asarray(partition.artifact_families)[
                    partition.indices[:, columns]
                ]
                assert np.all(drawn == family)

    def test_resampling_is_with_replacement(self, plan):
        assert any(len(np.unique(row)) < row.size for row in plan.test.indices)

    def test_indices_in_range(self, plan):
        for partition in (plan.calibration, plan.test):
            assert partition.indices.min() >= 0
            assert partition.indices.max() < len(partition.artifact_ids)


class TestClustering:
    def test_calibration_stack_keeps_nine_thresholds_together(self, tables, plan):
        cal, _ = tables
        margins, truths, family_index = day2_bootstrap.stack_calibration(
            _scores(cal, 0.0, 1), plan.calibration
        )
        assert margins.shape == (len(cal), N_STRICTNESS)
        assert truths.shape == (len(cal), N_STRICTNESS)
        assert family_index.shape == (len(cal),)

    def test_test_stack_keeps_nine_thresholds_together(self, tables, plan):
        _, test = tables
        stack = day2_bootstrap.stack_test(_scores(test, 0.0, 2), plan.test)
        assert stack.margins.shape == (len(test), N_STRICTNESS)
        assert stack.j_star.shape == (len(test),)

    def test_a_resampled_artifact_contributes_all_nine_rows(self, tables, plan):
        _, test = tables
        stack = day2_bootstrap.stack_test(_scores(test, 0.0, 2), plan.test)
        rows = plan.test.indices[0]
        assert stack.margins[rows].shape == (rows.size, N_STRICTNESS)

    def test_stack_rejects_a_missing_artifact(self, tables, plan):
        cal, _ = tables
        scores = _scores(cal, 0.0, 1)
        truncated = scores[scores["artifact_id"] != cal["artifact_id"].iloc[0]]
        with pytest.raises(ValueError):
            day2_bootstrap.stack_calibration(truncated, plan.calibration)

    def test_stack_rejects_a_partial_artifact(self, tables, plan):
        cal, _ = tables
        scores = _scores(cal, 0.0, 1)
        first = cal["artifact_id"].iloc[0]
        broken = scores[
            ~((scores["artifact_id"] == first) & (scores["strictness_index"] == 3))
        ]
        with pytest.raises(ValueError):
            day2_bootstrap.stack_calibration(broken, plan.calibration)


class TestRefitting:
    def test_alpha_is_refitted_inside_each_replicate(self, tables, plan):
        cal, _ = tables
        margins, truths, _ = day2_bootstrap.stack_calibration(
            _scores(cal, -2.0, 3), plan.calibration
        )
        alphas = [
            day2_bootstrap.refit_alpha_for_replicate(margins, truths, plan.calibration.indices[r])
            for r in range(25)
        ]
        assert len(set(np.round(alphas, 10))) > 1, "alpha did not vary across replicates"

    def test_refit_matches_a_direct_fit_on_the_same_rows(self, tables, plan):
        cal, _ = tables
        margins, truths, _ = day2_bootstrap.stack_calibration(
            _scores(cal, 1.0, 4), plan.calibration
        )
        rows = plan.calibration.indices[7]
        direct = day2.fit_intercept(margins[rows].ravel(), truths[rows].ravel()).alpha
        assert day2_bootstrap.refit_alpha_for_replicate(margins, truths, rows) == direct

    def test_family_refit_uses_only_that_family(self, tables, plan):
        cal, _ = tables
        margins, truths, family_index = day2_bootstrap.stack_calibration(
            _scores(cal, -1.0, 5), plan.calibration
        )
        rows = plan.calibration.indices[3]
        alphas = day2_bootstrap.refit_family_alphas_for_replicate(
            margins, truths, family_index, rows
        )
        assert set(alphas) == set(FAMILIES)
        for family in FAMILIES:
            mask = family_index[rows] == FAMILIES.index(family)
            expected = day2.fit_intercept(
                margins[rows[mask]].ravel(), truths[rows[mask]].ravel()
            ).alpha
            assert alphas[family] == pytest.approx(expected, abs=1e-12)

    def test_calibration_data_never_enters_the_test_stage(self, tables, plan):
        """Test-side TCE must depend on calibration only through alpha."""
        cal, test = tables
        cal_scores = _scores(cal, -2.0, 6)
        test_scores = _scores(test, -2.0, 7)
        first = day2_bootstrap.bootstrap_judge(cal_scores, test_scores, plan)

        stack = day2_bootstrap.stack_test(test_scores, plan.test)
        rows = plan.test.indices[0]
        sub = day2_bootstrap.TestStack(
            artifact_ids=tuple(stack.artifact_ids[i] for i in rows),
            margins=stack.margins[rows],
            truths=stack.truths[rows],
            j_star=stack.j_star[rows],
            family_index=stack.family_index[rows],
        )
        expected = day2_bootstrap.tce_for_shift(sub, first["alpha_global"][0]).mean()
        assert first["tce_global"][0] == pytest.approx(expected)


class TestCrossingVectorisation:
    def test_matches_the_scalar_implementation(self):
        rng = np.random.default_rng(11)
        block = rng.normal(0, 2, size=(50, N_STRICTNESS))
        vectorised = day2_bootstrap.crossing_index(block)
        scalar = [day2.predicted_first_fail_from_margin(row) for row in block]
        assert np.array_equal(vectorised, np.asarray(scalar))

    def test_never_negative_gives_the_sentinel(self):
        block = np.ones((3, N_STRICTNESS))
        assert np.all(day2_bootstrap.crossing_index(block) == N_STRICTNESS)

    def test_all_negative_gives_zero(self):
        block = -np.ones((3, N_STRICTNESS))
        assert np.all(day2_bootstrap.crossing_index(block) == 0)


class TestJudgeBootstrap:
    def test_output_shapes(self, tables, plan):
        cal, test = tables
        out = day2_bootstrap.bootstrap_judge(
            _scores(cal, -1.5, 8), _scores(test, -1.5, 9), plan
        )
        for key, values in out.items():
            assert values.shape == (plan.replicates,), key

    def test_deterministic(self, tables, plan):
        cal, test = tables
        cal_scores, test_scores = _scores(cal, -1.5, 8), _scores(test, -1.5, 9)
        a = day2_bootstrap.bootstrap_judge(cal_scores, test_scores, plan)
        b = day2_bootstrap.bootstrap_judge(cal_scores, test_scores, plan)
        for key in a:
            assert np.array_equal(a[key], b[key], equal_nan=True), key

    def test_a_biased_judge_gets_a_nonzero_alpha(self, tables, plan):
        cal, test = tables
        out = day2_bootstrap.bootstrap_judge(
            _scores(cal, -3.0, 10), _scores(test, -3.0, 11), plan
        )
        assert np.median(out["alpha_global"]) > 0.5

    def test_paired_comparison_uses_identical_resamples(self, tables, plan):
        cal, test = tables
        scores_cal, scores_test = _scores(cal, -1.0, 12), _scores(test, -1.0, 13)
        a = day2_bootstrap.bootstrap_judge(scores_cal, scores_test, plan)
        b = day2_bootstrap.bootstrap_judge(scores_cal, scores_test, plan)
        difference = a["tce_global"] - b["tce_global"]
        assert np.allclose(difference, 0.0)


class TestSummaries:
    def test_summarise_reports_interval_and_positive_fraction(self):
        samples = np.linspace(-1.0, 3.0, 5001)
        result = day2_bootstrap.summarise(samples, point=1.0)
        assert result["ci_low"] < result["point"] < result["ci_high"]
        assert result["fraction_positive"] == pytest.approx(0.75, abs=0.01)

    def test_excludes_zero_detected(self):
        result = day2_bootstrap.summarise(np.linspace(0.5, 1.5, 1000), point=1.0)
        assert result["excludes_zero"] is True

    def test_straddling_zero_detected(self):
        result = day2_bootstrap.summarise(np.linspace(-1.0, 1.0, 1000), point=0.0)
        assert result["excludes_zero"] is False

    def test_nan_replicates_are_dropped(self):
        samples = np.array([1.0, 2.0, np.nan, 3.0])
        assert day2_bootstrap.summarise(samples, point=2.0)["n_replicates"] == 3
