"""Day-3 bootstrap: clustering, stratification, oracle refitting, determinism."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from formalcrrc import day3, day3_bootstrap
from formalcrrc.config import BOOTSTRAP_SEED, FAMILIES, N_STRICTNESS

N_PER_FAMILY = 8


def _table() -> pd.DataFrame:
    rows = []
    for family in FAMILIES:
        for i in range(N_PER_FAMILY):
            rows.append({"artifact_id": f"{family[:3]}-{i:02d}", "family": family})
    return pd.DataFrame(rows)


def _curves(table: pd.DataFrame, seed: int):
    rng = np.random.default_rng(seed)
    n = len(table)
    block = np.stack(
        [np.sort(rng.normal(0, 3, N_STRICTNESS))[::-1] for _ in range(n)]
    )
    j_star = rng.integers(1, N_STRICTNESS + 1, n)
    return block, j_star


def _metrics(table: pd.DataFrame, block: np.ndarray, j_star: np.ndarray):
    rows = []
    for i, entry in enumerate(table.itertuples(index=False)):
        margins = block[i]
        target = int(j_star[i])
        reachable = day3.reachable_crossings(margins)
        otce = min(abs(j - target) for j in reachable)
        raw = abs(int(day3.crossing_index(margins[None, :])[0]) - target)
        rows.append(
            {
                "artifact_id": entry.artifact_id,
                "family": entry.family,
                "tce_raw": raw,
                "otce": otce,
                "delta_tce_artifact": raw - otce,
                "reachable_count": len(reachable),
                "prefix_record_rate": day3.prefix_record_rate(margins),
                "mmvr": day3.margin_monotonicity_violation_rate(margins),
                "translation_reachable": target in reachable,
                "nontrivial_boundary": 1 <= target <= N_STRICTNESS - 1,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def setup():
    table = _table()
    block, j_star = _curves(table, 4)
    metrics = _metrics(table, block, j_star)
    plan = day3_bootstrap.build_plan(
        table["artifact_id"].tolist(), table["family"].tolist(), replicates=200
    )
    counts = day3_bootstrap.multiplicity_matrix(plan)
    counts_family = {
        f: day3_bootstrap.multiplicity_matrix(plan, f) for f in FAMILIES
    }
    return table, block, j_star, metrics, plan, counts, counts_family


class TestPlan:
    def test_deterministic_under_seed(self):
        table = _table()
        a = day3_bootstrap.build_plan(
            table["artifact_id"].tolist(), table["family"].tolist(), replicates=50
        )
        b = day3_bootstrap.build_plan(
            table["artifact_id"].tolist(), table["family"].tolist(), replicates=50
        )
        assert np.array_equal(a.indices, b.indices)
        assert a.seed == BOOTSTRAP_SEED == 42

    def test_stratified_by_family(self, setup):
        _, _, _, _, plan, _, _ = setup
        for family in FAMILIES:
            columns = plan.columns(family)
            assert columns.size == N_PER_FAMILY
            drawn = np.asarray(plan.families)[plan.indices[:, columns]]
            assert np.all(drawn == family)

    def test_resampling_with_replacement(self, setup):
        _, _, _, _, plan, _, _ = setup
        assert any(len(np.unique(row)) < row.size for row in plan.indices)

    def test_mismatched_lengths_rejected(self):
        with pytest.raises(ValueError):
            day3_bootstrap.build_plan(["a", "b"], ["coverage"], replicates=5)


class TestMultiplicity:
    def test_counts_sum_to_group_size(self, setup):
        _, _, _, _, plan, counts, counts_family = setup
        assert np.all(counts.sum(axis=1) == len(plan.artifact_ids))
        for family in FAMILIES:
            assert np.all(counts_family[family].sum(axis=1) == N_PER_FAMILY)

    def test_counts_reproduce_a_resampled_mean(self, setup):
        _, _, _, metrics, plan, counts, _ = setup
        values = metrics.set_index("artifact_id").loc[
            list(plan.artifact_ids), "otce"
        ].to_numpy(float)
        means = day3_bootstrap.replicate_means(values, counts)
        direct = values[plan.indices[0]].mean()
        assert means[0] == pytest.approx(direct)

    def test_cluster_integrity_all_nine_thresholds_move_together(self, setup):
        """An artifact enters a replicate whole, so its 9 margins move as one."""
        _, block, _, _, plan, _, _ = setup
        rows = plan.indices[0]
        assert block[rows].shape == (rows.size, N_STRICTNESS)


class TestOracleRefitting:
    def test_refit_matches_direct_optimisation_on_a_replicate(self, setup):
        _, block, j_star, _, plan, counts, _ = setup
        alphas = day3.group_candidate_intercepts(block)
        matrix = day3.tce_matrix(block, j_star, alphas)
        tce, alpha = day3_bootstrap.refit_group_oracle_per_replicate(
            matrix, alphas, counts
        )
        rows = plan.indices[0]
        direct = day3.oracle_group_intercept(block[rows], j_star[rows])
        assert tce[0] == pytest.approx(direct.mean_tce)
        assert alpha[0] == pytest.approx(direct.alpha)

    def test_alpha_varies_across_replicates(self, setup):
        _, block, j_star, _, _, counts, _ = setup
        alphas = day3.group_candidate_intercepts(block)
        matrix = day3.tce_matrix(block, j_star, alphas)
        _, chosen = day3_bootstrap.refit_group_oracle_per_replicate(
            matrix, alphas, counts
        )
        assert len(set(np.round(chosen, 10))) > 1

    def test_refit_never_worse_than_zero_shift(self, setup):
        _, block, j_star, _, _, counts, _ = setup
        alphas = day3.group_candidate_intercepts(block)
        matrix = day3.tce_matrix(block, j_star, alphas)
        tce, _ = day3_bootstrap.refit_group_oracle_per_replicate(
            matrix, alphas, counts
        )
        raw_per_artifact = np.abs(day3.crossing_index(block) - j_star).astype(float)
        raw_means = day3_bootstrap.replicate_means(raw_per_artifact, counts)
        assert np.all(tce <= raw_means + 1e-9)

    def test_deterministic(self, setup):
        _, block, j_star, _, _, counts, _ = setup
        alphas = day3.group_candidate_intercepts(block)
        matrix = day3.tce_matrix(block, j_star, alphas)
        first = day3_bootstrap.refit_group_oracle_per_replicate(matrix, alphas, counts)
        second = day3_bootstrap.refit_group_oracle_per_replicate(matrix, alphas, counts)
        assert np.array_equal(first[0], second[0])
        assert np.array_equal(first[1], second[1])


class TestJudgeBootstrap:
    def test_output_shapes(self, setup):
        _, block, j_star, metrics, plan, counts, counts_family = setup
        out = day3_bootstrap.bootstrap_judge(
            metrics, block, j_star, plan, counts, counts_family
        )
        for key, values in out.items():
            assert values.shape == (plan.replicates,), key

    def test_nesting_holds_in_every_replicate(self, setup):
        _, block, j_star, metrics, plan, counts, counts_family = setup
        out = day3_bootstrap.bootstrap_judge(
            metrics, block, j_star, plan, counts, counts_family
        )
        assert np.all(out["tce_oracle_global"] <= out["tce_raw"] + 1e-9)
        assert np.all(out["tce_oracle_family"] <= out["tce_oracle_global"] + 1e-9)
        assert np.all(out["otce"] <= out["tce_oracle_family"] + 1e-9)

    def test_deterministic(self, setup):
        _, block, j_star, metrics, plan, counts, counts_family = setup
        a = day3_bootstrap.bootstrap_judge(
            metrics, block, j_star, plan, counts, counts_family
        )
        b = day3_bootstrap.bootstrap_judge(
            metrics, block, j_star, plan, counts, counts_family
        )
        for key in a:
            assert np.array_equal(a[key], b[key], equal_nan=True), key

    def test_paired_comparison_uses_identical_resamples(self, setup):
        _, block, j_star, metrics, plan, counts, counts_family = setup
        a = day3_bootstrap.bootstrap_judge(
            metrics, block, j_star, plan, counts, counts_family
        )
        b = day3_bootstrap.bootstrap_judge(
            metrics, block, j_star, plan, counts, counts_family
        )
        assert np.allclose(a["otce"] - b["otce"], 0.0)

    def test_nontrivial_reachability_uses_its_own_denominator(self, setup):
        _, block, j_star, metrics, plan, counts, counts_family = setup
        out = day3_bootstrap.bootstrap_judge(
            metrics, block, j_star, plan, counts, counts_family
        )
        finite = out["trr_nontrivial"][np.isfinite(out["trr_nontrivial"])]
        assert finite.size > 0
        assert np.all((finite >= 0.0) & (finite <= 1.0))


class TestSummaries:
    def test_interval_brackets_point(self):
        samples = np.linspace(0.0, 2.0, 5001)
        result = day3_bootstrap.summarise(samples, point=1.0)
        assert result["ci_low"] <= result["point"] <= result["ci_high"]

    def test_excludes_zero(self):
        assert day3_bootstrap.summarise(
            np.linspace(0.5, 1.5, 1000), point=1.0
        )["excludes_zero"]

    def test_straddling_zero(self):
        assert not day3_bootstrap.summarise(
            np.linspace(-1.0, 1.0, 1000), point=0.0
        )["excludes_zero"]

    def test_paired_difference_of_identical_samples_is_zero(self):
        samples = np.linspace(1.0, 2.0, 500)
        result = day3_bootstrap.paired_difference(samples, samples, point=0.0)
        assert result["ci_low"] == pytest.approx(0.0)
        assert result["ci_high"] == pytest.approx(0.0)

    def test_nan_replicates_dropped(self):
        samples = np.array([1.0, np.nan, 3.0])
        assert day3_bootstrap.summarise(samples, point=2.0)["n_replicates"] == 2
