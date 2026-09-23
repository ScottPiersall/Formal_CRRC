"""Bootstrap determinism, cluster integrity, stratification and pairing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from formalcrrc import bootstrap
from formalcrrc.config import BOOTSTRAP_SEED, FAMILIES, N_STRICTNESS

N_PER_FAMILY = 12


def _artifact_table() -> tuple[list[str], list[str]]:
    ids, families = [], []
    for family in FAMILIES:
        for i in range(N_PER_FAMILY):
            ids.append(f"{family[:1].upper()}-{i:02d}")
            families.append(family)
    return ids, families


def _metric_frame(ids, families, values) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "artifact_id": ids,
            "family": families,
            "mvr": values,
            "mvm": np.asarray(values) / 2.0,
            "tce": np.arange(len(ids)) % 4,
            "accuracy": 1.0 - np.asarray(values),
            "brier": np.asarray(values) / 4.0,
        }
    )


@pytest.fixture()
def plan():
    ids, families = _artifact_table()
    return bootstrap.build_plan(ids, families, replicates=200, seed=BOOTSTRAP_SEED)


class TestDeterminism:
    def test_same_seed_gives_the_identical_index_matrix(self):
        ids, families = _artifact_table()
        a = bootstrap.build_plan(ids, families, replicates=50, seed=BOOTSTRAP_SEED)
        b = bootstrap.build_plan(ids, families, replicates=50, seed=BOOTSTRAP_SEED)
        assert np.array_equal(a.indices, b.indices)

    def test_different_seed_gives_a_different_matrix(self):
        ids, families = _artifact_table()
        a = bootstrap.build_plan(ids, families, replicates=50, seed=BOOTSTRAP_SEED)
        b = bootstrap.build_plan(ids, families, replicates=50, seed=BOOTSTRAP_SEED + 1)
        assert not np.array_equal(a.indices, b.indices)

    def test_intervals_are_reproducible(self, plan):
        ids, families = _artifact_table()
        rng = np.random.default_rng(0)
        frame = _metric_frame(ids, families, rng.random(len(ids)))
        first = bootstrap.bootstrap_metric(frame, plan, "mvr")
        second = bootstrap.bootstrap_metric(frame, plan, "mvr")
        assert first == second

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError):
            bootstrap.build_plan(["a", "b"], ["coverage"], replicates=5)


class TestStructure:
    def test_replicate_size_matches_the_design(self, plan):
        assert plan.indices.shape == (200, len(FAMILIES) * N_PER_FAMILY)

    def test_every_replicate_holds_the_designed_family_counts(self, plan):
        for family in FAMILIES:
            columns = plan.index_columns(family)
            assert columns.size == N_PER_FAMILY
            # Every index drawn for a family stratum belongs to that family.
            drawn_families = np.asarray(plan.artifact_families)[
                plan.indices[:, columns]
            ]
            assert np.all(drawn_families == family)

    def test_resampling_is_with_replacement(self, plan):
        # Over 200 replicates of 36 draws, repeats are essentially certain.
        repeats = [
            len(np.unique(row)) < row.size for row in plan.indices
        ]
        assert any(repeats)

    def test_indices_stay_in_range(self, plan):
        assert plan.indices.min() >= 0
        assert plan.indices.max() < len(plan.artifact_ids)


class TestClusterIntegrity:
    """An artifact enters a replicate with all nine of its thresholds or not at all."""

    def test_resampled_units_carry_all_nine_thresholds(self, plan):
        rows = []
        for artifact_id in plan.artifact_ids:
            for s in range(N_STRICTNESS):
                rows.append({"artifact_id": artifact_id, "strictness_index": s})
        scores = pd.DataFrame(rows)

        replicate = plan.indices[0]
        chosen = [plan.artifact_ids[i] for i in replicate]
        resampled = pd.concat(
            [scores[scores["artifact_id"] == aid] for aid in chosen],
            ignore_index=True,
        )
        assert len(resampled) == len(chosen) * N_STRICTNESS
        counts = resampled.groupby("artifact_id")["strictness_index"].nunique()
        assert set(counts) == {N_STRICTNESS}

    def test_metric_values_are_per_artifact_not_per_cell(self, plan):
        ids, families = _artifact_table()
        frame = _metric_frame(ids, families, np.linspace(0, 1, len(ids)))
        values = bootstrap.aligned_values(frame, plan, "mvr")
        assert values.shape == (len(ids),)


class TestEstimates:
    def test_point_estimate_is_the_plain_mean(self, plan):
        ids, families = _artifact_table()
        values = np.linspace(0.0, 1.0, len(ids))
        frame = _metric_frame(ids, families, values)
        result = bootstrap.bootstrap_metric(frame, plan, "mvr")
        assert result["point"] == pytest.approx(values.mean())
        assert result["n_units"] == len(ids)

    def test_family_scope_uses_only_that_family(self, plan):
        ids, families = _artifact_table()
        values = np.zeros(len(ids))
        target = FAMILIES[1]
        mask = np.asarray(families) == target
        values[mask] = 1.0
        frame = _metric_frame(ids, families, values)

        assert bootstrap.bootstrap_metric(frame, plan, "mvr", family=target)[
            "point"
        ] == pytest.approx(1.0)
        assert bootstrap.bootstrap_metric(frame, plan, "mvr", family=FAMILIES[0])[
            "point"
        ] == pytest.approx(0.0)

    def test_constant_metric_gives_a_degenerate_interval(self, plan):
        ids, families = _artifact_table()
        frame = _metric_frame(ids, families, np.full(len(ids), 0.25))
        result = bootstrap.bootstrap_metric(frame, plan, "mvr")
        assert result["ci_low"] == pytest.approx(0.25)
        assert result["ci_high"] == pytest.approx(0.25)

    def test_interval_brackets_the_point_estimate(self, plan):
        ids, families = _artifact_table()
        rng = np.random.default_rng(1)
        frame = _metric_frame(ids, families, rng.random(len(ids)))
        result = bootstrap.bootstrap_metric(frame, plan, "mvr")
        assert result["ci_low"] <= result["point"] <= result["ci_high"]

    def test_missing_artifact_is_rejected(self, plan):
        ids, families = _artifact_table()
        frame = _metric_frame(ids, families, np.zeros(len(ids))).iloc[1:]
        with pytest.raises(ValueError):
            bootstrap.bootstrap_metric(frame, plan, "mvr")


class TestPairedComparison:
    def test_identical_judges_give_a_zero_difference(self, plan):
        ids, families = _artifact_table()
        rng = np.random.default_rng(2)
        frame = _metric_frame(ids, families, rng.random(len(ids)))
        result = bootstrap.paired_difference(frame, frame, plan, "mvr")
        assert result["point"] == pytest.approx(0.0)
        assert result["ci_low"] == pytest.approx(0.0)
        assert result["ci_high"] == pytest.approx(0.0)
        assert result["excludes_zero"] is False

    def test_constant_offset_is_recovered_exactly(self, plan):
        """Pairing removes artifact-level variance entirely for a fixed shift."""
        ids, families = _artifact_table()
        rng = np.random.default_rng(3)
        base = rng.random(len(ids)) * 0.5
        a = _metric_frame(ids, families, base + 0.1)
        b = _metric_frame(ids, families, base)
        result = bootstrap.paired_difference(a, b, plan, "mvr")
        assert result["point"] == pytest.approx(0.1)
        assert result["ci_low"] == pytest.approx(0.1)
        assert result["ci_high"] == pytest.approx(0.1)
        assert result["excludes_zero"] is True

    def test_pairing_uses_the_same_resampled_artifacts_for_both_judges(self, plan):
        ids, families = _artifact_table()
        a_values = np.zeros(len(ids))
        b_values = np.ones(len(ids))
        a = _metric_frame(ids, families, a_values)
        b = _metric_frame(ids, families, b_values)
        result = bootstrap.paired_difference(a, b, plan, "mvr")
        assert result["point"] == pytest.approx(-1.0)
        assert result["ci_low"] == pytest.approx(-1.0)

    def test_comparison_table_covers_every_unordered_pair(self, plan):
        ids, families = _artifact_table()
        frames = {
            f"model{i}": _metric_frame(ids, families, np.full(len(ids), 0.1 * i))
            for i in range(3)
        }
        table = bootstrap.paired_comparison_table(
            frames, plan, metrics=["mvr"]
        )
        pairs = set(zip(table["model_a"], table["model_b"]))
        assert pairs == {
            ("model0", "model1"),
            ("model0", "model2"),
            ("model1", "model2"),
        }
        assert set(table["scope"]) == {"overall", *FAMILIES}


class TestTables:
    def test_table_covers_every_model_metric_and_scope(self, plan):
        ids, families = _artifact_table()
        frames = {
            "m1": _metric_frame(ids, families, np.full(len(ids), 0.2)),
            "m2": _metric_frame(ids, families, np.full(len(ids), 0.3)),
        }
        table = bootstrap.bootstrap_table(frames, plan)
        expected = (
            len(frames) * len(bootstrap.BOOTSTRAP_METRICS) * (1 + len(FAMILIES))
        )
        assert len(table) == expected
        assert set(table["metric"]) == set(bootstrap.BOOTSTRAP_METRICS)

    def test_percentile_interval_matches_numpy_quantiles(self):
        samples = np.linspace(0.0, 1.0, 10001)
        low, high = bootstrap.percentile_interval(samples, ci=0.95)
        assert low == pytest.approx(0.025, abs=1e-3)
        assert high == pytest.approx(0.975, abs=1e-3)
