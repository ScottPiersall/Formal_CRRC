"""The post-study analytic reference baselines.

These baselines are **not preregistered**. They were derived after Days 1-3 were
complete, they involve no model inference, and nothing here may change a frozen
result. The last class in this file enforces exactly that.

Everything is exact rational arithmetic: an analytic baseline with a closed form
is never estimated by sampling, so there is no tolerance to tune and no seed to
record.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from formalcrrc import baselines, predicates, provenance
from formalcrrc.config import N_STRICTNESS, NO_CROSSING_INDEX

#: The design's boundary distribution: uniform over 1..9.
UNIFORM = {j: 1 for j in range(1, NO_CROSSING_INDEX + 1)}

#: Frozen manifests that between them pin every Day-1/2/3 file.
PRIOR_MANIFEST = "artifacts/day3/prior_experiments_manifest.json"
DAY3_MANIFEST = "artifacts/day3/integrity_manifest.json"

#: The frozen partitions the baselines are verified against.
PARTITION_PATHS = (
    "artifacts/day1/formal_dataset.parquet",
    "artifacts/day2/calibration_dataset.parquet",
    "artifacts/day2/test_dataset.parquet",
    "artifacts/day3/test_dataset.parquet",
)


# --------------------------------------------------------------------------
# 1. Exact fractions and decimals for all four baselines
# --------------------------------------------------------------------------


class TestExactBaselineValues:
    def test_fixed_center_tce_is_twenty_ninths(self):
        assert baselines.constant_crossing_tce(5, UNIFORM) == Fraction(20, 9)

    def test_fixed_center_accuracy_is_sixty_one_eighty_firsts(self):
        tce = baselines.constant_crossing_tce(5, UNIFORM)
        assert baselines.accuracy_from_tce(tce) == Fraction(61, 81)

    def test_uniform_random_tce_is_eighty_twenty_sevenths(self):
        assert baselines.uniform_random_crossing_tce(UNIFORM) == Fraction(80, 27)

    def test_uniform_random_accuracy_is_one_sixty_three_two_forty_thirds(self):
        tce = baselines.uniform_random_crossing_tce(UNIFORM)
        assert baselines.accuracy_from_tce(tce) == Fraction(163, 243)

    def test_always_met_tce_is_exactly_four(self):
        assert baselines.constant_crossing_tce(NO_CROSSING_INDEX, UNIFORM) == Fraction(4)

    def test_always_met_accuracy_is_five_ninths(self):
        tce = baselines.constant_crossing_tce(NO_CROSSING_INDEX, UNIFORM)
        assert baselines.accuracy_from_tce(tce) == Fraction(5, 9)

    def test_always_not_met_tce_is_exactly_five(self):
        assert baselines.constant_crossing_tce(0, UNIFORM) == Fraction(5)

    def test_always_not_met_accuracy_is_four_ninths(self):
        tce = baselines.constant_crossing_tce(0, UNIFORM)
        assert baselines.accuracy_from_tce(tce) == Fraction(4, 9)

    @pytest.mark.parametrize(
        "key,tce,accuracy",
        [
            ("fixed_center", 2.222222, 0.753086),
            ("uniform_random_crossing", 2.962963, 0.670782),
            ("always_met", 4.000000, 0.555556),
            ("always_not_met", 5.000000, 0.444444),
        ],
    )
    def test_reported_decimals(self, key, tce, accuracy):
        baseline = baselines.BASELINES_BY_KEY[key]
        assert float(baseline.tce(UNIFORM)) == pytest.approx(tce, abs=5e-7)
        assert float(baseline.accuracy(UNIFORM)) == pytest.approx(accuracy, abs=5e-7)

    def test_the_random_baseline_is_the_mean_of_the_constant_baselines(self):
        """The expectation is exact by construction, never sampled."""
        support = baselines.BOUNDARY_SUPPORT
        manual = sum(
            (baselines.constant_crossing_tce(j, UNIFORM) for j in support),
            Fraction(0),
        ) / len(support)
        assert baselines.uniform_random_crossing_tce(UNIFORM) == manual

    def test_the_random_baseline_matches_the_double_sum_closed_form(self):
        support = range(1, NO_CROSSING_INDEX + 1)
        double_sum = sum(abs(j - k) for j in support for k in support)
        assert baselines.uniform_random_crossing_tce(UNIFORM) == Fraction(
            double_sum, 81
        )

    def test_no_baseline_inspects_the_artifact(self):
        assert not any(b.uses_artifact_input for b in baselines.BASELINES)


# --------------------------------------------------------------------------
# 2. Fixed center is the minimum-TCE constant crossing
# --------------------------------------------------------------------------


class TestOptimalConstantCrossing:
    def test_five_minimises_expected_tce(self):
        best_j, best_tce = baselines.minimum_tce_constant_crossing(UNIFORM)
        assert best_j == 5
        assert best_tce == Fraction(20, 9)

    def test_the_minimum_is_unique(self):
        best = baselines.constant_crossing_tce(5, UNIFORM)
        others = [
            j
            for j in baselines.PREDICTION_SUPPORT
            if j != 5 and baselines.constant_crossing_tce(j, UNIFORM) == best
        ]
        assert others == []

    @pytest.mark.parametrize("j_hat", [j for j in range(0, 10) if j != 5])
    def test_every_other_constant_crossing_is_strictly_worse(self, j_hat):
        assert baselines.constant_crossing_tce(j_hat, UNIFORM) > Fraction(20, 9)

    def test_the_curve_is_symmetric_about_five(self):
        for offset in range(1, 5):
            assert baselines.constant_crossing_tce(
                5 - offset, UNIFORM
            ) == baselines.constant_crossing_tce(5 + offset, UNIFORM)

    def test_fixed_center_beats_the_uniform_random_expectation(self):
        assert baselines.constant_crossing_tce(
            5, UNIFORM
        ) < baselines.uniform_random_crossing_tce(UNIFORM)


# --------------------------------------------------------------------------
# 3. Majority-class label
# --------------------------------------------------------------------------


class TestMajorityClass:
    def test_the_majority_row_label_is_met(self):
        label, _ = baselines.majority_label(UNIFORM)
        assert label == 1

    def test_the_majority_share_is_five_ninths(self):
        _, share = baselines.majority_label(UNIFORM)
        assert share == Fraction(5, 9)

    def test_always_met_reproduces_the_majority_share(self):
        _, share = baselines.majority_label(UNIFORM)
        assert baselines.BASELINES_BY_KEY["always_met"].accuracy(UNIFORM) == share

    def test_fixed_center_is_not_merely_a_majority_classifier(self):
        """61/81 is far above 5/9; the gap is what design knowledge buys."""
        assert baselines.BASELINES_BY_KEY["fixed_center"].accuracy(
            UNIFORM
        ) > Fraction(5, 9)


# --------------------------------------------------------------------------
# 4. The step identity, over every valid crossing pair
# --------------------------------------------------------------------------


class TestStepIdentity:
    @pytest.mark.parametrize("j_hat", range(0, NO_CROSSING_INDEX + 1))
    @pytest.mark.parametrize("j_star", range(1, NO_CROSSING_INDEX + 1))
    def test_row_errors_equal_absolute_crossing_difference(self, j_hat, j_star):
        assert baselines.row_errors(j_hat, j_star) == abs(j_hat - j_star)

    @pytest.mark.parametrize("j_hat", range(0, NO_CROSSING_INDEX + 1))
    @pytest.mark.parametrize("j_star", range(1, NO_CROSSING_INDEX + 1))
    def test_accuracy_equals_one_minus_tce_over_nine(self, j_hat, j_star):
        errors = baselines.row_errors(j_hat, j_star)
        accuracy = Fraction(N_STRICTNESS - errors, N_STRICTNESS)
        assert accuracy == baselines.accuracy_from_tce(
            Fraction(baselines.crossing_error(j_hat, j_star))
        )

    def test_a_correct_crossing_makes_no_row_errors(self):
        for j in range(1, NO_CROSSING_INDEX + 1):
            assert baselines.row_errors(j, j) == 0

    def test_step_predictions_are_monotone_non_increasing(self):
        for j_hat in baselines.PREDICTION_SUPPORT:
            labels = baselines.step_predictions(j_hat)
            assert np.all(np.diff(labels) <= 0)

    def test_the_extremes_behave_as_named(self):
        assert baselines.step_predictions(0).sum() == 0
        assert baselines.step_predictions(NO_CROSSING_INDEX).sum() == N_STRICTNESS

    def test_the_identity_is_scoped_to_step_predictors(self):
        """A non-step predictor can break it -- which is why judges do."""
        truth = baselines.step_predictions(5)
        assert list(truth) == [1, 1, 1, 1, 1, 0, 0, 0, 0]
        # One dip at level 2, then it recovers. First "not met" is at index 2,
        # so the FormalCRRC crossing convention reads j_hat = 2 and TCE = 3 ...
        non_step = np.array([1, 1, 0, 1, 1, 0, 0, 0, 0])
        assert int(np.argmin(non_step)) == 2
        assert baselines.crossing_error(2, 5) == 3
        # ... yet it disagrees with the truth on a single row.
        assert int(np.sum(non_step != truth)) == 1
        # A step predictor at the same crossing would have made three errors.
        assert baselines.row_errors(2, 5) == 3


# --------------------------------------------------------------------------
# 5. Empirical frozen-partition results equal the analytic baselines
# --------------------------------------------------------------------------


def _load_block(path: Path):
    import pandas as pd

    frame = pd.read_parquet(path)
    wide = (
        frame.pivot(index="artifact_id", columns="strictness_index", values="formal_truth")
        .sort_index()
        .astype(int)
    )
    boundaries = (
        frame.groupby("artifact_id")["true_first_fail_index"].first().sort_index()
    )
    return wide.to_numpy(), boundaries.to_numpy().astype(int), frame


@pytest.fixture(scope="module", params=PARTITION_PATHS)
def frozen_partition(request, repo_root: Path):
    path = repo_root / request.param
    if not path.is_file():
        pytest.skip(f"{request.param} not present")
    truth, boundaries, frame = _load_block(path)
    return request.param, truth, boundaries, frame


class TestFrozenPartitions:
    def test_boundary_distribution_is_exactly_uniform(self, frozen_partition):
        _, _, boundaries, _ = frozen_partition
        values, counts = np.unique(boundaries, return_counts=True)
        assert list(values) == list(range(1, NO_CROSSING_INDEX + 1))
        assert len(set(counts)) == 1, f"boundary counts are not balanced: {counts}"

    def test_formal_labels_are_themselves_steps(self, frozen_partition):
        _, truth, boundaries, _ = frozen_partition
        expected = np.arange(N_STRICTNESS)[None, :] < boundaries[:, None]
        assert np.array_equal(truth.astype(bool), expected)

    def test_labels_recompute_from_the_predicate_definitions(self, frozen_partition):
        """§4.3: rebuild truth from the design fields, touching no stored value."""
        _, _, _, frame = frozen_partition
        recomputed = [
            predicates.formal_truth(r.family, r.latent_level, r.strictness_index)
            for r in frame.itertuples()
        ]
        assert recomputed == list(frame["formal_truth"].astype(int))

    def test_boundaries_recompute_from_the_predicate_definitions(
        self, frozen_partition
    ):
        _, _, _, frame = frozen_partition
        recomputed = [
            predicates.true_first_fail_index(r.family, r.latent_level)
            for r in frame.itertuples()
        ]
        assert recomputed == list(frame["true_first_fail_index"].astype(int))

    @pytest.mark.parametrize("key", ["fixed_center", "always_met", "always_not_met"])
    def test_deterministic_baselines_match_their_closed_forms(
        self, frozen_partition, key
    ):
        _, truth, boundaries, _ = frozen_partition
        baseline = baselines.BASELINES_BY_KEY[key]
        observed = baselines.empirical_baseline(baseline.j_hat, truth, boundaries)
        assert observed["mean_tce"] == pytest.approx(
            float(baseline.tce(UNIFORM)), abs=1e-12
        )
        assert observed["accuracy"] == pytest.approx(
            float(baseline.accuracy(UNIFORM)), abs=1e-12
        )

    def test_random_crossing_matches_its_exact_expectation(self, frozen_partition):
        """Compared over the whole support -- no draws are taken."""
        _, truth, boundaries, _ = frozen_partition
        observed = baselines.empirical_random_crossing(truth, boundaries)
        assert observed["mean_tce"] == pytest.approx(float(Fraction(80, 27)), abs=1e-12)
        assert observed["accuracy"] == pytest.approx(
            float(Fraction(163, 243)), abs=1e-12
        )

    def test_row_errors_equal_crossing_errors_on_real_data(self, frozen_partition):
        _, truth, boundaries, _ = frozen_partition
        for j_hat in baselines.PREDICTION_SUPPORT:
            observed = baselines.empirical_baseline(j_hat, truth, boundaries)
            assert observed["row_errors_total"] == observed["crossing_errors_total"]

    def test_met_rate_is_five_ninths(self, frozen_partition):
        _, truth, _, _ = frozen_partition
        assert Fraction(int(truth.sum()), int(truth.size)) == Fraction(5, 9)


class TestBaselineArtifacts:
    """The written outputs agree with the module they were produced from."""

    @staticmethod
    @pytest.fixture(scope="module")
    def summary(repo_root: Path):
        path = repo_root / "artifacts/poststudy_baselines/baseline_summary.json"
        if not path.is_file():
            pytest.skip("run scripts/compute_reference_baselines.py")
        return json.loads(path.read_text(encoding="utf-8"))

    def test_status_is_pass(self, summary):
        assert summary["status"] == "PASS"
        assert all(summary["checks"].values())

    def test_recorded_as_post_study_not_preregistered(self, summary):
        assert summary["preregistered"] is False
        assert summary["model_inference_performed"] is False

    def test_no_randomness_or_bootstrap_was_used(self, summary):
        assert summary["randomness_used"] is False
        assert summary["bootstrap_used"] is False

    def test_written_values_match_the_module(self, summary):
        for row in summary["analytic"]["baselines"]:
            baseline = baselines.BASELINES_BY_KEY[row["key"]]
            assert row["tce_exact"] == str(baseline.tce(UNIFORM))
            assert row["accuracy_exact"] == str(baseline.accuracy(UNIFORM))

    def test_every_partition_passed(self, summary):
        assert len(summary["partitions"]) == len(PARTITION_PATHS)
        assert all(p["status"] == "PASS" for p in summary["partitions"])

    def test_no_judge_view_is_a_step_predictor(self, summary):
        """The reason judge accuracy is not recoverable from judge TCE."""
        scored = [r for r in summary["judge_comparison"]["rows"] if "accuracy" in r]
        assert scored, "expected judge rows carrying accuracy"
        assert not any(r["is_step_predictor"] for r in scored)


# --------------------------------------------------------------------------
# 6. No frozen Day-1/2/3 file changed
# --------------------------------------------------------------------------


class TestFrozenFilesUnchanged:
    """The audit is additive. Every prior hash must still verify."""

    def test_day1_and_day2_files_still_hash_as_recorded(self, repo_root: Path):
        manifest = json.loads(
            (repo_root / PRIOR_MANIFEST).read_text(encoding="utf-8")
        )
        changed = [
            relative
            for relative, digest in manifest["files"].items()
            if provenance.sha256_file(repo_root / relative) != digest
        ]
        assert changed == [], f"frozen prior files changed: {changed}"

    def test_day3_files_still_hash_as_recorded(self, repo_root: Path):
        manifest = json.loads((repo_root / DAY3_MANIFEST).read_text(encoding="utf-8"))
        changed = [
            entry["path"]
            for entry in manifest["files"].values()
            if entry["present"]
            and provenance.sha256_file(repo_root / entry["path"]) != entry["sha256"]
        ]
        assert changed == [], f"frozen Day-3 files changed: {changed}"

    def test_prior_combined_digest_is_unchanged(self, repo_root: Path):
        manifest = json.loads(
            (repo_root / PRIOR_MANIFEST).read_text(encoding="utf-8")
        )
        combined = provenance.sha256_text(
            "\n".join(
                f"{relative}:{digest}"
                for relative, digest in sorted(manifest["files"].items())
            )
        )
        assert combined == manifest["combined_sha256"]

    def test_baseline_outputs_live_only_in_their_own_directory(self, repo_root: Path):
        manifest_path = repo_root / "artifacts/poststudy_baselines/integrity_manifest.json"
        if not manifest_path.is_file():
            pytest.skip("run scripts/compute_reference_baselines.py")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in manifest["files"].values():
            assert entry["path"].startswith("artifacts/poststudy_baselines/")

    def test_the_baseline_script_writes_nowhere_else(self, repo_root: Path):
        """No Day-1/2/3 path may appear as a write target in the new code."""
        for relative in (
            "scripts/compute_reference_baselines.py",
            "src/formalcrrc/baselines.py",
        ):
            source = (repo_root / relative).read_text(encoding="utf-8")
            for line in source.splitlines():
                if "write_json" in line or ".open(" in line or "to_csv" in line:
                    for frozen in ("artifacts/day1", "artifacts/day2", "artifacts/day3",
                                   "figures/day"):
                        assert frozen not in line, f"{relative}: {line.strip()}"
