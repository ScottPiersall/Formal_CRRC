"""Day-3 partition integrity and disjointness from every prior evaluated set."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from formalcrrc import dataset as ds
from formalcrrc import day3_config as d3c
from formalcrrc import predicates, prompts
from formalcrrc.config import (
    FAMILIES,
    FAMILY_NUMERIC_TOLERANCE,
    LIST_SIZE,
    MARKERS_PER_RESPONSE,
    N_ARTIFACTS,
    N_INSTANCES,
    N_LATENT_LEVELS,
    N_ROWS,
    N_STRICTNESS,
)


def _pairs(frame):
    subset = frame[frame["family"] == FAMILY_NUMERIC_TOLERANCE].drop_duplicates(
        "artifact_id"
    )
    return frozenset(
        zip(subset["target_value"].astype(int), subset["reported_value"].astype(int))
    )


def _prompt_set(frame):
    return {
        prompts.build_user_message(r, c)
        for r, c in zip(frame["rubric_text"], frame["candidate_response"], strict=True)
    }


class TestDimensions:
    def test_artifact_count(self, day3_test):
        assert day3_test["artifact_id"].nunique() == N_ARTIFACTS == 324

    def test_row_count(self, day3_test):
        assert len(day3_test) == N_ROWS == 2916

    def test_nine_thresholds_per_artifact(self, day3_test):
        for artifact_id, group in day3_test.groupby("artifact_id"):
            assert sorted(group["strictness_index"]) == list(range(N_STRICTNESS))

    def test_prompt_ids_unique(self, day3_test):
        assert day3_test["prompt_id"].is_unique

    def test_seed_is_45(self, day3_test):
        assert set(day3_test["seed"]) == {45}

    def test_all_latent_levels_equally_represented(self, day3_test):
        cells = Counter(zip(day3_test["family"], day3_test["latent_level"]))
        assert len(cells) == len(FAMILIES) * N_LATENT_LEVELS
        assert set(cells.values()) == {N_INSTANCES * N_STRICTNESS}

    def test_namespaced_artifact_ids(self, day3_test):
        assert day3_test["artifact_id"].str.startswith(f"{d3c.PARTITION_TAG}-").all()


class TestFormalTruth:
    def test_labels_recompute(self, day3_test):
        for row in day3_test.itertuples(index=False):
            assert row.formal_truth == predicates.formal_truth(
                row.family, row.latent_level, row.strictness_index
            )

    def test_labels_monotone(self, day3_test):
        for artifact_id, group in day3_test.groupby("artifact_id"):
            truth = group.sort_values("strictness_index")["formal_truth"].tolist()
            assert truth == sorted(truth, reverse=True), artifact_id

    def test_first_fail_index_matches_curve(self, day3_test):
        for artifact_id, group in day3_test.groupby("artifact_id"):
            ordered = group.sort_values("strictness_index")
            truth = ordered["formal_truth"].tolist()
            zeros = [s for s, y in enumerate(truth) if y == 0]
            expected = zeros[0] if zeros else N_STRICTNESS
            assert set(ordered["true_first_fail_index"]) == {expected}, artifact_id

    def test_boundaries_span_one_to_nine(self, day3_test):
        boundaries = set(day3_test["true_first_fail_index"])
        assert boundaries == set(range(1, N_STRICTNESS + 1))

    def test_numeric_error_equals_latent_level(self, day3_test):
        numeric = day3_test[day3_test["family"] == FAMILY_NUMERIC_TOLERANCE]
        for row in numeric.itertuples(index=False):
            assert abs(int(row.reported_value) - int(row.target_value)) == (
                row.latent_level
            )


class TestCandidateAndRubricControl:
    def test_candidate_byte_identical_across_thresholds(self, day3_test):
        for artifact_id, group in day3_test.groupby("artifact_id"):
            assert group["candidate_response"].nunique() == 1, artifact_id

    def test_candidate_hash_correct(self, day3_test):
        for artifact_id, group in day3_test.groupby("artifact_id"):
            assert group["candidate_sha256"].iloc[0] == ds.sha256_text(
                group["candidate_response"].iloc[0]
            )

    def test_normalised_rubrics_identical_within_artifact(self, day3_test):
        for artifact_id, group in day3_test.groupby("artifact_id"):
            family = group["family"].iloc[0]
            normalized = {
                predicates.normalize_rubric(family, t) for t in group["rubric_text"]
            }
            assert len(normalized) == 1, artifact_id

    def test_rubric_varies_with_threshold(self, day3_test):
        for artifact_id, group in day3_test.groupby("artifact_id"):
            assert group["rubric_sha256"].nunique() == N_STRICTNESS, artifact_id

    def test_embedded_threshold_matches_column(self, day3_test):
        for row in day3_test.itertuples(index=False):
            assert (
                predicates.extract_threshold(row.family, row.rubric_text)
                == row.raw_threshold
            )

    def test_marker_response_length_constant(self, day3_test):
        markers = day3_test[day3_test["family"] != FAMILY_NUMERIC_TOLERANCE]
        for payload in markers["response_markers_json"].unique():
            assert payload.count("MK-") == MARKERS_PER_RESPONSE

    def test_item_lists_full_size(self, day3_test):
        markers = day3_test[day3_test["family"] != FAMILY_NUMERIC_TOLERANCE]
        for payload in markers["list_items_json"].unique():
            assert payload.count("MK-") == LIST_SIZE


class TestDisjointnessFromPriors:
    def test_no_shared_artifact_ids(
        self, day3_test, formal_dataset, day2_calibration, day2_test
    ):
        day3_ids = set(day3_test["artifact_id"])
        for prior in (formal_dataset, day2_calibration, day2_test):
            assert day3_ids & set(prior["artifact_id"]) == set()

    def test_no_shared_rendered_prompts(
        self, day3_test, formal_dataset, day2_calibration, day2_test
    ):
        day3_prompts = _prompt_set(day3_test)
        for prior in (formal_dataset, day2_calibration, day2_test):
            assert day3_prompts & _prompt_set(prior) == set()

    def test_family_c_target_pairs_disjoint(
        self, day3_test, formal_dataset, day2_calibration, day2_test
    ):
        day3_pairs = _pairs(day3_test)
        for prior in (formal_dataset, day2_calibration, day2_test):
            assert day3_pairs & _pairs(prior) == frozenset()

    @pytest.mark.parametrize("family", ["coverage", "max_violation"])
    def test_marker_families_share_no_candidate(
        self, family, day3_test, formal_dataset, day2_calibration, day2_test
    ):
        def candidates(frame):
            subset = frame[frame["family"] == family]
            return set(subset.drop_duplicates("artifact_id")["candidate_sha256"])

        day3_candidates = candidates(day3_test)
        for prior in (formal_dataset, day2_calibration, day2_test):
            assert day3_candidates & candidates(prior) == set()

    def test_family_c_coincidences_are_bounded(self, day3_test, day2_test):
        """Expected under a bounded integer range; not reused items."""

        def candidates(frame):
            subset = frame[frame["family"] == FAMILY_NUMERIC_TOLERANCE]
            return set(subset.drop_duplicates("artifact_id")["candidate_sha256"])

        overlap = candidates(day3_test) & candidates(day2_test)
        assert 0 < len(overlap) < 108
        assert _pairs(day3_test) & _pairs(day2_test) == frozenset()


class TestNoLeakage:
    def test_artifact_id_never_in_prompt(self, day3_test):
        for row in day3_test.itertuples(index=False):
            message = prompts.build_user_message(row.rubric_text, row.candidate_response)
            assert row.artifact_id not in message

    def test_no_partition_wording_in_prompt(self, day3_test):
        for row in day3_test.head(400).itertuples(index=False):
            lowered = prompts.build_user_message(
                row.rubric_text, row.candidate_response
            ).lower()
            for banned in ("calibration", "partition", "held-out", "oracle", "day 3"):
                assert banned not in lowered, banned

    def test_prompt_template_matches_prior_days(self):
        assert (
            prompts.prompt_template_sha256()
            == "e5f3c5ba3dfd1ccaca0655307d8ad254598a17184c7ae40dfde63c1f660cae14"
        )


class TestDeterminismAndFreeze:
    def test_rebuild_is_identical(self, day3_test, day2_calibration, day2_test):
        day1 = ds.build_dataset()
        forbidden = set()
        for frame in (day1, day2_calibration, day2_test):
            forbidden |= _pairs(frame)
        rebuilt = ds.build_dataset(
            seed=d3c.SEED,
            partition=d3c.PARTITION_TAG,
            forbidden_pairs=frozenset(forbidden),
            enforce_unique_targets=True,
        )
        assert ds.dataset_content_hash(rebuilt) == ds.dataset_content_hash(day3_test)

    def test_content_hash_differs_from_prior_partitions(
        self, day3_test, formal_dataset, day2_calibration, day2_test
    ):
        day3_hash = ds.dataset_content_hash(day3_test)
        for prior in (formal_dataset, day2_calibration, day2_test):
            assert day3_hash != ds.dataset_content_hash(prior)

    def test_frozen_manifest_matches_rebuild(self, repo_root, day3_test):
        manifest_path = repo_root / d3c.DATASET_MANIFEST_PATH
        if not manifest_path.is_file():
            pytest.skip("Day-3 dataset not generated yet")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["hashes"]["dataset_content_sha256"] == (
            ds.dataset_content_hash(day3_test)
        )
        assert manifest["seed"] == 45
