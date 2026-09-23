"""Day-2 CALIBRATION and TEST partition integrity, and their disjointness."""

from __future__ import annotations

from collections import Counter

import pytest

from formalcrrc import dataset as ds
from formalcrrc import day2_config as d2c
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


@pytest.fixture(params=["calibration", "test"])
def partition(request, day2_calibration, day2_test):
    return {"calibration": day2_calibration, "test": day2_test}[request.param]


class TestDimensions:
    def test_artifact_count(self, partition):
        assert partition["artifact_id"].nunique() == N_ARTIFACTS == 324

    def test_row_count(self, partition):
        assert len(partition) == N_ROWS == 2916

    def test_nine_thresholds_per_artifact(self, partition):
        for artifact_id, group in partition.groupby("artifact_id"):
            assert sorted(group["strictness_index"]) == list(range(N_STRICTNESS))

    def test_prompt_ids_unique(self, partition):
        assert partition["prompt_id"].is_unique

    def test_all_latent_levels_equally_represented(self, partition):
        cells = Counter(zip(partition["family"], partition["latent_level"]))
        assert len(cells) == len(FAMILIES) * N_LATENT_LEVELS
        assert set(cells.values()) == {N_INSTANCES * N_STRICTNESS}

    def test_seeds(self, day2_calibration, day2_test):
        assert set(day2_calibration["seed"]) == {43}
        assert set(day2_test["seed"]) == {44}

    def test_total_day2_prompts_per_judge(self, day2_calibration, day2_test):
        assert len(day2_calibration) + len(day2_test) == 5832


class TestFormalTruth:
    def test_labels_recompute(self, partition):
        for row in partition.itertuples(index=False):
            assert row.formal_truth == predicates.formal_truth(
                row.family, row.latent_level, row.strictness_index
            )

    def test_labels_monotone(self, partition):
        for artifact_id, group in partition.groupby("artifact_id"):
            truth = group.sort_values("strictness_index")["formal_truth"].tolist()
            assert truth == sorted(truth, reverse=True), artifact_id

    def test_first_fail_index_matches_curve(self, partition):
        for artifact_id, group in partition.groupby("artifact_id"):
            ordered = group.sort_values("strictness_index")
            truth = ordered["formal_truth"].tolist()
            zeros = [s for s, y in enumerate(truth) if y == 0]
            expected = zeros[0] if zeros else N_STRICTNESS
            assert set(ordered["true_first_fail_index"]) == {expected}, artifact_id

    def test_numeric_error_equals_latent_level(self, partition):
        numeric = partition[partition["family"] == FAMILY_NUMERIC_TOLERANCE]
        for row in numeric.itertuples(index=False):
            assert abs(int(row.reported_value) - int(row.target_value)) == (
                row.latent_level
            )


class TestCandidateImmutability:
    def test_candidate_byte_identical_across_thresholds(self, partition):
        for artifact_id, group in partition.groupby("artifact_id"):
            assert group["candidate_response"].nunique() == 1, artifact_id

    def test_candidate_hash_constant_and_correct(self, partition):
        for artifact_id, group in partition.groupby("artifact_id"):
            assert group["candidate_sha256"].nunique() == 1
            assert group["candidate_sha256"].iloc[0] == ds.sha256_text(
                group["candidate_response"].iloc[0]
            )

    def test_marker_responses_have_constant_length(self, partition):
        markers = partition[partition["family"] != FAMILY_NUMERIC_TOLERANCE]
        for payload in markers["response_markers_json"].unique():
            assert payload.count("MK-") == MARKERS_PER_RESPONSE


class TestRubricControl:
    def test_normalised_rubrics_identical_within_artifact(self, partition):
        for artifact_id, group in partition.groupby("artifact_id"):
            family = group["family"].iloc[0]
            normalized = {
                predicates.normalize_rubric(family, t) for t in group["rubric_text"]
            }
            assert len(normalized) == 1, artifact_id

    def test_rubric_text_varies_with_threshold(self, partition):
        for artifact_id, group in partition.groupby("artifact_id"):
            assert group["rubric_sha256"].nunique() == N_STRICTNESS, artifact_id

    def test_embedded_threshold_matches_column(self, partition):
        for row in partition.itertuples(index=False):
            assert (
                predicates.extract_threshold(row.family, row.rubric_text)
                == row.raw_threshold
            )

    def test_item_lists_are_full_size(self, partition):
        markers = partition[partition["family"] != FAMILY_NUMERIC_TOLERANCE]
        for payload in markers["list_items_json"].unique():
            assert payload.count("MK-") == LIST_SIZE


class TestPartitionDisjointness:
    def test_no_shared_artifact_ids(self, day2_calibration, day2_test, formal_dataset):
        cal = set(day2_calibration["artifact_id"])
        test = set(day2_test["artifact_id"])
        day1 = set(formal_dataset["artifact_id"])
        assert cal & test == set()
        assert cal & day1 == set()
        assert test & day1 == set()

    def test_no_shared_rendered_prompts(
        self, day2_calibration, day2_test, formal_dataset
    ):
        cal, test, day1 = (
            _prompt_set(day2_calibration),
            _prompt_set(day2_test),
            _prompt_set(formal_dataset),
        )
        assert cal & test == set()
        assert cal & day1 == set()
        assert test & day1 == set()

    def test_family_c_target_pairs_disjoint(
        self, day2_calibration, day2_test, formal_dataset
    ):
        cal, test, day1 = (
            _pairs(day2_calibration),
            _pairs(day2_test),
            _pairs(formal_dataset),
        )
        assert cal & test == frozenset()
        assert cal & day1 == frozenset()
        assert test & day1 == frozenset()

    @pytest.mark.parametrize("family", ["coverage", "max_violation"])
    def test_marker_families_share_no_candidate_response(
        self, family, day2_calibration, day2_test, formal_dataset
    ):
        def candidates(frame):
            subset = frame[frame["family"] == family]
            return set(subset.drop_duplicates("artifact_id")["candidate_sha256"])

        cal, test, day1 = (
            candidates(day2_calibration),
            candidates(day2_test),
            candidates(formal_dataset),
        )
        assert cal & test == set()
        assert cal & day1 == set()
        assert test & day1 == set()

    def test_family_c_candidate_coincidence_is_bounded(
        self, day2_calibration, day2_test
    ):
        """A family-C candidate is fixed by its reported value alone.

        The reported value is a bounded integer, so byte-identical candidate
        responses across independently seeded partitions are expected. They are
        not reused items -- the (target, reported) pairs and the rendered
        prompts are disjoint -- but the count must stay small enough to be a
        coincidence rather than a construction error.
        """

        def candidates(frame):
            subset = frame[frame["family"] == FAMILY_NUMERIC_TOLERANCE]
            return set(subset.drop_duplicates("artifact_id")["candidate_sha256"])

        overlap = candidates(day2_calibration) & candidates(day2_test)
        assert 0 < len(overlap) < 108
        assert _pairs(day2_calibration) & _pairs(day2_test) == frozenset()


class TestNoPartitionLeakage:
    def test_artifact_id_never_reaches_a_prompt(self, partition):
        for row in partition.itertuples(index=False):
            message = prompts.build_user_message(row.rubric_text, row.candidate_response)
            assert row.artifact_id not in message

    def test_partition_wording_never_reaches_a_prompt(self, partition):
        for row in partition.head(400).itertuples(index=False):
            lowered = prompts.build_user_message(
                row.rubric_text, row.candidate_response
            ).lower()
            for banned in ("calibration", "partition", "held-out", "test set"):
                assert banned not in lowered, banned

    def test_prompt_template_is_the_day1_template(self):
        assert (
            prompts.prompt_template_sha256()
            == "e5f3c5ba3dfd1ccaca0655307d8ad254598a17184c7ae40dfde63c1f660cae14"
        )


class TestDeterminism:
    def test_calibration_rebuilds_identically(self, day2_calibration):
        forbidden = _pairs(ds.build_dataset())
        rebuilt = ds.build_dataset(
            seed=d2c.SEED_CALIBRATION,
            partition=d2c.PARTITION_CALIBRATION,
            forbidden_pairs=forbidden,
            enforce_unique_targets=True,
        )
        assert ds.dataset_content_hash(rebuilt) == ds.dataset_content_hash(
            day2_calibration
        )

    def test_partitions_have_different_content_hashes(
        self, day2_calibration, day2_test
    ):
        assert ds.dataset_content_hash(day2_calibration) != ds.dataset_content_hash(
            day2_test
        )

    def test_frozen_files_match_the_rebuild(self, repo_root, day2_calibration, day2_test):
        import json

        for path_key, frame in (
            (d2c.CALIBRATION_MANIFEST_PATH, day2_calibration),
            (d2c.TEST_MANIFEST_PATH, day2_test),
        ):
            manifest_path = repo_root / path_key
            if not manifest_path.is_file():
                pytest.skip("Day-2 datasets not generated yet")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            assert manifest["hashes"]["dataset_content_sha256"] == (
                ds.dataset_content_hash(frame)
            )
