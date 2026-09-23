"""Dataset dimensions, candidate immutability, rubric control, and an
independent recomputation of every formal label from the rendered text alone.
"""

from __future__ import annotations

import json
import re
from collections import Counter

import pytest

from formalcrrc import dataset as ds
from formalcrrc import predicates
from formalcrrc.config import (
    FAMILIES,
    FAMILY_COVERAGE,
    FAMILY_MAX_VIOLATION,
    FAMILY_NUMERIC_TOLERANCE,
    LIST_SIZE,
    MARKERS_PER_RESPONSE,
    N_ARTIFACTS,
    N_INSTANCES,
    N_LATENT_LEVELS,
    N_ROWS,
    N_STRICTNESS,
    SEED,
)

MARKER_LINE = re.compile(r"^- (MK-[A-Z]+)$", re.MULTILINE)


class TestDimensions:
    def test_exact_artifact_count(self, formal_dataset):
        assert formal_dataset["artifact_id"].nunique() == N_ARTIFACTS == 324

    def test_exact_threshold_pair_count(self, formal_dataset):
        assert len(formal_dataset) == N_ROWS == 2916

    def test_nine_thresholds_per_artifact_each_exactly_once(self, formal_dataset):
        for artifact_id, group in formal_dataset.groupby("artifact_id"):
            levels = sorted(group["strictness_index"].tolist())
            assert levels == list(range(N_STRICTNESS)), artifact_id

    def test_full_family_by_latent_by_instance_crossing(self, formal_dataset):
        cells = Counter(
            zip(
                formal_dataset["family"],
                formal_dataset["latent_level"],
                formal_dataset["instance_index"],
            )
        )
        assert len(cells) == N_ARTIFACTS
        assert set(cells.values()) == {N_STRICTNESS}
        for family in FAMILIES:
            for latent in range(N_LATENT_LEVELS):
                instances = {
                    key[2] for key in cells if key[0] == family and key[1] == latent
                }
                assert instances == set(range(N_INSTANCES))

    def test_prompt_ids_are_unique(self, formal_dataset):
        assert formal_dataset["prompt_id"].is_unique

    def test_seed_recorded_on_every_row(self, formal_dataset):
        assert set(formal_dataset["seed"]) == {SEED}


class TestCandidateImmutability:
    """The evaluated artifact must not move while the threshold does."""

    def test_candidate_response_is_byte_identical_across_thresholds(
        self, formal_dataset
    ):
        for artifact_id, group in formal_dataset.groupby("artifact_id"):
            responses = set(group["candidate_response"])
            assert len(responses) == 1, artifact_id

    def test_candidate_hash_is_constant_and_correct(self, formal_dataset):
        for artifact_id, group in formal_dataset.groupby("artifact_id"):
            hashes = set(group["candidate_sha256"])
            assert len(hashes) == 1, artifact_id
            expected = ds.sha256_text(group["candidate_response"].iloc[0])
            assert hashes.pop() == expected

    def test_marker_responses_have_constant_length(self, formal_dataset):
        """Response length carries no information about the latent level."""
        markers = formal_dataset[
            formal_dataset["family"].isin([FAMILY_COVERAGE, FAMILY_MAX_VIOLATION])
        ]
        counts = {
            len(MARKER_LINE.findall(text))
            for text in markers["candidate_response"].unique()
        }
        assert counts == {MARKERS_PER_RESPONSE}


class TestRubricControl:
    """Rubric variants for an artifact may differ only in the threshold field."""

    def test_normalised_rubrics_are_identical_within_an_artifact(
        self, formal_dataset
    ):
        for artifact_id, group in formal_dataset.groupby("artifact_id"):
            family = group["family"].iloc[0]
            normalized = {
                predicates.normalize_rubric(family, text)
                for text in group["rubric_text"]
            }
            assert len(normalized) == 1, artifact_id

    def test_normalised_rubric_hash_is_constant_within_an_artifact(
        self, formal_dataset
    ):
        for artifact_id, group in formal_dataset.groupby("artifact_id"):
            assert group["rubric_normalized_sha256"].nunique() == 1, artifact_id

    def test_rubric_text_actually_differs_between_thresholds(self, formal_dataset):
        for artifact_id, group in formal_dataset.groupby("artifact_id"):
            assert group["rubric_sha256"].nunique() == N_STRICTNESS, artifact_id

    def test_embedded_threshold_matches_the_raw_threshold_column(
        self, formal_dataset
    ):
        for row in formal_dataset.itertuples(index=False):
            found = predicates.extract_threshold(row.family, row.rubric_text)
            assert found == row.raw_threshold, row.prompt_id

    def test_recorded_rubric_hashes_are_correct(self, formal_dataset):
        for row in formal_dataset.itertuples(index=False):
            assert row.rubric_sha256 == ds.sha256_text(row.rubric_text)
            normalized = predicates.normalize_rubric(row.family, row.rubric_text)
            assert row.rubric_normalized_sha256 == ds.sha256_text(normalized)


class TestFormalTruth:
    def test_labels_are_monotone_in_strictness(self, formal_dataset):
        for artifact_id, group in formal_dataset.groupby("artifact_id"):
            truth = group.sort_values("strictness_index")["formal_truth"].tolist()
            assert truth == sorted(truth, reverse=True), artifact_id

    def test_labels_reproduce_the_predicate(self, formal_dataset):
        for row in formal_dataset.itertuples(index=False):
            assert row.formal_truth == predicates.formal_truth(
                row.family, row.latent_level, row.strictness_index
            )

    def test_first_fail_index_matches_the_stored_curve(self, formal_dataset):
        for artifact_id, group in formal_dataset.groupby("artifact_id"):
            ordered = group.sort_values("strictness_index")
            truth = ordered["formal_truth"].tolist()
            zeros = [s for s, y in enumerate(truth) if y == 0]
            expected = zeros[0] if zeros else N_STRICTNESS
            assert set(ordered["true_first_fail_index"]) == {expected}, artifact_id

    def test_labels_are_never_all_zero_or_ambiguous(self, formal_dataset):
        for artifact_id, group in formal_dataset.groupby("artifact_id"):
            truth = group["formal_truth"].tolist()
            assert set(truth) <= {0, 1}
            assert truth.count(1) >= 1, artifact_id


class TestIndependentRecomputation:
    """Recompute every label from the rendered strings, ignoring stored columns.

    This is deliberately a second implementation: it parses the item list out of
    the rubric and the markers out of the candidate response, so it would fail if
    the rendered text ever disagreed with the arithmetic label.
    """

    @staticmethod
    def _rubric_items(rubric_text: str) -> list[str]:
        head = rubric_text.split("\n\n", 1)[0]
        return MARKER_LINE.findall(head)

    @staticmethod
    def _response_markers(candidate_response: str) -> list[str]:
        return MARKER_LINE.findall(candidate_response)

    def test_recomputed_labels_match_for_every_row(self, formal_dataset):
        target_re = re.compile(r"^Target value: (\d+)$", re.MULTILINE)
        value_re = re.compile(r"is (-?\d+)\.$", re.MULTILINE)

        for row in formal_dataset.itertuples(index=False):
            threshold = predicates.extract_threshold(row.family, row.rubric_text)
            if row.family == FAMILY_NUMERIC_TOLERANCE:
                target = int(target_re.search(row.rubric_text).group(1))
                reported = int(value_re.search(row.candidate_response).group(1))
                recomputed = int(abs(reported - target) <= threshold)
            else:
                items = set(self._rubric_items(row.rubric_text))
                present = set(self._response_markers(row.candidate_response))
                overlap = len(items & present)
                if row.family == FAMILY_COVERAGE:
                    recomputed = int(overlap >= threshold)
                else:
                    recomputed = int(overlap <= threshold)
            assert recomputed == row.formal_truth, row.prompt_id

    def test_overlap_equals_the_latent_level(self, formal_dataset):
        markers = formal_dataset[
            formal_dataset["family"].isin([FAMILY_COVERAGE, FAMILY_MAX_VIOLATION])
        ]
        for row in markers.itertuples(index=False):
            items = set(self._rubric_items(row.rubric_text))
            present = set(self._response_markers(row.candidate_response))
            assert len(items) == LIST_SIZE
            assert len(items & present) == row.latent_level, row.prompt_id

    def test_numeric_absolute_error_equals_the_latent_level(self, formal_dataset):
        numeric = formal_dataset[
            formal_dataset["family"] == FAMILY_NUMERIC_TOLERANCE
        ]
        for row in numeric.itertuples(index=False):
            assert abs(int(row.reported_value) - int(row.target_value)) == (
                row.latent_level
            )

    def test_numeric_errors_use_both_signs(self, formal_dataset):
        numeric = formal_dataset[
            (formal_dataset["family"] == FAMILY_NUMERIC_TOLERANCE)
            & (formal_dataset["latent_level"] > 0)
        ]
        for latent, group in numeric.groupby("latent_level"):
            signs = set(group["error_sign"].dropna().astype(int))
            assert signs == {-1, 1}, latent

    def test_numeric_targets_vary_within_a_latent_level(self, formal_dataset):
        numeric = formal_dataset[
            formal_dataset["family"] == FAMILY_NUMERIC_TOLERANCE
        ]
        for latent, group in numeric.groupby("latent_level"):
            targets = set(group["target_value"].astype(int))
            assert len(targets) > 1, latent

    def test_marker_pools_are_internally_disjoint(self, formal_dataset):
        markers = formal_dataset[
            formal_dataset["family"].isin([FAMILY_COVERAGE, FAMILY_MAX_VIOLATION])
        ]
        for row in markers.itertuples(index=False):
            items = json.loads(row.list_items_json)
            emitted = json.loads(row.response_markers_json)
            assert len(set(items)) == LIST_SIZE
            assert len(set(emitted)) == MARKERS_PER_RESPONSE
            off_list = set(emitted) - set(items)
            assert len(off_list) == MARKERS_PER_RESPONSE - row.latent_level


class TestDeterminism:
    def test_rebuilding_reproduces_the_identical_content_hash(self, formal_dataset):
        rebuilt = ds.build_dataset()
        assert ds.dataset_content_hash(rebuilt) == ds.dataset_content_hash(
            formal_dataset
        )

    def test_single_artifact_construction_is_reproducible(self):
        for family in FAMILIES:
            first = ds.build_artifact(family, 4, 3)
            second = ds.build_artifact(family, 4, 3)
            assert first == second

    def test_content_hash_detects_a_single_character_change(self, formal_dataset):
        tampered = formal_dataset.copy()
        original = tampered.loc[0, "candidate_response"]
        tampered.loc[0, "candidate_response"] = original + " "
        assert ds.dataset_content_hash(tampered) != ds.dataset_content_hash(
            formal_dataset
        )

    def test_rng_provenance_recorded(self, formal_dataset):
        assert formal_dataset["rng_provenance"].str.startswith("numpy.").all()


class TestNoModelInvolvement:
    """Ground truth must not depend on any language model or external service."""

    def test_dataset_module_imports_no_model_libraries(self):
        source = (ds.__file__,)
        text = open(source[0], encoding="utf-8").read()
        for forbidden in ("transformers", "torch", "openai", "requests", "urllib"):
            assert forbidden not in text, forbidden

    def test_predicates_module_imports_no_model_libraries(self):
        text = open(predicates.__file__, encoding="utf-8").read()
        for forbidden in ("transformers", "torch", "openai", "requests", "urllib"):
            assert forbidden not in text, forbidden
