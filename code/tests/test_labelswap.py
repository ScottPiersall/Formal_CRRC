"""The label-swap counterbalance intervention.

Preregistered robustness study (`docs/PREREGISTRATION_LABEL_SWAP.md`), frozen
before any swapped-label inference. These tests run before the full panel and
guard the one thing that can silently ruin the experiment: getting the canonical
sign wrong, so that the swapped condition is analysed as if it were the original.

The seven blocks are the preregistered Phase-6 requirements A-G.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from formalcrrc import day3, labelswap, predicates, prompts, provenance
from formalcrrc import labelswap_config as lsc
from formalcrrc.config import (
    LABEL_MET,
    LABEL_NOT_MET,
    N_STRICTNESS,
    NO_CROSSING_INDEX,
    PROMPT_TEMPLATE,
)
from formalcrrc.day2_bootstrap import crossing_index

# --------------------------------------------------------------------------
# A. Semantic sign canonicalization
# --------------------------------------------------------------------------


class TestSemanticSignCanonicalization:
    def test_original_a_high_means_met(self):
        """A high, B low -> canonical margin positive under the original map."""
        margin = labelswap.original_margin(logit_a=5.0, logit_b=-3.0)
        assert margin > 0
        assert labelswap.semantic_decision(np.array([margin]))[0] == 1

    def test_swapped_b_high_means_met(self):
        """B high, A low -> canonical margin positive under the swapped map."""
        margin = labelswap.swapped_margin(logit_a=-3.0, logit_b=5.0)
        assert margin > 0
        assert labelswap.semantic_decision(np.array([margin]))[0] == 1

    def test_both_conditions_agree_on_the_same_semantic_evidence(self):
        """Identical evidence for 'met', expressed through opposite tokens."""
        original = labelswap.original_margin(logit_a=5.0, logit_b=-3.0)
        swapped = labelswap.swapped_margin(logit_a=-3.0, logit_b=5.0)
        assert original == swapped == 8.0

    def test_swapped_margin_is_not_the_original_difference(self):
        """The whole point: S_B - S_A, never S_A - S_B."""
        a, b = 2.0, -1.0
        assert labelswap.swapped_margin(a, b) == -labelswap.original_margin(a, b)
        assert labelswap.swapped_margin(a, b) == b - a

    def test_a_high_under_swapped_mapping_means_not_met(self):
        """Under the swapped instruction, A is the 'not met' token."""
        margin = labelswap.swapped_margin(logit_a=6.0, logit_b=1.0)
        assert margin < 0
        assert labelswap.semantic_decision(np.array([margin]))[0] == 0

    def test_canonical_margin_dispatches_on_condition(self):
        a, b = 4.0, 1.0
        assert labelswap.canonical_margin(a, b, lsc.CONDITION_ORIGINAL) == 3.0
        assert labelswap.canonical_margin(a, b, lsc.CONDITION_SWAPPED) == -3.0

    def test_unknown_condition_is_rejected(self):
        with pytest.raises(ValueError):
            labelswap.canonical_margin(1.0, 2.0, "sideways")

    def test_zero_margin_counts_as_met_in_both_conditions(self):
        assert labelswap.semantic_decision(np.array([0.0]))[0] == 1

    def test_vectorised_margins_match_scalar(self):
        a = np.array([1.0, -2.0, 0.0])
        b = np.array([0.0, 3.0, 0.0])
        assert np.allclose(labelswap.swapped_margin(a, b), b - a)
        assert np.allclose(labelswap.original_margin(a, b), a - b)


class TestSwappedTokenizationSemantics:
    """The tokenisation object must carry the reversed assignment."""

    class _FakeTokenizer:
        """Minimal stand-in: 'A' -> id 10, 'B' -> id 11 as continuations."""

        def encode(self, text, add_special_tokens=False):
            base = "PROMPT"
            if text == base:
                return [1, 2, 3]
            if text == base + "A":
                return [1, 2, 3, 10]
            if text == base + "B":
                return [1, 2, 3, 11]
            if text == "A":
                return [10]
            if text == "B":
                return [11]
            raise AssertionError(f"unexpected encode({text!r})")

    def test_met_token_is_b_under_swap(self):
        tokenization = labelswap.resolve_swapped_label_tokenization(
            self._FakeTokenizer(), "PROMPT"
        )
        assert tokenization.label_met == LABEL_NOT_MET == "B"
        assert tokenization.met_token_ids == (11,)

    def test_not_met_token_is_a_under_swap(self):
        tokenization = labelswap.resolve_swapped_label_tokenization(
            self._FakeTokenizer(), "PROMPT"
        )
        assert tokenization.label_not_met == LABEL_MET == "A"
        assert tokenization.not_met_token_ids == (10,)

    def test_mirrors_a_day3_record(self):
        tokenization = labelswap.resolve_swapped_label_tokenization(
            self._FakeTokenizer(), "PROMPT"
        )
        day3_record = {
            "label_met": "A",
            "label_not_met": "B",
            "met_token_ids": [10],
            "not_met_token_ids": [11],
            "single_token_equivalence": True,
            "scoring_method": "single_token_next_logit",
        }
        result = labelswap.tokenization_is_day3_mirror(tokenization, day3_record)
        assert result["status"] == "PASS", result

    def test_a_non_mirror_is_rejected(self):
        tokenization = labelswap.resolve_swapped_label_tokenization(
            self._FakeTokenizer(), "PROMPT"
        )
        wrong = {
            "label_met": "A",
            "label_not_met": "B",
            "met_token_ids": [99],
            "not_met_token_ids": [11],
            "single_token_equivalence": True,
            "scoring_method": "single_token_next_logit",
        }
        assert labelswap.tokenization_is_day3_mirror(tokenization, wrong)["status"] == "FAIL"


# --------------------------------------------------------------------------
# B. Crossing equivalence under semantically identical canonical margins
# --------------------------------------------------------------------------

CURVES = [
    np.array([5.0, 4, 3, 2, 1, -1, -2, -3, -4]),
    np.array([2.0, 3, 1, 0, -1, 2, -3, -1, -5]),
    np.array([-1.0, -2, -3, -4, -5, -6, -7, -8, -9]),
    np.array([1.0, 1, 1, 1, 1, 1, 1, 1, 1]),
    np.array([4.0, 4, 3, 3, 2, 2, 1, 1, 0]),
]


class TestCrossingEquivalence:
    """Same canonical margins reached through opposite token mappings agree."""

    @pytest.mark.parametrize("curve", CURVES, ids=lambda c: f"{c[0]:+.0f}")
    def test_every_curve_quantity_is_identical(self, curve):
        # Original: A carries the margin. Swapped: B carries it, A is its negation.
        original = labelswap.original_margin(curve, np.zeros(N_STRICTNESS))
        swapped = labelswap.swapped_margin(-curve, np.zeros(N_STRICTNESS))
        assert np.array_equal(original, swapped)

        for j_star in range(1, NO_CROSSING_INDEX + 1):
            a = labelswap.curve_metrics(original, j_star)
            b = labelswap.curve_metrics(swapped, j_star)
            assert a.keys() == b.keys()
            for key in a:
                # Spearman is undefined (NaN) for a flat curve; NaN != NaN, so
                # compare those by both-undefined rather than by equality.
                if isinstance(a[key], float) and np.isnan(a[key]):
                    assert np.isnan(b[key]), key
                else:
                    assert a[key] == b[key], key

    @pytest.mark.parametrize("curve", CURVES, ids=lambda c: f"{c[0]:+.0f}")
    def test_decisions_crossing_and_tce_agree(self, curve):
        original = curve
        swapped = labelswap.swapped_margin(-curve, np.zeros(N_STRICTNESS))
        assert np.array_equal(
            labelswap.semantic_decision(original), labelswap.semantic_decision(swapped)
        )
        assert int(crossing_index(original[None, :])[0]) == int(
            crossing_index(swapped[None, :])[0]
        )

    @pytest.mark.parametrize("curve", CURVES, ids=lambda c: f"{c[0]:+.0f}")
    def test_prefix_records_reachable_set_tr_and_rc_agree(self, curve):
        swapped = labelswap.swapped_margin(-curve, np.zeros(N_STRICTNESS))
        assert day3.prefix_record_lows(curve) == day3.prefix_record_lows(swapped)
        assert day3.reachable_crossings(curve) == day3.reachable_crossings(swapped)
        assert day3.reachable_count(curve) == day3.reachable_count(swapped)
        for j_star in range(1, NO_CROSSING_INDEX + 1):
            assert day3.translation_reachable(
                curve, j_star
            ) == day3.translation_reachable(swapped, j_star)

    def test_using_the_wrong_sign_would_be_detected(self):
        """Analysing a swapped curve with the original convention changes answers."""
        curve = np.array([5.0, 4, 3, 2, 1, -1, -2, -3, -4])
        wrong = -curve  # what S_A - S_B would give under the swapped instruction
        assert int(crossing_index(curve[None, :])[0]) != int(
            crossing_index(wrong[None, :])[0]
        )
        assert day3.reachable_crossings(curve) != day3.reachable_crossings(wrong)


# --------------------------------------------------------------------------
# C. Label-location perturbation: TCE may move, ordering does not
# --------------------------------------------------------------------------


class TestLocationPerturbation:
    BASE = np.array([5.0, 4, 3, 2, 1, -1, -2, -3, -4])

    @pytest.mark.parametrize("shift", [-4.0, -1.5, 0.0, 1.5, 4.0, 10.0])
    def test_prefix_record_structure_is_unchanged_by_a_shift(self, shift):
        shifted = self.BASE + shift
        assert day3.prefix_record_lows(shifted) == day3.prefix_record_lows(self.BASE)
        assert day3.reachable_crossings(shifted) == day3.reachable_crossings(self.BASE)

    def test_tce_can_change_under_a_pure_location_shift(self):
        j_star = 5
        base_tce = labelswap.curve_metrics(self.BASE, j_star)["tce"]
        shifted_tce = labelswap.curve_metrics(self.BASE - 3.5, j_star)["tce"]
        assert base_tce != shifted_tce

    def test_a_location_shift_never_changes_tr(self):
        for shift in (-10.0, -2.0, 2.0, 10.0):
            for j_star in range(1, NO_CROSSING_INDEX + 1):
                assert day3.translation_reachable(
                    self.BASE + shift, j_star
                ) == day3.translation_reachable(self.BASE, j_star)

    def test_this_is_exactly_the_outcome_b_signature(self):
        """Location moves, ordering persists -- the preregistered Outcome B."""
        shifted = self.BASE - 3.5
        assert labelswap.curve_metrics(shifted, 5)["tce"] != (
            labelswap.curve_metrics(self.BASE, 5)["tce"]
        )
        assert (
            labelswap.curve_metrics(shifted, 5)["reachable_crossings"]
            == labelswap.curve_metrics(self.BASE, 5)["reachable_crossings"]
        )


# --------------------------------------------------------------------------
# D. Ordering perturbation: TR and reachable sets can change
# --------------------------------------------------------------------------


class TestOrderingPerturbation:
    ORDERED = np.array([5.0, 4, 3, 2, 1, 0, -1, -2, -3])
    REORDERED = np.array([5.0, 4, 2, 3, 1, 0, -1, -2, -3])

    def test_reachable_sets_differ(self):
        assert day3.reachable_crossings(self.ORDERED) != day3.reachable_crossings(
            self.REORDERED
        )

    def test_a_boundary_reachable_before_becomes_unreachable(self):
        assert day3.translation_reachable(self.ORDERED, 3)
        assert not day3.translation_reachable(self.REORDERED, 3)

    def test_reachable_count_drops(self):
        assert day3.reachable_count(self.REORDERED) < day3.reachable_count(self.ORDERED)

    def test_reachable_set_agreement_detects_the_change(self):
        a = labelswap.curve_metrics(self.ORDERED, 3)["reachable_crossings"]
        b = labelswap.curve_metrics(self.REORDERED, 3)["reachable_crossings"]
        same, jaccard = labelswap.reachable_set_agreement(a, b)
        assert not same
        assert 0.0 < jaccard < 1.0

    def test_identical_sets_give_perfect_agreement(self):
        a = labelswap.curve_metrics(self.ORDERED, 3)["reachable_crossings"]
        same, jaccard = labelswap.reachable_set_agreement(a, a)
        assert same and jaccard == 1.0

    def test_this_is_the_outcome_c_signature(self):
        """Ordering changes materially -- the preregistered Outcome C."""
        assert day3.prefix_record_lows(self.ORDERED) != day3.prefix_record_lows(
            self.REORDERED
        )


# --------------------------------------------------------------------------
# E. Tie convention
# --------------------------------------------------------------------------


class TestTieConvention:
    TIED = np.array([5.0, 4, 4, 3, 2, 1, 0, -1, -2])

    def test_equality_is_excluded_from_strict_prefix_record_lows(self):
        assert self.TIED[1] == self.TIED[2]
        assert 1 in day3.prefix_record_lows(self.TIED)
        assert 2 not in day3.prefix_record_lows(self.TIED)

    def test_tied_crossing_is_unreachable(self):
        assert not day3.translation_reachable(self.TIED, 2)

    def test_no_tolerance_is_applied(self):
        """One ULP below the running minimum is a record; exact equality is not."""
        nudged = self.TIED.copy()
        nudged[2] = np.nextafter(nudged[1], -np.inf)
        assert 2 in day3.prefix_record_lows(nudged)
        assert 2 not in day3.prefix_record_lows(self.TIED)

    def test_flat_curve_reaches_only_the_endpoints(self):
        flat = np.zeros(N_STRICTNESS)
        assert day3.reachable_crossings(flat) == (0, NO_CROSSING_INDEX)

    def test_tie_convention_survives_the_swap(self):
        swapped = labelswap.swapped_margin(-self.TIED, np.zeros(N_STRICTNESS))
        assert day3.prefix_record_lows(swapped) == day3.prefix_record_lows(self.TIED)


# --------------------------------------------------------------------------
# F. Prompt-diff whitelist
# --------------------------------------------------------------------------


class TestPromptDiff:
    def test_diff_passes(self):
        assert labelswap.prompt_diff()["status"] == "PASS"

    def test_exactly_two_positions_differ(self):
        diff = labelswap.prompt_diff()
        assert diff["n_differing_positions"] == 2
        assert diff["same_length"]

    def test_both_differences_are_label_characters(self):
        for entry in labelswap.prompt_diff()["differing_positions"]:
            assert {entry["original"], entry["swapped"]} == {LABEL_MET, LABEL_NOT_MET}

    def test_everything_before_the_response_instruction_is_identical(self):
        original = PROMPT_TEMPLATE
        swapped = labelswap.swapped_prompt_template()
        assert original[: -len(labelswap.ORIGINAL_RESPONSE_INSTRUCTION)] == (
            swapped[: -len(labelswap.SWAPPED_RESPONSE_INSTRUCTION)]
        )

    def test_the_swapped_template_is_derived_not_retyped(self):
        """Rebuilding from the frozen template reproduces it byte for byte."""
        head = PROMPT_TEMPLATE[: -len(labelswap.ORIGINAL_RESPONSE_INSTRUCTION)]
        assert labelswap.swapped_prompt_template() == (
            head + labelswap.SWAPPED_RESPONSE_INSTRUCTION
        )

    def test_criterion_and_response_placeholders_survive(self):
        swapped = labelswap.swapped_prompt_template()
        assert "{rubric_text}" in swapped and "{candidate_response}" in swapped

    def test_rendered_messages_differ_only_in_the_instruction(self):
        rubric, response = "RUBRIC TEXT", "CANDIDATE RESPONSE"
        original = prompts.build_user_message(rubric, response)
        swapped = labelswap.build_swapped_user_message(rubric, response)
        assert len(original) == len(swapped)
        positions = [i for i, (x, y) in enumerate(zip(original, swapped)) if x != y]
        assert len(positions) == 2

    def test_the_two_instructions_are_a_permutation_of_each_other(self):
        assert sorted(labelswap.ORIGINAL_RESPONSE_INSTRUCTION) == sorted(
            labelswap.SWAPPED_RESPONSE_INSTRUCTION
        )

    def test_a_tampered_template_is_rejected(self):
        tampered = labelswap.swapped_prompt_template().replace(
            "Criterion:", "Criteria:"
        )
        assert labelswap.prompt_diff(swapped=tampered)["status"] == "FAIL"

    def test_a_length_change_is_rejected(self):
        assert (
            labelswap.prompt_diff(swapped=labelswap.swapped_prompt_template() + " ")[
                "status"
            ]
            == "FAIL"
        )


class TestFrozenPromptArtifacts:
    @staticmethod
    @pytest.fixture(scope="module")
    def diff_record(repo_root: Path):
        path = repo_root / lsc.PROMPT_DIFF_PATH
        if not path.is_file():
            pytest.skip("run scripts/freeze_labelswap_preregistration.py")
        return json.loads(path.read_text(encoding="utf-8"))

    def test_recorded_diff_matches_the_module(self, diff_record):
        live = labelswap.prompt_diff()
        assert diff_record["original_sha256"] == live["original_sha256"]
        assert diff_record["swapped_sha256"] == live["swapped_sha256"]
        assert diff_record["status"] == "PASS"

    def test_every_row_body_was_verified_identical(self, diff_record):
        equivalence = diff_record["row_equivalence"]
        assert equivalence["status"] == "PASS"
        assert equivalence["n_bodies_identical"] == equivalence["n_rows"] == 2916
        assert equivalence["metadata_identical"]


# --------------------------------------------------------------------------
# G. Frozen-record immutability
# --------------------------------------------------------------------------


class TestFrozenRecordImmutability:
    @staticmethod
    @pytest.fixture(scope="module")
    def fingerprint(repo_root: Path):
        path = repo_root / lsc.FROZEN_FINGERPRINT_PATH
        if not path.is_file():
            pytest.skip("run scripts/freeze_labelswap_preregistration.py")
        return json.loads(path.read_text(encoding="utf-8"))

    def test_every_protected_file_still_hashes_as_recorded(
        self, fingerprint, repo_root: Path
    ):
        changed = [
            relative
            for relative, digest in fingerprint["files"].items()
            if provenance.sha256_file(repo_root / relative) != digest
        ]
        assert changed == [], f"frozen scientific files changed: {changed}"

    def test_combined_fingerprint_is_unchanged(self, fingerprint, repo_root: Path):
        combined = provenance.sha256_text(
            "\n".join(
                f"{relative}:{provenance.sha256_file(repo_root / relative)}"
                for relative in sorted(fingerprint["files"])
            )
        )
        assert combined == fingerprint["combined_sha256"]

    def test_day3_prior_manifest_still_verifies(self, fingerprint):
        assert fingerprint["recorded_day3_prior_manifest_verifies"]

    def test_label_swap_writes_only_to_additive_locations(self, repo_root: Path):
        """No label-swap source may write into a protected tree."""
        sources = [
            "src/formalcrrc/labelswap.py",
            "src/formalcrrc/labelswap_config.py",
            "scripts/freeze_labelswap_preregistration.py",
            "scripts/run_labelswap_inference.py",
            "scripts/analyze_labelswap.py",
        ]
        forbidden = ("artifacts/day1", "artifacts/day2", "artifacts/day3", "figures/day")
        for relative in sources:
            path = repo_root / relative
            if not path.is_file():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if any(
                    token in line
                    for token in ("write_json", "to_parquet", "write_text", "to_csv", "savefig")
                ):
                    for tree in forbidden:
                        assert tree not in line, f"{relative}: {line.strip()}"

    def test_all_declared_paths_are_additive(self):
        for attribute in dir(lsc):
            if attribute.endswith("_PATH") or attribute.endswith("_DIR"):
                value = getattr(lsc, attribute)
                if isinstance(value, str) and value.startswith(("artifacts/", "figures/")):
                    assert value.startswith(
                        ("artifacts/labelswap", "figures/labelswap")
                    ), f"{attribute} = {value}"


# --------------------------------------------------------------------------
# Preregistration integrity
# --------------------------------------------------------------------------


class TestPreregistration:
    @staticmethod
    @pytest.fixture(scope="module")
    def prereg(repo_root: Path):
        path = repo_root / lsc.PREREG_JSON_PATH
        if not path.is_file():
            pytest.skip("run scripts/freeze_labelswap_preregistration.py")
        return json.loads(path.read_text(encoding="utf-8"))

    def test_checksum_verifies(self, repo_root: Path, prereg):
        assert provenance.verify_checksum_file(
            repo_root / lsc.PREREG_JSON_PATH, repo_root / lsc.PREREG_SHA_PATH
        )

    def test_zero_amendments(self, prereg):
        assert prereg["amendments"] == 0

    def test_frozen_before_inference(self, prereg, repo_root: Path):
        assert prereg["frozen_at"]
        assert prereg["preregistered"] is True

    def test_canonical_swapped_margin_is_sb_minus_sa(self, prereg):
        assert prereg["intervention"]["canonical_margin_swapped"] == "M_S(s) = S_B - S_A"

    def test_all_four_day3_revisions_recorded(self, prereg):
        models = prereg["day3_facts"]["models"]
        assert len(models) == 4
        for record in models.values():
            assert len(record["revision"]) == 40

    def test_robustness_bands_are_the_preregistered_ones(self, prereg):
        bands = prereg["robustness_bands"]
        assert bands["delta_tce"] == [-0.5, 0.5]
        assert bands["delta_trr"] == [-0.10, 0.10]
        assert bands["agreement_lower_bound"] == 0.90

    def test_conventions_are_inherited_unchanged(self, prereg):
        assert "min{s : M(s) < 0}" in prereg["conventions"]["crossing"]
        assert "strict prefix record low" in prereg["conventions"]["reachability"]

    def test_draft_consistency_checks_all_passed(self, prereg):
        assert all(prereg["draft_consistency_checks"].values())


# --------------------------------------------------------------------------
# Dataset verification (Phase 5)
# --------------------------------------------------------------------------


class TestDatasetReuse:
    @staticmethod
    @pytest.fixture(scope="module")
    def manifest(repo_root: Path):
        import pandas as pd

        path = repo_root / lsc.PROMPT_MANIFEST_PATH
        if not path.is_file():
            pytest.skip("run scripts/freeze_labelswap_preregistration.py")
        return pd.read_parquet(path)

    def test_row_and_artifact_counts(self, manifest):
        assert len(manifest) == lsc.N_ROWS_PER_MODEL == 2916
        assert manifest["artifact_id"].nunique() == lsc.N_ARTIFACTS == 324

    def test_strictness_levels_complete(self, manifest):
        assert sorted(manifest["strictness_index"].unique()) == list(range(N_STRICTNESS))

    def test_family_balance(self, manifest):
        counts = manifest.drop_duplicates("artifact_id")["family"].value_counts()
        assert set(counts.values) == {108}

    def test_thirty_six_artifacts_at_each_true_boundary(self, manifest):
        artifacts = manifest.drop_duplicates("artifact_id")
        boundaries = [
            labelswap.true_first_fail(row.family, row.latent_level)
            for row in artifacts.itertuples()
        ]
        counts = {j: boundaries.count(j) for j in set(boundaries)}
        assert counts == {j: 36 for j in range(1, 10)}

    def test_artifact_ids_match_day3_exactly(self, manifest, repo_root: Path):
        import pandas as pd

        day3_dataset = pd.read_parquet(repo_root / "artifacts/day3/test_dataset.parquet")
        assert set(manifest["artifact_id"]) == set(day3_dataset["artifact_id"])
        assert set(manifest["prompt_id"]) == set(day3_dataset["prompt_id"])

    def test_labels_recompute_from_the_predicates(self, repo_root: Path):
        import pandas as pd

        dataset = pd.read_parquet(repo_root / "artifacts/day3/test_dataset.parquet")
        recomputed = [
            predicates.formal_truth(r.family, r.latent_level, r.strictness_index)
            for r in dataset.itertuples()
        ]
        assert recomputed == list(dataset["formal_truth"].astype(int))

    def test_j_star_equals_z_plus_one(self, repo_root: Path):
        import pandas as pd

        from formalcrrc import step_proposition as sp

        dataset = pd.read_parquet(repo_root / "artifacts/day3/test_dataset.parquet")
        for row in dataset.drop_duplicates("artifact_id").itertuples():
            z = sp.canonical_compliance(row.family, row.latent_level)
            assert labelswap.true_first_fail(row.family, row.latent_level) == z + 1
