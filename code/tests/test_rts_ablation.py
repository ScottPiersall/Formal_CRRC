"""Elicited reason-then-score protocol ablation on the frozen Qwen2.5 judge.

Preregistered within-model ablation
(`docs/PREREGISTRATION_REASON_THEN_SCORE_ABLATION.md`), frozen before any
Qwen2.5 reason-then-score row is run. These tests guard the failures that would
make the comparison meaningless:

* drifting off the exact Day-3 revision, which would break the pairing with the
  frozen comparator;
* changing the prompt by more than the preregistered scaffold, which would
  confound the protocol change with a formatting change;
* letting the model sample its own A/B label instead of scoring it, which would
  measure something other than the readout;
* fabricating the marker on a truncated trace;
* tokenising the candidates in isolation rather than in context;
* smuggling a tolerance into the strict reachability rule or the
  tie-versus-inversion decomposition.

The blocks below are the preregistered Section-27 requirements A-O, plus the
frozen-record check.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from formalcrrc import dataset as ds
from formalcrrc import day3, provenance
from formalcrrc import rts_ablation as ra
from formalcrrc import rts_ablation_config as rc
from formalcrrc.config import N_STRICTNESS, NO_CROSSING_INDEX, PROMPT_TEMPLATE
from formalcrrc.day2_bootstrap import crossing_index


class MarkerTokenizer:
    """A tokenizer under which the marker and candidates span several tokens.

    Prefix-preserving, so a continuation really is the tail of the longer
    encoding, and deliberately unlike isolated encoding for short strings -- the
    SentencePiece behaviour that made Day-1 label resolution non-trivial.
    """

    BOUNDARY = 9000
    PIECES = {"FINAL": [700], ":": [701], " A": [801, 802], " B": [803]}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids: list[int] = []
        rest = text
        while rest:
            for piece, mapped in self.PIECES.items():
                if rest.startswith(piece):
                    ids.extend(mapped)
                    rest = rest[len(piece) :]
                    break
            else:
                ids.append(ord(rest[0]))
                rest = rest[1:]
        if len(text) <= 2 and not text.startswith(" "):
            return [self.BOUNDARY] + ids
        return ids


CONTEXT = "a long standardised decision context ending with the marker FINAL:"


# --------------------------------------------------------------------------
# A. Qwen2.5 revision continuity
# --------------------------------------------------------------------------


class TestRevisionContinuity:
    def test_pinned_revision_matches_frozen_day3(self, repo_root: Path):
        record = json.loads(
            (repo_root / "artifacts/day3/model_provenance.json").read_text(
                encoding="utf-8"
            )
        )["models"][rc.MODEL_ID]
        assert rc.DAY3_REVISION == record["revision"]
        assert rc.DAY3_TOKENIZER_REVISION == record["tokenizer_revision"]

    def test_pinned_chat_template_matches_frozen_day3(self, repo_root: Path):
        record = json.loads(
            (repo_root / "artifacts/day3/model_provenance.json").read_text(
                encoding="utf-8"
            )
        )["models"][rc.MODEL_ID]
        assert rc.DAY3_CHAT_TEMPLATE_SHA256 == record["chat_template_sha256"]

    def test_revision_is_resolved_not_a_branch_name(self):
        assert rc.DAY3_REVISION not in {"main", "master", "HEAD"}
        assert len(rc.DAY3_REVISION) == 40

    def test_dtype_and_quantization(self):
        assert rc.MODEL_DTYPE == "bfloat16"
        assert rc.QUANTIZATION == "none"

    def test_revision_matches_the_raw_scores_it_will_be_compared_against(
        self, repo_root: Path
    ):
        import pandas as pd

        raw = pd.read_parquet(
            repo_root / "artifacts/day3/raw_scores/qwen2_5_14b_instruct.parquet"
        )
        assert set(raw["revision"].unique()) == {rc.DAY3_REVISION}


# --------------------------------------------------------------------------
# B. Frozen dataset identity
# --------------------------------------------------------------------------


class TestDatasetIdentity:
    def test_content_hash_matches_manifest(self, repo_root: Path):
        import pandas as pd

        manifest = json.loads(
            (repo_root / "artifacts/day3/test_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        stored = pd.read_parquet(repo_root / "artifacts/day3/test_dataset.parquet")
        assert ds.dataset_content_hash(stored) == (
            manifest["hashes"]["dataset_content_sha256"]
        )

    def test_design_dimensions(self, repo_root: Path):
        import pandas as pd

        stored = pd.read_parquet(repo_root / "artifacts/day3/test_dataset.parquet")
        per = stored.drop_duplicates("artifact_id")
        assert len(stored) == rc.N_ROWS == 2916
        assert per.shape[0] == rc.N_ARTIFACTS == 324
        assert set(stored.groupby("artifact_id")["strictness_index"].nunique()) == {9}
        assert set(per["family"].value_counts().unique()) == {108}
        counts = per["true_first_fail_index"].value_counts()
        assert all(int(counts[j]) == 36 for j in range(1, 10))
        assert int(per["true_first_fail_index"].between(1, 8).sum()) == rc.N_NONTRIVIAL

    def test_every_study_row_pairs_to_one_frozen_row(self, repo_root: Path):
        import pandas as pd

        stored = pd.read_parquet(repo_root / "artifacts/day3/test_dataset.parquet")
        raw = pd.read_parquet(
            repo_root / "artifacts/day3/raw_scores/qwen2_5_14b_instruct.parquet"
        )
        assert set(stored["prompt_id"]) == set(raw["prompt_id"])
        assert raw["prompt_id"].is_unique


# --------------------------------------------------------------------------
# C. Prompt construction / D. prompt diff
# --------------------------------------------------------------------------


class TestPromptConstruction:
    def test_reason_then_score_is_a_pure_suffix_extension(self):
        """The intervention adds text; it never edits the frozen prompt."""
        assert rc.RTS_PROMPT_TEMPLATE.startswith(PROMPT_TEMPLATE)

    def test_diff_removes_nothing(self):
        diff = ra.prompt_diff()
        assert diff["removed_lines"] == []
        assert all(diff["checks"].values()), diff["checks"]

    def test_added_suffix_is_exactly_the_preregistered_scaffold(self):
        suffix = rc.RTS_PROMPT_TEMPLATE[len(PROMPT_TEMPLATE) :]
        assert suffix == rc.SCAFFOLD_SEPARATOR + rc.REASONING_SCAFFOLD
        assert rc.MARKER in suffix

    def test_placeholders_survive(self):
        assert "{rubric_text}" in rc.RTS_PROMPT_TEMPLATE
        assert "{candidate_response}" in rc.RTS_PROMPT_TEMPLATE

    def test_bodies_identical_for_every_frozen_row(self, repo_root: Path):
        import pandas as pd

        stored = pd.read_parquet(repo_root / "artifacts/day3/test_dataset.parquet")
        verdict = ra.bodies_identical(stored)
        assert verdict["status"] == "PASS", verdict
        assert verdict["n_rows_checked"] == rc.N_ROWS

    def test_no_label_swap(self):
        assert rc.CANDIDATE_MET.strip() == "A"
        assert rc.CANDIDATE_NOT_MET.strip() == "B"


# --------------------------------------------------------------------------
# D. Marker tokenization
# --------------------------------------------------------------------------


class TestMarkerTokenization:
    def test_marker_resolves_in_context(self):
        resolved = ra.resolve_marker_tokenization(MarkerTokenizer(), "some context ")
        assert resolved["contextual_token_ids"] == [700, 701]
        assert resolved["multi_token"]

    def test_marker_bytes_recorded(self):
        resolved = ra.resolve_marker_tokenization(MarkerTokenizer(), "some context ")
        assert resolved["marker_bytes_hex"] == "FINAL:".encode("utf-8").hex()

    def test_marker_is_the_preregistered_string(self):
        assert rc.MARKER == "FINAL:"


# --------------------------------------------------------------------------
# E. Sequence stopping
# --------------------------------------------------------------------------


class TestStopping:
    def test_generation_is_cut_at_the_marker(self):
        split = ra.split_at_marker("reasoning here\nFINAL:", budget_exhausted=False)
        assert split.marker_reached
        assert split.text_through_marker.endswith(rc.MARKER)

    def test_label_after_the_marker_is_discarded(self):
        """The primary margin must come from scoring, not from a sampled token."""
        split = ra.split_at_marker("reasoning\nFINAL: A", budget_exhausted=False)
        assert split.text_through_marker == "reasoning\nFINAL:"
        assert not split.text_through_marker.endswith("A")

    def test_only_the_first_marker_counts(self):
        split = ra.split_at_marker("a FINAL: b FINAL: c", budget_exhausted=False)
        assert split.text_through_marker == "a FINAL:"

    def test_missing_marker_is_truncated_not_repaired(self):
        split = ra.split_at_marker("thinking with no marker", budget_exhausted=True)
        assert split.truncated and not split.marker_reached
        assert rc.MARKER not in split.text_through_marker
        assert split.stop_reason == "length"

    def test_stopping_early_without_marker_is_distinguished(self):
        split = ra.split_at_marker("short", budget_exhausted=False)
        assert split.stop_reason == "stopped_without_marker"

    def test_premature_label_is_flagged_but_not_fatal(self):
        split = ra.split_at_marker("maybe\nA\nthen more\nFINAL:", budget_exhausted=False)
        assert split.premature_label
        assert split.marker_reached

    def test_reasoning_without_a_standalone_label_is_not_flagged(self):
        split = ra.split_at_marker(
            "The response satisfies A-grade coverage overall.\nFINAL:",
            budget_exhausted=False,
        )
        assert not split.premature_label


# --------------------------------------------------------------------------
# F. Candidate contextual tokenization / G. multi-token likelihood
# --------------------------------------------------------------------------


class TestCandidates:
    def test_candidates_carry_a_leading_space(self):
        assert rc.CANDIDATE_MET == " A"
        assert rc.CANDIDATE_NOT_MET == " B"

    def test_candidates_resolve_as_continuations(self):
        resolved = ra.resolve_candidate_tokenization(MarkerTokenizer(), CONTEXT)
        assert resolved["met_token_ids"] == [801, 802]
        assert resolved["not_met_token_ids"] == [803]
        assert resolved["multi_token"]

    def test_multi_token_candidate_is_summed_over_all_positions(self):
        table = [None, {1: -0.1}, {801: -0.4}, {802: -0.35}]
        assert ra.sequence_logprob(table, 2, [801, 802]) == pytest.approx(-0.75)

    def test_missing_logprob_raises_rather_than_substituting(self):
        with pytest.raises(ValueError, match="no log-probability recorded"):
            ra.sequence_logprob([None, {9: -1.0}], 1, [801])

    def test_no_length_normalization(self):
        """A two-token candidate is not divided by its length."""
        table = [None, {801: -1.0}, {802: -1.0}]
        assert ra.sequence_logprob(table, 1, [801, 802]) == pytest.approx(-2.0)


# --------------------------------------------------------------------------
# H. Margin sign / K. tie convention
# --------------------------------------------------------------------------


class TestMarginAndTies:
    def test_higher_a_likelihood_gives_positive_margin(self):
        assert ra.canonical_margin(-0.3, -2.1) > 0

    def test_higher_b_likelihood_gives_negative_margin(self):
        assert ra.canonical_margin(-2.1, -0.3) < 0

    def test_zero_margin_counts_as_met(self):
        assert ra.semantic_decision(0.0) == 1

    def test_zero_margin_does_not_trigger_a_crossing(self):
        curve = np.array([[2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0]])
        assert int(crossing_index(curve)[0]) == 3


# --------------------------------------------------------------------------
# I. Deterministic seed
# --------------------------------------------------------------------------


class TestSeed:
    def test_same_row_same_seed(self):
        assert ra.row_seed("D3T-A-L3-I07", 4) == ra.row_seed("D3T-A-L3-I07", 4)

    def test_strictness_changes_the_seed(self):
        assert len({ra.row_seed("D3T-A-L3-I07", s) for s in range(9)}) == 9

    def test_shares_the_reasoning_anchor_convention(self):
        """Deliberate reuse, so the two studies are procedurally comparable."""
        from formalcrrc import reasoning_anchor

        assert ra.row_seed is reasoning_anchor.row_seed
        assert rc.SEED_MODULUS == 2147483647

    def test_seed_in_range(self):
        for s in range(9):
            assert 0 <= ra.row_seed("D3T-C-L8-I11", s) < rc.SEED_MODULUS


# --------------------------------------------------------------------------
# J. Crossing / L. strict prefix record
# --------------------------------------------------------------------------


class TestCrossingAndReachability:
    def test_first_negative_is_the_crossing(self):
        curve = np.array([[3.0, 2.0, -0.5, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0]])
        assert int(crossing_index(curve)[0]) == 2

    def test_no_negative_gives_the_sentinel(self):
        assert int(crossing_index(np.ones((1, N_STRICTNESS)))[0]) == NO_CROSSING_INDEX

    def test_strict_inequality_is_used(self):
        curve = [5.0, 3.0, 3.0, 1.0, 0.5, 0.4, 0.3, 0.2, 0.1]
        assert 2 not in day3.prefix_record_lows(curve)

    def test_extremes_always_reachable(self):
        reach = day3.reachable_crossings([1.0] * N_STRICTNESS)
        assert 0 in reach and NO_CROSSING_INDEX in reach


# --------------------------------------------------------------------------
# M. Tie-vs-inversion classification
# --------------------------------------------------------------------------


class TestFailureDecomposition:
    def test_reachable_curve(self):
        curve = [5.0, 4.0, 3.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0]
        assert ra.classify_failure(curve, 3) == rc.FAILURE_NONE

    def test_strict_inversion(self):
        """An earlier threshold scored strictly below the true boundary."""
        curve = [5.0, -9.0, 3.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0]
        assert ra.classify_failure(curve, 3) == rc.FAILURE_STRICT_INVERSION

    def test_tie_only(self):
        """Nothing below it, but something exactly equal to it."""
        curve = [5.0, -1.0, 3.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0]
        assert ra.classify_failure(curve, 3) == rc.FAILURE_TIE_ONLY

    def test_inversion_takes_precedence_over_a_tie(self):
        curve = [-1.0, -9.0, 3.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0]
        assert ra.classify_failure(curve, 3) == rc.FAILURE_STRICT_INVERSION

    def test_classification_agrees_with_the_reachability_rule(self):
        """Whatever the label, 'reachable' must mean exactly what Day 3 means."""
        rng = np.random.default_rng(42)
        for _ in range(400):
            curve = np.round(rng.normal(size=N_STRICTNESS), 2)
            target = int(rng.integers(1, 9))
            label = ra.classify_failure(curve, target)
            expected = target in day3.reachable_crossings(curve)
            assert (label == rc.FAILURE_NONE) == expected

    def test_no_tolerance_is_introduced(self):
        """A difference of one ulp is an inversion, not a tie."""
        base = 1.0
        nudged = np.nextafter(base, -np.inf)
        curve = [5.0, nudged, 3.0, base, -2.0, -3.0, -4.0, -5.0, -6.0]
        assert ra.classify_failure(curve, 3) == rc.FAILURE_STRICT_INVERSION

    def test_trivial_boundaries_are_never_classified_as_failures(self):
        curve = [1.0] * N_STRICTNESS
        assert ra.classify_failure(curve, 0) == rc.FAILURE_NONE
        assert ra.classify_failure(curve, 9) == rc.FAILURE_NONE


# --------------------------------------------------------------------------
# N. Frozen original Qwen2.5 recovery
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def recovered(repo_root: Path):
    """Condition O, recomputed once from the frozen raw scores."""
    import pandas as pd

    raw = pd.read_parquet(
        repo_root / "artifacts/day3/raw_scores/qwen2_5_14b_instruct.parquet"
    )
    data = pd.read_parquet(repo_root / "artifacts/day3/test_dataset.parquet")
    return ra.recover_comparator(raw, data)


class TestComparatorRecovery:
    def test_headline_values_reproduce(self, recovered):
        nontrivial = recovered[recovered["nontrivial_boundary"]]
        assert float(recovered["tce_o"].mean()) == pytest.approx(3.373, abs=5e-4)
        assert float(nontrivial["translation_reachable_o"].mean()) == pytest.approx(
            0.4792, abs=5e-5
        )
        unreachable = 100.0 * (
            1.0 - float(nontrivial["translation_reachable_o"].mean())
        )
        assert unreachable == pytest.approx(52.08, abs=5e-3)

    def test_matches_the_frozen_metrics_table_per_artifact(
        self, repo_root: Path, recovered
    ):
        import pandas as pd

        frozen = pd.read_parquet(
            repo_root / "artifacts/day3/metrics_by_artifact.parquet"
        )
        frozen = frozen[frozen["model_id"] == rc.MODEL_ID]
        verdict = ra.verify_comparator(recovered, frozen)
        assert verdict["status"] == "PASS", verdict

    def test_nontrivial_count(self, recovered):
        assert int(recovered["nontrivial_boundary"].sum()) == rc.N_NONTRIVIAL


# --------------------------------------------------------------------------
# Exact interval for a zero count
# --------------------------------------------------------------------------


class TestClopperPearson:
    def test_zero_of_288_is_not_a_point_interval(self):
        low, high = ra.clopper_pearson(0, 288)
        assert low == 0.0
        assert 0.0 < high < 0.02
        assert (low, high) != (0.0, 0.0)

    def test_interval_contains_the_observed_proportion(self):
        for successes in (0, 1, 37, 150, 288):
            low, high = ra.clopper_pearson(successes, 288)
            assert low <= successes / 288 <= high

    def test_all_successes_reaches_one(self):
        assert ra.clopper_pearson(288, 288)[1] == 1.0

    def test_invalid_counts_rejected(self):
        with pytest.raises(ValueError, match="outside"):
            ra.clopper_pearson(300, 288)


# --------------------------------------------------------------------------
# O. Frozen-file integrity
# --------------------------------------------------------------------------


class TestFrozenRecord:
    def test_fingerprint_round_trips(self, repo_root: Path):
        fingerprint = ra.build_frozen_fingerprint(repo_root)
        assert fingerprint["n_files"] > 140
        assert ra.verify_frozen_fingerprint(fingerprint, repo_root)["status"] == "PASS"

    def test_comparator_and_anchor_are_both_protected(self, repo_root: Path):
        covered = ra.build_frozen_fingerprint(repo_root)["files"]
        assert "artifacts/day3/raw_scores/qwen2_5_14b_instruct.parquet" in covered
        assert "artifacts/reasoning_anchor/summary.json" in covered
        assert "docs/RESULTS_REASONING_ANCHOR.md" in covered

    def test_prior_preregistration_checksums_still_verify(self, repo_root: Path):
        for study in ("day1", "day2", "day3", "labelswap", "reasoning_anchor"):
            json_path = repo_root / f"artifacts/{study}/preregistration.json"
            sha_path = repo_root / f"artifacts/{study}/preregistration.sha256"
            if not (json_path.is_file() and sha_path.is_file()):
                continue
            assert provenance.verify_checksum_file(json_path, sha_path), study

    def test_stored_fingerprint_matches_when_present(self, repo_root: Path):
        stored = repo_root / rc.FROZEN_FINGERPRINT_PATH
        if not stored.is_file():
            pytest.skip("ablation fingerprint not yet written")
        fingerprint = json.loads(stored.read_text(encoding="utf-8"))
        assert ra.verify_frozen_fingerprint(fingerprint, repo_root)["status"] == "PASS"
