"""Reason-then-score measurement for the reasoning-capable external anchor.

Preregistered post-study external-anchor experiment
(`docs/PREREGISTRATION_REASONING_ANCHOR.md`), frozen before any FormalCRRC
inference with this model. These tests run before the cluster job and guard the
failures that would silently invalidate the result:

* scoring A/B before the model has finished reasoning, which would measure the
  start of a thought rather than a judgment;
* fabricating an end-of-thinking delimiter for a trace that never produced one;
* encoding the candidate answers in isolation instead of in the decision
  context, the defect that was caught for Mistral in Day 1;
* assuming the candidates are single tokens;
* drifting away from the Day-3 crossing, tie and reachability conventions, which
  would make the anchor's numbers incomparable with the frozen panel.

The blocks below are the preregistered Section-21 requirements A-M.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from formalcrrc import dataset as ds
from formalcrrc import day3, provenance
from formalcrrc import reasoning_anchor as ra
from formalcrrc import reasoning_anchor_config as rc
from formalcrrc.config import N_STRICTNESS, NO_CROSSING_INDEX
from formalcrrc.day2_bootstrap import crossing_index

# --------------------------------------------------------------------------
# Tokenizer doubles
# --------------------------------------------------------------------------


class BoundaryMarkerTokenizer:
    """A tokenizer that marks a word boundary on short standalone strings.

    This is the SentencePiece behaviour that made Day-1 label resolution
    non-trivial: encoding ``"A"`` alone yields a different id from the one that
    actually continues a prompt. Encoding is otherwise prefix-preserving, so a
    continuation really is the tail of the longer encoding.
    """

    BOUNDARY = 9000

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = [ord(ch) for ch in text]
        if len(text) <= 2:
            return [self.BOUNDARY] + ids
        return ids


class MultiTokenTokenizer:
    """A tokenizer under which the candidate ``A`` spans two tokens."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids: list[int] = []
        for ch in text:
            if ch == "A":
                ids.extend([401, 402])
            elif ch == "B":
                ids.append(501)
            else:
                ids.append(ord(ch))
        return ids


CONTEXT = "a long standardised decision context ending with the separator\n\n"


# --------------------------------------------------------------------------
# A. Native thinking boundary
# --------------------------------------------------------------------------


class TestThinkingDelimiter:
    def test_trace_is_cut_at_the_first_delimiter(self):
        trace = ra.split_reasoning([11, 12, 13, 77, 14, 15], think_end_id=77, budget=100)
        assert trace.think_end_reached
        assert not trace.truncated
        assert trace.token_ids == (11, 12, 13, 77)
        assert trace.n_tokens == 4

    def test_tokens_after_the_delimiter_are_discarded(self):
        """The decision boundary is the delimiter; a free-form answer is not used."""
        trace = ra.split_reasoning([1, 99, 2, 99, 3], think_end_id=99, budget=100)
        assert trace.token_ids == (1, 99)

    def test_delimiter_is_included_not_stripped(self):
        trace = ra.split_reasoning([5, 6, 42], think_end_id=42, budget=100)
        assert trace.token_ids[-1] == 42

    def test_stop_reason_records_how_it_ended(self):
        assert ra.split_reasoning([1, 9], 9, 100).stop_reason == "think_end"


# --------------------------------------------------------------------------
# B. Missing delimiter -> truncated, never repaired
# --------------------------------------------------------------------------


class TestBudgetExhaustion:
    def test_budget_exhaustion_is_marked_truncated(self):
        ids = list(range(1, 65))
        trace = ra.split_reasoning(ids, think_end_id=999_999, budget=64)
        assert trace.truncated
        assert not trace.think_end_reached
        assert trace.stop_reason == "length"

    def test_no_delimiter_is_fabricated(self):
        """A truncated trace must not gain a delimiter it never produced."""
        ids = [1, 2, 3]
        trace = ra.split_reasoning(ids, think_end_id=77, budget=3)
        assert 77 not in trace.token_ids
        assert trace.token_ids == (1, 2, 3)

    def test_stopping_early_without_delimiter_is_distinguished(self):
        trace = ra.split_reasoning([1, 2], think_end_id=77, budget=64)
        assert trace.truncated
        assert trace.stop_reason == "stopped_without_think_end"


# --------------------------------------------------------------------------
# C. Deterministic per-row seed
# --------------------------------------------------------------------------


class TestDeterministicRowSeed:
    def test_same_row_gives_same_seed(self):
        assert ra.row_seed("D3T-A-L3-I07", 4) == ra.row_seed("D3T-A-L3-I07", 4)

    def test_different_strictness_gives_different_seed(self):
        seeds = {ra.row_seed("D3T-A-L3-I07", s) for s in range(N_STRICTNESS)}
        assert len(seeds) == N_STRICTNESS

    def test_different_artifact_gives_different_seed(self):
        assert ra.row_seed("D3T-A-L3-I07", 0) != ra.row_seed("D3T-B-L3-I07", 0)

    def test_seed_is_in_sampler_range(self):
        for artifact in ("D3T-A-L0-I00", "D3T-C-L8-I11"):
            for s in range(N_STRICTNESS):
                seed = ra.row_seed(artifact, s)
                assert 0 <= seed < rc.SEED_MODULUS

    def test_base_seed_participates(self):
        assert ra.row_seed("X", 0, base_seed=42) != ra.row_seed("X", 0, base_seed=43)

    def test_seed_does_not_use_process_randomised_hash(self):
        """Hard-coded expectation: a SHA-256 derivation reproduces across runs."""
        import hashlib

        payload = f"{rc.EXPERIMENT_SEED}|D3T-A-L0-I00|0"
        expected = (
            int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")
            % rc.SEED_MODULUS
        )
        assert ra.row_seed("D3T-A-L0-I00", 0) == expected


# --------------------------------------------------------------------------
# D. Contextual candidate scoring
# --------------------------------------------------------------------------


class TestContextualCandidateTokenization:
    def test_candidates_are_resolved_as_continuations(self):
        tok = BoundaryMarkerTokenizer()
        resolved = ra.resolve_candidate_tokenization(tok, CONTEXT)
        assert resolved.met_token_ids == (ord("A"),)
        assert resolved.not_met_token_ids == (ord("B"),)

    def test_isolated_encoding_would_have_been_wrong(self):
        """The whole point: the bare label encodes to a different id."""
        tok = BoundaryMarkerTokenizer()
        resolved = ra.resolve_candidate_tokenization(tok, CONTEXT)
        assert resolved.isolated_met_token_ids == (BoundaryMarkerTokenizer.BOUNDARY, ord("A"))
        assert resolved.differs_from_isolated

    def test_separator_resolves_in_context(self):
        tok = BoundaryMarkerTokenizer()
        ids = ra.resolve_separator_ids(tok, "some context that is long enough")
        assert ids == (ord("\n"), ord("\n"))

    def test_continuation_that_perturbs_the_prefix_is_rejected(self):
        class Perturbing:
            def encode(self, text, add_special_tokens=False):
                return [7] if text.endswith("A") else [1, 2, 3]

        with pytest.raises(ValueError, match="changed the preceding tokenisation"):
            ra.continuation_ids(Perturbing(), "ctx", "A")


# --------------------------------------------------------------------------
# E. Multi-token candidate support
# --------------------------------------------------------------------------


class TestMultiTokenCandidates:
    def test_candidate_spanning_two_tokens_is_resolved(self):
        resolved = ra.resolve_candidate_tokenization(MultiTokenTokenizer(), CONTEXT)
        assert resolved.met_token_ids == (401, 402)
        assert resolved.not_met_token_ids == (501,)
        assert resolved.multi_token

    def test_sequence_loglikelihood_sums_every_candidate_token(self):
        table = [None, {1: -0.1}, {401: -0.5}, {402: -0.25}]
        assert ra.sequence_logprob(table, 2, [401, 402]) == pytest.approx(-0.75)

    def test_single_token_case_is_the_same_function(self):
        table = [None, {501: -1.5}]
        assert ra.sequence_logprob(table, 1, [501]) == pytest.approx(-1.5)

    def test_missing_logprob_raises_rather_than_substituting(self):
        table = [None, {999: -0.1}]
        with pytest.raises(ValueError, match="no log-probability recorded"):
            ra.sequence_logprob(table, 1, [401])

    def test_running_past_the_end_raises(self):
        with pytest.raises(ValueError, match="past the end"):
            ra.sequence_logprob([None, {1: -1.0}], 1, [1, 2])

    def test_empty_candidate_is_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            ra.sequence_logprob([None, {1: -1.0}], 1, [])


# --------------------------------------------------------------------------
# F. Margin sign
# --------------------------------------------------------------------------


class TestMarginSign:
    def test_higher_a_likelihood_gives_positive_margin(self):
        assert ra.canonical_margin(-0.2, -1.9) > 0

    def test_higher_b_likelihood_gives_negative_margin(self):
        assert ra.canonical_margin(-2.4, -0.3) < 0

    def test_margin_is_the_difference(self):
        assert ra.canonical_margin(-1.0, -4.0) == pytest.approx(3.0)

    def test_positive_margin_means_met(self):
        assert ra.semantic_decision(ra.canonical_margin(-0.1, -3.0)) == 1

    def test_non_finite_scores_are_rejected(self):
        with pytest.raises(ValueError, match="non-finite"):
            ra.canonical_margin(float("-inf"), -1.0)


# --------------------------------------------------------------------------
# G. Crossing rule matches Day 3
# --------------------------------------------------------------------------


class TestCrossingRule:
    def test_first_negative_level_is_the_crossing(self):
        curve = np.array([[3.0, 2.0, 1.0, -0.5, -2.0, -3.0, -4.0, -5.0, -6.0]])
        assert int(crossing_index(curve)[0]) == 3

    def test_no_negative_level_gives_the_sentinel(self):
        curve = np.ones((1, N_STRICTNESS))
        assert int(crossing_index(curve)[0]) == NO_CROSSING_INDEX == 9

    def test_all_negative_gives_zero(self):
        curve = -np.ones((1, N_STRICTNESS))
        assert int(crossing_index(curve)[0]) == 0

    def test_matches_day3_convention_string(self):
        assert rc.CROSSING_CONVENTION.startswith("j_hat = min{s : M(s) < 0}")


# --------------------------------------------------------------------------
# H. Tie rule -- M = 0 counts as met
# --------------------------------------------------------------------------


class TestTieRule:
    def test_zero_margin_counts_as_met(self):
        assert ra.semantic_decision(0.0) == 1

    def test_zero_margin_does_not_trigger_a_crossing(self):
        curve = np.array([[2.0, 1.0, 0.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0]])
        assert int(crossing_index(curve)[0]) == 4

    def test_negative_zero_still_counts_as_met(self):
        assert ra.semantic_decision(-0.0) == 1

    def test_tie_does_not_create_a_prefix_record_low(self):
        """Equality never makes a crossing reachable under the strict rule."""
        curve = [5.0, 3.0, 3.0, 3.0, 1.0, 0.5, 0.4, 0.3, 0.2]
        assert 2 not in day3.prefix_record_lows(curve)
        assert 3 not in day3.prefix_record_lows(curve)


# --------------------------------------------------------------------------
# I. Prefix-record reachability, strict, as in Day 3
# --------------------------------------------------------------------------


class TestPrefixRecordReachability:
    def test_strictly_decreasing_curve_makes_every_level_reachable(self):
        curve = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
        assert day3.prefix_record_lows(curve) == (1, 2, 3, 4, 5, 6, 7, 8)

    def test_a_rise_blocks_the_level(self):
        curve = [5.0, 4.0, 6.0, 3.0, 2.0, 1.0, 0.5, 0.25, 0.1]
        assert 2 not in day3.prefix_record_lows(curve)
        assert 3 in day3.prefix_record_lows(curve)

    def test_extremes_are_always_reachable(self):
        curve = [1.0] * N_STRICTNESS
        reach = day3.reachable_crossings(curve)
        assert 0 in reach and NO_CROSSING_INDEX in reach

    def test_theorem_and_enumeration_agree(self):
        rng = np.random.default_rng(rc.EXPERIMENT_SEED)
        for _ in range(200):
            curve = rng.normal(size=N_STRICTNESS)
            assert day3.reachable_crossings(curve) == (
                day3.reachable_crossings_by_enumeration(curve)
            )


# --------------------------------------------------------------------------
# J. RC calculation on known synthetic curves
# --------------------------------------------------------------------------


class TestReachableCount:
    def test_monotone_decreasing_curve_reaches_every_crossing(self):
        curve = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
        assert day3.reachable_count(curve) == 10

    def test_flat_curve_reaches_only_the_extremes(self):
        assert day3.reachable_count([2.0] * N_STRICTNESS) == 2

    def test_single_record_low_adds_exactly_one(self):
        curve = [5.0, 5.0, 5.0, 1.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        assert day3.reachable_count(curve) == 3
        assert day3.reachable_crossings(curve) == (0, 3, 9)

    def test_rc_equals_ten_requires_all_eight_record_lows(self):
        curve = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
        assert len(day3.prefix_record_lows(curve)) == 8
        assert day3.reachable_count(curve) == 10


# --------------------------------------------------------------------------
# K. Day-3 dataset identity
# --------------------------------------------------------------------------


class TestDatasetIdentity:
    def test_stored_partition_matches_its_recorded_content_hash(self, repo_root: Path):
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

    def test_rebuilt_partition_is_identical_to_the_stored_one(
        self, repo_root: Path, day3_test
    ):
        """Rebuilding from seed 45 must reproduce the frozen partition exactly."""
        import pandas as pd

        stored = pd.read_parquet(repo_root / "artifacts/day3/test_dataset.parquet")
        assert ds.dataset_content_hash(day3_test) == ds.dataset_content_hash(stored)

    def test_design_dimensions(self, repo_root: Path):
        import pandas as pd

        stored = pd.read_parquet(repo_root / "artifacts/day3/test_dataset.parquet")
        assert len(stored) == rc.N_ROWS == 2916
        assert stored["artifact_id"].nunique() == rc.N_ARTIFACTS == 324
        counts = stored.groupby("artifact_id")["strictness_index"].nunique()
        assert set(counts.unique()) == {rc.N_STRICTNESS_LEVELS}
        per_family = stored.drop_duplicates("artifact_id")["family"].value_counts()
        assert set(per_family.unique()) == {108}

    def test_nontrivial_curve_count(self, repo_root: Path):
        import pandas as pd

        stored = pd.read_parquet(repo_root / "artifacts/day3/test_dataset.parquet")
        per_artifact = stored.drop_duplicates("artifact_id")
        nontrivial = per_artifact["true_first_fail_index"].between(1, 8).sum()
        assert int(nontrivial) == rc.N_NONTRIVIAL == 288

    def test_thirty_six_artifacts_at_each_true_crossing(self, repo_root: Path):
        import pandas as pd

        stored = pd.read_parquet(repo_root / "artifacts/day3/test_dataset.parquet")
        counts = stored.drop_duplicates("artifact_id")[
            "true_first_fail_index"
        ].value_counts()
        for crossing in range(1, 10):
            assert int(counts[crossing]) == 36


# --------------------------------------------------------------------------
# L. Predicate reconstruction from primitive fields
# --------------------------------------------------------------------------


class TestPredicateReconstruction:
    @staticmethod
    def _canonical_z(family: str, latent: int) -> int:
        """z from the stored latent, per the canonical form in Section 5."""
        if family == "coverage":
            return int(latent)
        if family in ("max_violation", "numeric_tolerance"):
            return 8 - int(latent)
        raise AssertionError(f"unknown family {family!r}")

    def test_all_three_families_reproduce_true_crossing(self, repo_root: Path):
        import pandas as pd

        stored = pd.read_parquet(repo_root / "artifacts/day3/test_dataset.parquet")
        per_artifact = stored.drop_duplicates("artifact_id")
        for row in per_artifact.itertuples(index=False):
            z = self._canonical_z(row.family, row.latent_level)
            assert z + 1 == int(row.true_first_fail_index), row.artifact_id

    def test_formal_truth_reproduces_from_the_canonical_predicate(
        self, repo_root: Path
    ):
        import pandas as pd

        stored = pd.read_parquet(repo_root / "artifacts/day3/test_dataset.parquet")
        for row in stored.itertuples(index=False):
            z = self._canonical_z(row.family, row.latent_level)
            assert int(z >= int(row.strictness_index)) == int(row.formal_truth)

    def test_every_family_label_is_one_of_the_three(self, repo_root: Path):
        import pandas as pd

        stored = pd.read_parquet(repo_root / "artifacts/day3/test_dataset.parquet")
        assert set(stored["family"].unique()) == {
            "coverage",
            "max_violation",
            "numeric_tolerance",
        }


# --------------------------------------------------------------------------
# M. Frozen-record integrity
# --------------------------------------------------------------------------


class TestFrozenRecordIntegrity:
    def test_fingerprint_round_trips(self, repo_root: Path):
        fingerprint = ra.build_frozen_fingerprint(repo_root)
        assert fingerprint["n_files"] > 100
        verdict = ra.verify_frozen_fingerprint(fingerprint, repo_root)
        assert verdict["status"] == "PASS", verdict

    def test_labelswap_record_is_covered(self, repo_root: Path):
        fingerprint = ra.build_frozen_fingerprint(repo_root)
        covered = fingerprint["files"]
        assert "artifacts/labelswap/preregistration.json" in covered
        assert "docs/RESULTS_LABEL_SWAP.md" in covered

    def test_prior_preregistration_checksums_still_verify(self, repo_root: Path):
        for study in ("day1", "day2", "day3", "labelswap"):
            json_path = repo_root / f"artifacts/{study}/preregistration.json"
            sha_path = repo_root / f"artifacts/{study}/preregistration.sha256"
            if not (json_path.is_file() and sha_path.is_file()):
                continue
            assert provenance.verify_checksum_file(json_path, sha_path), study

    def test_stored_fingerprint_matches_when_present(self, repo_root: Path):
        """Once the anchor study freezes, its own fingerprint must keep verifying."""
        stored = repo_root / rc.FROZEN_FINGERPRINT_PATH
        if not stored.is_file():
            pytest.skip("anchor fingerprint not yet written")
        fingerprint = json.loads(stored.read_text(encoding="utf-8"))
        verdict = ra.verify_frozen_fingerprint(fingerprint, repo_root)
        assert verdict["status"] == "PASS", verdict


# --------------------------------------------------------------------------
# Curve completeness -- mechanical exclusion only
# --------------------------------------------------------------------------


class TestCurveCompleteness:
    @staticmethod
    def _frame(margins_by_artifact):
        import pandas as pd

        rows = []
        for artifact, margins in margins_by_artifact.items():
            for level, margin in enumerate(margins):
                rows.append(
                    {
                        "artifact_id": artifact,
                        "strictness_index": level,
                        "margin": margin,
                    }
                )
        return pd.DataFrame(rows)

    def test_full_curve_is_complete(self):
        frame = self._frame({"A": [1.0] * N_STRICTNESS})
        assert ra.curve_completeness(frame).complete == ["A"]

    def test_a_missing_margin_excludes_the_curve(self):
        margins = [1.0] * N_STRICTNESS
        margins[4] = float("nan")
        frame = self._frame({"A": margins})
        completeness = ra.curve_completeness(frame)
        assert completeness.complete == []
        assert completeness.incomplete == {"A": [4]}

    def test_exclusion_does_not_depend_on_margin_value(self):
        """Extreme but present margins stay in; only absence excludes."""
        frame = self._frame({"A": [-99.0, 99.0] + [0.0] * (N_STRICTNESS - 2)})
        assert ra.curve_completeness(frame).complete == ["A"]

    def test_complete_rows_filters_to_complete_curves_only(self):
        good = [1.0] * N_STRICTNESS
        bad = [1.0] * N_STRICTNESS
        bad[0] = float("nan")
        frame = self._frame({"A": good, "B": bad})
        kept = ra.complete_rows(frame)
        assert set(kept["artifact_id"]) == {"A"}
