"""The scalar latent-estimation / step-function proposition.

**Post-study theoretical analysis**, not preregistered. These tests verify the
proposition's own logic and its compatibility with the Day-3 reachability
theorem. They use synthetic curves and the frozen Day-3 outputs read-only; no
inference runs and no frozen file is touched.

The five required synthetic cases are A-E below. The point of the pair (B, C) is
the distinction the whole proposition rests on: a *wrong location* keeps every
crossing reachable, a *wrong ordering* does not.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from formalcrrc import day3, predicates, step_proposition as sp
from formalcrrc.config import FAMILIES, N_STRICTNESS, NO_CROSSING_INDEX

# --- the required synthetic curves -----------------------------------------

#: A. A coherent scalar readout: strictly decreasing.
SCALAR_STEP = np.array([5.0, 4, 3, 2, 1, -1, -2, -3, -4])

#: B. Still coherent, but the zero crossing sits in the wrong place.
MISLOCATED = np.array([4.0, 3, 2, 1, 0, -1, -2, -3, -4])

#: C. An ordering failure: margin rises from 2 to 3 at level 3.
ORDER_FAILURE = np.array([5.0, 4, 2, 3, 1, 0, -1, -2, -3])

#: D. An adjacent equality at level 2.
TIED = np.array([5.0, 4, 4, 3, 2, 1, 0, -1, -2])

TRANSLATIONS = (-100.0, -3.5, -1.0, 0.0, 0.25, 2.0, 17.0, 1e6)


# --------------------------------------------------------------------------
# Task 1 -- canonicalization of the three predicate families
# --------------------------------------------------------------------------


class TestCanonicalization:
    def test_canonicalization_passes_overall(self):
        assert sp.verify_canonicalization()["status"] == "PASS"

    def test_canonical_thresholds_are_strictly_increasing(self):
        assert sp.thresholds_strictly_increasing()

    @pytest.mark.parametrize("family", FAMILIES)
    def test_canonical_labels_reproduce_the_implementation(self, family):
        for latent in range(N_STRICTNESS):
            assert list(sp.canonical_labels(family, latent)) == predicates.truth_curve(
                family, latent
            )

    @pytest.mark.parametrize("family", FAMILIES)
    def test_canonical_first_fail_reproduces_the_implementation(self, family):
        for latent in range(N_STRICTNESS):
            assert sp.canonical_first_fail(family, latent) == (
                predicates.true_first_fail_index(family, latent)
            )

    @pytest.mark.parametrize("family", FAMILIES)
    def test_j_star_equals_z_plus_one(self, family):
        for latent in range(N_STRICTNESS):
            z = sp.canonical_compliance(family, latent)
            assert predicates.true_first_fail_index(family, latent) == z + 1

    def test_coverage_needs_no_reflection(self):
        for latent in range(N_STRICTNESS):
            assert sp.canonical_compliance("coverage", latent) == latent

    @pytest.mark.parametrize("family", ["max_violation", "numeric_tolerance"])
    def test_defect_families_are_reflected(self, family):
        for latent in range(N_STRICTNESS):
            assert sp.canonical_compliance(family, latent) == 8 - latent

    @pytest.mark.parametrize("family", ["max_violation", "numeric_tolerance"])
    def test_raw_printed_threshold_decreases_before_canonicalization(self, family):
        """Why canonicalization is required rather than cosmetic."""
        raw = [predicates.raw_threshold(family, s) for s in range(N_STRICTNESS)]
        assert all(b < a for a, b in zip(raw, raw[1:]))

    def test_every_family_is_permissive_at_strictness_zero(self):
        """Guarantees j* >= 1, so the boundary support is 1..9."""
        for family in FAMILIES:
            for latent in range(N_STRICTNESS):
                assert predicates.formal_truth(family, latent, 0) == 1


# --------------------------------------------------------------------------
# Tier 1 -- labels form a nonincreasing step with at most one transition
# --------------------------------------------------------------------------


class TestTierOneStepStructure:
    @pytest.mark.parametrize("z_hat", [-2.0, -0.5, 0.0, 1.0, 3.7, 8.0, 12.0])
    def test_scalar_readout_labels_are_nonincreasing(self, z_hat):
        assert sp.is_nonincreasing_step(sp.scalar_model_labels(z_hat))

    @pytest.mark.parametrize("z_hat", [-2.0, -0.5, 0.0, 1.0, 3.7, 8.0, 12.0])
    def test_at_most_one_met_to_not_met_transition(self, z_hat):
        assert sp.transition_count(sp.scalar_model_labels(z_hat)) <= 1

    def test_labels_match_the_formal_curve_when_the_estimate_is_correct(self):
        for family in FAMILIES:
            for latent in range(N_STRICTNESS):
                z = sp.canonical_compliance(family, latent)
                assert list(sp.scalar_model_labels(z)) == predicates.truth_curve(
                    family, latent
                )

    def test_a_non_step_label_vector_is_rejected(self):
        assert not sp.is_nonincreasing_step([1, 1, 0, 1, 0, 0, 0, 0, 0])
        assert sp.transition_count([1, 1, 0, 1, 0, 0, 0, 0, 0]) == 2


# --------------------------------------------------------------------------
# Tier 2 -- strictly increasing g forces strictly decreasing margins
# --------------------------------------------------------------------------


class TestTierTwoMarginStructure:
    @pytest.mark.parametrize(
        "g",
        [
            None,
            np.tanh,
            lambda d: 3.0 * d + 1.5,
            lambda d: d**3,
            lambda d: 1.0 / (1.0 + np.exp(-d)),
        ],
        ids=["identity", "tanh", "affine", "cube", "logistic"],
    )
    @pytest.mark.parametrize("z_hat", [-1.0, 0.0, 2.5, 4.0, 9.0])
    def test_any_strictly_increasing_g_gives_strictly_decreasing_margins(
        self, g, z_hat
    ):
        assert sp.is_strictly_decreasing(sp.scalar_model_margins(z_hat, g))

    @pytest.mark.parametrize("z_hat", [-1.0, 0.0, 2.5, 4.0, 9.0])
    def test_scalar_model_reaches_every_crossing(self, z_hat):
        margins = sp.scalar_model_margins(z_hat)
        assert sp.implies_full_reachability(margins)
        assert set(day3.reachable_crossings(margins)) == set(range(0, 10))

    @pytest.mark.parametrize("z_hat", [-1.0, 0.0, 2.5, 4.0, 9.0])
    def test_scalar_model_never_falsified_for_any_boundary(self, z_hat):
        """TR_x = 1 for every possible formal boundary, which is the proposition."""
        margins = sp.scalar_model_margins(z_hat)
        for j_star in range(1, NO_CROSSING_INDEX + 1):
            assert not sp.scalar_model_is_falsified(margins, j_star)

    def test_strictly_decreasing_iff_all_noninitial_are_record_lows(self):
        """The equivalence the audit relies on, checked on random curves."""
        rng = np.random.default_rng(20260908)
        for _ in range(2000):
            curve = rng.normal(size=N_STRICTNESS) * rng.choice([0.1, 1.0, 10.0])
            assert sp.is_strictly_decreasing(curve) == (
                sp.all_noninitial_are_prefix_record_lows(curve)
            )

    def test_full_reachability_iff_strictly_decreasing(self):
        rng = np.random.default_rng(11)
        for _ in range(2000):
            curve = rng.integers(-4, 5, size=N_STRICTNESS).astype(float)
            assert sp.is_strictly_decreasing(curve) == sp.implies_full_reachability(
                curve
            )


# --------------------------------------------------------------------------
# A. Correct scalar-step curve
# --------------------------------------------------------------------------


class TestCaseA_ScalarStep:
    def test_every_noninitial_index_is_a_strict_prefix_record_low(self):
        assert day3.prefix_record_lows(SCALAR_STEP) == tuple(range(1, 9))

    def test_curve_is_strictly_decreasing(self):
        assert sp.is_strictly_decreasing(SCALAR_STEP)

    def test_all_ten_crossings_are_reachable(self):
        assert day3.reachable_crossings(SCALAR_STEP) == tuple(range(0, 10))
        assert day3.reachable_count(SCALAR_STEP) == 10

    def test_prefix_record_rate_is_one(self):
        assert day3.prefix_record_rate(SCALAR_STEP) == 1.0

    def test_no_boundary_can_falsify_this_curve(self):
        for j_star in range(1, NO_CROSSING_INDEX + 1):
            assert day3.translation_reachable(SCALAR_STEP, j_star)


# --------------------------------------------------------------------------
# B. Mislocated but coherent -- wrong location, right ordering
# --------------------------------------------------------------------------


class TestCaseB_MislocatedButCoherent:
    def test_the_crossing_is_where_the_convention_says(self):
        from formalcrrc.day2_bootstrap import crossing_index

        assert int(crossing_index(MISLOCATED[None, :])[0]) == 5

    def test_raw_tce_is_nonzero_against_an_earlier_boundary(self):
        j_star = 2
        j_hat = 5
        assert abs(j_hat - j_star) == 3

    def test_all_candidate_crossings_remain_reachable(self):
        assert day3.reachable_crossings(MISLOCATED) == tuple(range(0, 10))

    def test_oracle_recovers_the_boundary_exactly_despite_raw_error(self):
        """Wrong location is fully correctable by translation; ordering is intact."""
        for j_star in range(1, NO_CROSSING_INDEX + 1):
            assert day3.oracle_translation_tce(MISLOCATED, j_star) == 0
            assert day3.translation_reachable(MISLOCATED, j_star)

    def test_this_curve_does_not_falsify_the_scalar_model(self):
        for j_star in range(1, NO_CROSSING_INDEX + 1):
            assert not sp.scalar_model_is_falsified(MISLOCATED, j_star)


# --------------------------------------------------------------------------
# C. Ordering failure -- the boundary is unreachable
# --------------------------------------------------------------------------


class TestCaseC_OrderingFailure:
    def test_the_curve_is_not_strictly_decreasing(self):
        assert not sp.is_strictly_decreasing(ORDER_FAILURE)

    def test_level_three_is_not_a_strict_prefix_record_low(self):
        """Margin 3 at level 3 follows margin 2 at level 2, so it sets no record."""
        assert ORDER_FAILURE[3] == 3.0
        assert min(ORDER_FAILURE[:3]) == 2.0
        assert not ORDER_FAILURE[3] < min(ORDER_FAILURE[:3])
        assert 3 not in day3.prefix_record_lows(ORDER_FAILURE)

    def test_crossing_three_is_unreachable(self):
        assert 3 not in day3.reachable_crossings(ORDER_FAILURE)
        assert not day3.translation_reachable(ORDER_FAILURE, 3)

    def test_every_other_crossing_is_reachable(self):
        assert day3.reachable_crossings(ORDER_FAILURE) == (0, 1, 2, 4, 5, 6, 7, 8, 9)

    def test_the_scalar_model_is_falsified_for_that_boundary_only(self):
        assert sp.scalar_model_is_falsified(ORDER_FAILURE, 3)
        for j_star in (1, 2, 4, 5, 6, 7, 8, 9):
            assert not sp.scalar_model_is_falsified(ORDER_FAILURE, j_star)

    def test_no_translation_can_produce_the_missing_crossing(self):
        """Checked by brute force, independently of the theorem."""
        achieved = set()
        for alpha in np.linspace(-20, 20, 200001):
            shifted = ORDER_FAILURE + alpha
            below = shifted < 0.0
            achieved.add(int(np.argmax(below)) if below.any() else 9)
        assert 3 not in achieved

    def test_oracle_tce_is_bounded_away_from_zero_at_that_boundary(self):
        assert day3.oracle_translation_tce(ORDER_FAILURE, 3) == 1


# --------------------------------------------------------------------------
# D. Ties -- equality is not a strict record low
# --------------------------------------------------------------------------


class TestCaseD_Ties:
    def test_the_later_tied_index_is_not_a_strict_record_low(self):
        assert TIED[1] == TIED[2]
        assert 1 in day3.prefix_record_lows(TIED)
        assert 2 not in day3.prefix_record_lows(TIED)

    def test_the_tied_crossing_is_unreachable(self):
        assert 2 not in day3.reachable_crossings(TIED)
        assert not day3.translation_reachable(TIED, 2)

    def test_a_tie_violates_the_strict_scalar_model(self):
        """Strictly increasing g and tau force strict decrease, so ties cannot arise."""
        assert not sp.is_strictly_decreasing(TIED)
        assert not sp.implies_full_reachability(TIED)

    def test_tie_diagnostics_report_the_equality(self):
        record = day3.tie_diagnostics(TIED, 2)
        assert record["adjacent_equal_margins"] == 1
        assert record["prefix_tie_levels"] == 1
        assert record["target_blocked_by_equality"] is True

    def test_no_scalar_model_curve_ever_contains_a_tie(self):
        for z_hat in np.linspace(-3, 12, 121):
            margins = sp.scalar_model_margins(float(z_hat))
            assert len(set(margins)) == N_STRICTNESS

    def test_brute_force_confirms_the_tied_crossing_is_unreachable(self):
        achieved = set()
        for alpha in np.linspace(-20, 20, 200001):
            shifted = TIED + alpha
            below = shifted < 0.0
            achieved.add(int(np.argmax(below)) if below.any() else 9)
        assert 2 not in achieved


# --------------------------------------------------------------------------
# E. Translation invariance
# --------------------------------------------------------------------------


class TestCaseE_TranslationInvariance:
    @pytest.mark.parametrize(
        "curve", [SCALAR_STEP, MISLOCATED, ORDER_FAILURE, TIED], ids="A B C D".split()
    )
    @pytest.mark.parametrize("alpha", TRANSLATIONS)
    def test_prefix_record_set_is_translation_invariant(self, curve, alpha):
        assert day3.prefix_record_lows(curve + alpha) == day3.prefix_record_lows(curve)

    @pytest.mark.parametrize(
        "curve", [SCALAR_STEP, MISLOCATED, ORDER_FAILURE, TIED], ids="A B C D".split()
    )
    @pytest.mark.parametrize("alpha", TRANSLATIONS)
    def test_reachable_set_is_translation_invariant(self, curve, alpha):
        assert day3.reachable_crossings(curve + alpha) == day3.reachable_crossings(
            curve
        )

    @pytest.mark.parametrize(
        "curve", [SCALAR_STEP, MISLOCATED, ORDER_FAILURE, TIED], ids="A B C D".split()
    )
    @pytest.mark.parametrize("alpha", TRANSLATIONS)
    def test_strict_decrease_is_translation_invariant(self, curve, alpha):
        assert sp.is_strictly_decreasing(curve + alpha) == sp.is_strictly_decreasing(
            curve
        )

    @pytest.mark.parametrize("alpha", TRANSLATIONS)
    def test_falsification_verdict_is_translation_invariant(self, alpha):
        for j_star in range(1, NO_CROSSING_INDEX + 1):
            assert sp.scalar_model_is_falsified(
                ORDER_FAILURE + alpha, j_star
            ) == sp.scalar_model_is_falsified(ORDER_FAILURE, j_star)

    def test_scaling_by_a_positive_constant_also_preserves_the_set(self):
        """Any strictly increasing affine readout leaves the ordering alone."""
        for scale in (0.01, 0.5, 3.0, 250.0):
            assert day3.reachable_crossings(
                ORDER_FAILURE * scale
            ) == day3.reachable_crossings(ORDER_FAILURE)


# --------------------------------------------------------------------------
# Task 2 -- compatibility with the Day-3 implementation
# --------------------------------------------------------------------------


class TestDay3Compatibility:
    def test_crossings_zero_and_nine_are_always_reachable(self):
        rng = np.random.default_rng(7)
        for _ in range(500):
            curve = rng.normal(size=N_STRICTNESS)
            reachable = day3.reachable_crossings(curve)
            assert 0 in reachable and NO_CROSSING_INDEX in reachable

    def test_reachability_uses_strict_inequality(self):
        """Equality must not create a reachable crossing."""
        flat = np.zeros(N_STRICTNESS)
        assert day3.prefix_record_lows(flat) == ()
        assert day3.reachable_crossings(flat) == (0, NO_CROSSING_INDEX)

    def test_index_zero_is_never_a_prefix_record_low(self):
        """Level 0 has no prefix, so records are enumerated over 1..8 only."""
        for curve in (SCALAR_STEP, MISLOCATED, ORDER_FAILURE, TIED):
            assert 0 not in day3.prefix_record_lows(curve)

    def test_theorem_agrees_with_explicit_enumeration(self):
        rng = np.random.default_rng(4242)
        for _ in range(1500):
            curve = rng.integers(-3, 4, size=N_STRICTNESS).astype(float)
            assert set(day3.reachable_crossings(curve)) == set(
                day3.reachable_crossings_by_enumeration(curve)
            )

    def test_zero_crossing_convention_matches_day2(self):
        """A crossing is the first strictly negative level, else 9."""
        from formalcrrc.day2_bootstrap import crossing_index

        cases = {
            (5.0, 4, 3, 2, 1, -1, -2, -3, -4): 5,
            (1.0, 1, 1, 1, 1, 1, 1, 1, 1): 9,
            (0.0, 0, 0, 0, 0, 0, 0, 0, 0): 9,
            (-1.0, 2, 3, 4, 5, 6, 7, 8, 9): 0,
        }
        for curve, expected in cases.items():
            assert int(crossing_index(np.array(curve)[None, :])[0]) == expected

    def test_exactly_zero_margin_counts_as_met(self):
        """M = 0 gives P = 0.5, which the convention treats as not-yet-crossed."""
        from formalcrrc.day2_bootstrap import crossing_index

        curve = np.array([2.0, 1, 0, -1, -2, -3, -4, -5, -6])
        assert int(crossing_index(curve[None, :])[0]) == 3


# --------------------------------------------------------------------------
# Task 3 -- frozen Day-3 counts, read-only
# --------------------------------------------------------------------------

#: Recomputed from frozen raw margins; must equal the frozen Day-3 report.
EXPECTED_TRR = {
    "Qwen/Qwen2.5-14B-Instruct": Fraction(23, 48),
    "meta-llama/Llama-3.1-8B-Instruct": Fraction(13, 48),
    "mistralai/Mistral-7B-Instruct-v0.3": Fraction(1, 3),
    "google/gemma-2-9b-it": Fraction(29, 48),
}


@pytest.fixture(scope="module")
def day3_metrics(repo_root: Path):
    import pandas as pd

    path = repo_root / "artifacts/day3/metrics_by_artifact.parquet"
    if not path.is_file():
        pytest.skip("Day-3 metrics not present")
    return pd.read_parquet(path)


class TestFrozenDay3Counts:
    def test_nontrivial_boundary_means_j_star_in_one_to_eight(self, day3_metrics):
        nontrivial = day3_metrics["nontrivial_boundary"]
        expected = day3_metrics["true_first_fail_index"].between(1, 8)
        assert (nontrivial == expected).all()

    def test_each_judge_has_288_nontrivial_curves(self, day3_metrics):
        for _, group in day3_metrics.groupby("model_id"):
            assert int(group["nontrivial_boundary"].sum()) == 288

    @pytest.mark.parametrize("model_id", sorted(EXPECTED_TRR))
    def test_trr_matches_the_frozen_report(self, day3_metrics, model_id):
        subset = day3_metrics[
            (day3_metrics["model_id"] == model_id)
            & (day3_metrics["nontrivial_boundary"])
        ]
        observed = Fraction(
            int(subset["translation_reachable"].sum()), int(len(subset))
        )
        assert observed == EXPECTED_TRR[model_id]

    @pytest.mark.parametrize("model_id", sorted(EXPECTED_TRR))
    def test_translation_reachable_recomputes_from_margins(
        self, day3_metrics, model_id, repo_root: Path
    ):
        """Rebuild TR from the raw margins rather than trusting the stored flag."""
        import pandas as pd

        slug = {
            "Qwen/Qwen2.5-14B-Instruct": "qwen2_5_14b_instruct",
            "meta-llama/Llama-3.1-8B-Instruct": "llama3_1_8b_instruct",
            "mistralai/Mistral-7B-Instruct-v0.3": "mistral_7b_instruct_v0_3",
            "google/gemma-2-9b-it": "gemma_2_9b_it",
        }[model_id]
        path = repo_root / f"artifacts/day3/raw_scores/{slug}.parquet"
        if not path.is_file():
            pytest.skip("raw scores not present")
        scores = pd.read_parquet(path)
        stored = day3_metrics[day3_metrics["model_id"] == model_id].set_index(
            "artifact_id"
        )
        for artifact_id, group in scores.groupby("artifact_id"):
            group = group.sort_values("strictness_index")
            curve = group["margin"].to_numpy(dtype=float)
            j_star = predicates.true_first_fail_index(
                group["family"].iloc[0], int(group["latent_level"].iloc[0])
            )
            assert bool(day3.translation_reachable(curve, j_star)) == bool(
                stored.loc[artifact_id, "translation_reachable"]
            )

    def test_no_judge_is_a_pure_scalar_readout_on_most_curves(self, day3_metrics):
        """Full reachability (RC = 10) is the strict model's actual signature."""
        for _, group in day3_metrics.groupby("model_id"):
            share = float((group["reachable_count"] == 10).mean())
            assert 0.0 <= share < 0.05


class TestTheoryArtifacts:
    @staticmethod
    @pytest.fixture(scope="module")
    def audit(repo_root: Path):
        path = repo_root / "artifacts/poststudy_theory/step_proposition_audit.json"
        if not path.is_file():
            pytest.skip("run scripts/audit_step_proposition.py")
        return json.loads(path.read_text(encoding="utf-8"))

    def test_status_is_pass(self, audit):
        assert audit["status"] == "PASS"
        assert all(audit["checks"].values())

    def test_labelled_post_study_not_preregistered(self, audit):
        assert audit["preregistered"] is False
        assert audit["model_inference_performed"] is False

    def test_recorded_trr_matches_the_module(self, audit):
        for row in audit["empirical"]["by_model"]:
            assert Fraction(row["trr_exact"]) == EXPECTED_TRR[row["model_id"]]
