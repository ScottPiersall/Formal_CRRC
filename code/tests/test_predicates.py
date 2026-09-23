"""Formal truth functions, threshold direction, and rubric normalisation."""

from __future__ import annotations

import pytest

from formalcrrc import predicates
from formalcrrc.config import (
    FAMILIES,
    FAMILY_COVERAGE,
    FAMILY_MAX_VIOLATION,
    FAMILY_NUMERIC_TOLERANCE,
    N_LATENT_LEVELS,
    N_STRICTNESS,
    NO_CROSSING_INDEX,
    SCALE_MAX,
)


class TestThresholdDirection:
    """Family A's printed threshold rises with strictness; B and C fall."""

    def test_coverage_threshold_equals_strictness(self):
        for s in range(N_STRICTNESS):
            assert predicates.raw_threshold(FAMILY_COVERAGE, s) == s

    @pytest.mark.parametrize(
        "family", [FAMILY_MAX_VIOLATION, FAMILY_NUMERIC_TOLERANCE]
    )
    def test_allowance_families_invert(self, family):
        for s in range(N_STRICTNESS):
            assert predicates.raw_threshold(family, s) == SCALE_MAX - s

    def test_most_permissive_and_most_strict_endpoints(self):
        # s = 0 is satisfied by every artifact in every family.
        for family in FAMILIES:
            for latent in range(N_LATENT_LEVELS):
                assert predicates.formal_truth(family, latent, 0) == 1
        # s = 8 is satisfied only at the extreme latent level.
        assert predicates.formal_truth(FAMILY_COVERAGE, 8, 8) == 1
        assert predicates.formal_truth(FAMILY_COVERAGE, 7, 8) == 0
        for family in (FAMILY_MAX_VIOLATION, FAMILY_NUMERIC_TOLERANCE):
            assert predicates.formal_truth(family, 0, 8) == 1
            assert predicates.formal_truth(family, 1, 8) == 0


class TestFormalTruth:
    def test_coverage_truth_matches_definition(self):
        for m in range(N_LATENT_LEVELS):
            for s in range(N_STRICTNESS):
                assert predicates.formal_truth(FAMILY_COVERAGE, m, s) == int(m >= s)

    def test_max_violation_truth_matches_definition(self):
        for v in range(N_LATENT_LEVELS):
            for s in range(N_STRICTNESS):
                expected = int(v <= SCALE_MAX - s)
                assert (
                    predicates.formal_truth(FAMILY_MAX_VIOLATION, v, s) == expected
                )

    def test_numeric_tolerance_truth_matches_definition(self):
        for e in range(N_LATENT_LEVELS):
            for s in range(N_STRICTNESS):
                expected = int(e <= SCALE_MAX - s)
                assert (
                    predicates.formal_truth(FAMILY_NUMERIC_TOLERANCE, e, s)
                    == expected
                )

    def test_every_curve_is_ones_then_zeros(self):
        for family in FAMILIES:
            for latent in range(N_LATENT_LEVELS):
                curve = predicates.truth_curve(family, latent)
                assert curve == sorted(curve, reverse=True), (family, latent, curve)

    def test_out_of_range_arguments_raise(self):
        with pytest.raises(ValueError):
            predicates.formal_truth("nonexistent", 0, 0)
        with pytest.raises(ValueError):
            predicates.formal_truth(FAMILY_COVERAGE, 9, 0)
        with pytest.raises(ValueError):
            predicates.formal_truth(FAMILY_COVERAGE, 0, 9)


class TestFirstFailIndex:
    """j* is the programmatically known crossing point for every latent level."""

    def test_coverage_closed_form(self):
        for m in range(N_LATENT_LEVELS):
            expected = NO_CROSSING_INDEX if m == SCALE_MAX else m + 1
            assert predicates.true_first_fail_index(FAMILY_COVERAGE, m) == expected

    @pytest.mark.parametrize(
        "family", [FAMILY_MAX_VIOLATION, FAMILY_NUMERIC_TOLERANCE]
    )
    def test_allowance_closed_form(self, family):
        for latent in range(N_LATENT_LEVELS):
            assert predicates.true_first_fail_index(family, latent) == 9 - latent

    def test_matches_scan_of_the_curve(self):
        for family in FAMILIES:
            for latent in range(N_LATENT_LEVELS):
                curve = predicates.truth_curve(family, latent)
                zeros = [s for s, y in enumerate(curve) if y == 0]
                expected = zeros[0] if zeros else NO_CROSSING_INDEX
                assert (
                    predicates.true_first_fail_index(family, latent) == expected
                )

    def test_no_crossing_only_at_the_permissive_extreme(self):
        assert (
            predicates.true_first_fail_index(FAMILY_COVERAGE, 8)
            == NO_CROSSING_INDEX
        )
        assert (
            predicates.true_first_fail_index(FAMILY_MAX_VIOLATION, 0)
            == NO_CROSSING_INDEX
        )
        assert (
            predicates.true_first_fail_index(FAMILY_NUMERIC_TOLERANCE, 0)
            == NO_CROSSING_INDEX
        )


class TestRubricNormalisation:
    ITEM_BLOCK = "\n".join(f"- MK-AAA{c}" for c in "ABCDEFGH")

    def test_only_the_threshold_varies_for_marker_families(self):
        for family in (FAMILY_COVERAGE, FAMILY_MAX_VIOLATION):
            normalized = {
                predicates.normalize_rubric(
                    family,
                    predicates.render_rubric(
                        family, s, item_block=self.ITEM_BLOCK
                    ),
                )
                for s in range(N_STRICTNESS)
            }
            assert len(normalized) == 1

    def test_only_the_threshold_varies_for_numeric_family(self):
        normalized = {
            predicates.normalize_rubric(
                FAMILY_NUMERIC_TOLERANCE,
                predicates.render_rubric(
                    FAMILY_NUMERIC_TOLERANCE, s, target=137
                ),
            )
            for s in range(N_STRICTNESS)
        }
        assert len(normalized) == 1

    def test_numeric_normalisation_leaves_the_target_alone(self):
        rubric = predicates.render_rubric(
            FAMILY_NUMERIC_TOLERANCE, 3, target=137
        )
        normalized = predicates.normalize_rubric(FAMILY_NUMERIC_TOLERANCE, rubric)
        assert "137" in normalized
        assert predicates.THRESHOLD_PLACEHOLDER in normalized

    def test_threshold_reads_back_out_of_the_rendered_rubric(self):
        for family in FAMILIES:
            for s in range(N_STRICTNESS):
                rubric = predicates.render_rubric(
                    family,
                    s,
                    item_block=self.ITEM_BLOCK,
                    target=137,
                )
                assert predicates.extract_threshold(family, rubric) == (
                    predicates.raw_threshold(family, s)
                )

    def test_normalisation_rejects_a_rubric_without_a_threshold_field(self):
        with pytest.raises(ValueError):
            predicates.normalize_rubric(FAMILY_COVERAGE, "no threshold here")

    def test_normalisation_rejects_a_duplicated_threshold_field(self):
        rubric = predicates.render_rubric(
            FAMILY_COVERAGE, 3, item_block=self.ITEM_BLOCK
        )
        with pytest.raises(ValueError):
            predicates.normalize_rubric(FAMILY_COVERAGE, rubric + "\n" + rubric)

    def test_a_smuggled_change_defeats_normalisation(self):
        """Normalisation must not mask an edit outside the threshold field."""
        a = predicates.normalize_rubric(
            FAMILY_COVERAGE,
            predicates.render_rubric(
                FAMILY_COVERAGE, 3, item_block=self.ITEM_BLOCK
            ),
        )
        b = predicates.normalize_rubric(
            FAMILY_COVERAGE,
            predicates.render_rubric(
                FAMILY_COVERAGE,
                4,
                item_block=self.ITEM_BLOCK.replace("MK-AAAA", "MK-ZZZZ"),
            ),
        )
        assert a != b

    def test_rendering_requires_family_specific_inputs(self):
        with pytest.raises(ValueError):
            predicates.render_rubric(FAMILY_COVERAGE, 0)
        with pytest.raises(ValueError):
            predicates.render_rubric(FAMILY_NUMERIC_TOLERANCE, 0)
