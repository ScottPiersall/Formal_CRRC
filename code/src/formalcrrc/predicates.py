"""Formal predicate families, rubric rendering, and programmatic ground truth.

Every label produced here is computed by an arithmetic predicate over quantities
that are fixed at construction time. No language model participates in labelling,
and no label depends on how any rubric string happens to be read.

Three families, each with nine ordered strictness levels ``s = 0..8``:

======================  ===============================  =====================
family                  rubric threshold at strictness s  formal truth
======================  ===============================  =====================
``coverage``            ``at least s`` required items     ``1[m >= s]``
``max_violation``       ``at most 8 - s`` prohibited      ``1[v <= 8 - s]``
``numeric_tolerance``   ``at most 8 - s`` absolute error  ``1[e <= 8 - s]``
======================  ===============================  =====================

In all three families ``s = 0`` is the most permissive level and ``s = 8`` the
most strict. Family A's numeric threshold rises with strictness while families
B and C move in the opposite direction; that asymmetry is deliberate.
"""

from __future__ import annotations

import re
from typing import Final

from formalcrrc.config import (
    FAMILIES,
    FAMILY_COVERAGE,
    FAMILY_MAX_VIOLATION,
    FAMILY_NUMERIC_TOLERANCE,
    N_STRICTNESS,
    NO_CROSSING_INDEX,
    SCALE_MAX,
)

# --------------------------------------------------------------------------
# Rubric templates
# --------------------------------------------------------------------------
#
# The ``{threshold}`` slot is the ONLY field permitted to vary across the nine
# rubric variants belonging to one artifact. Everything else -- including the
# item list and the family-C target value -- is fixed per artifact.

RUBRIC_TEMPLATE_COVERAGE: Final[str] = (
    "Required Items list:\n"
    "{item_block}\n"
    "\n"
    "The criterion is met if and only if the candidate response contains at "
    "least {threshold} of the items in the Required Items list. Count each "
    "required item at most once."
)

RUBRIC_TEMPLATE_MAX_VIOLATION: Final[str] = (
    "Prohibited Items list:\n"
    "{item_block}\n"
    "\n"
    "The criterion is met if and only if the candidate response contains at "
    "most {threshold} of the items in the Prohibited Items list. Count each "
    "prohibited item at most once."
)

RUBRIC_TEMPLATE_NUMERIC_TOLERANCE: Final[str] = (
    "Target value: {target}\n"
    "\n"
    "The criterion is met if and only if the absolute difference between the "
    "value reported in the candidate response and the target value is at most "
    "{threshold}."
)

RUBRIC_TEMPLATES: Final[dict[str, str]] = {
    FAMILY_COVERAGE: RUBRIC_TEMPLATE_COVERAGE,
    FAMILY_MAX_VIOLATION: RUBRIC_TEMPLATE_MAX_VIOLATION,
    FAMILY_NUMERIC_TOLERANCE: RUBRIC_TEMPLATE_NUMERIC_TOLERANCE,
}

#: Placeholder substituted for the threshold during rubric normalisation.
THRESHOLD_PLACEHOLDER: Final[str] = "<THRESHOLD>"

#: Family-specific patterns locating the single permitted threshold field in a
#: *rendered* rubric string. These operate on the final text, not on the
#: template, so the normalisation test is not circular: it would fail if
#: generation ever varied anything other than the captured field.
THRESHOLD_PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    FAMILY_COVERAGE: re.compile(r"(?<=contains at least )(\d+)(?= of the items)"),
    FAMILY_MAX_VIOLATION: re.compile(r"(?<=contains at most )(\d+)(?= of the items)"),
    FAMILY_NUMERIC_TOLERANCE: re.compile(r"(?<=target value is at most )(\d+)(?=\.)"),
}


# --------------------------------------------------------------------------
# Threshold mapping
# --------------------------------------------------------------------------


def raw_threshold(family: str, strictness: int) -> int:
    """Return the numeric threshold printed in the rubric at strictness ``s``.

    Family A prints the minimum required count ``s``; families B and C print the
    allowance ``8 - s``, which *decreases* as strictness increases.
    """
    _check_family(family)
    _check_strictness(strictness)
    if family == FAMILY_COVERAGE:
        return strictness
    return SCALE_MAX - strictness


# --------------------------------------------------------------------------
# Formal truth
# --------------------------------------------------------------------------


def formal_truth(family: str, latent_level: int, strictness: int) -> int:
    """Return the exact binary label for one (artifact, threshold) pair.

    ``latent_level`` is ``m`` (coverage), ``v`` (max violation) or ``e``
    (numeric tolerance). The label is arithmetic, not textual.
    """
    _check_family(family)
    _check_latent(latent_level)
    _check_strictness(strictness)
    if family == FAMILY_COVERAGE:
        return int(latent_level >= strictness)
    return int(latent_level <= SCALE_MAX - strictness)


def truth_curve(family: str, latent_level: int) -> list[int]:
    """Return the nine formal labels for one artifact, ordered ``s = 0..8``."""
    return [formal_truth(family, latent_level, s) for s in range(N_STRICTNESS)]


def true_first_fail_index(family: str, latent_level: int) -> int:
    """Return ``j*``, the smallest strictness at which the formal label is 0.

    Returns :data:`~formalcrrc.config.NO_CROSSING_INDEX` (9) when the artifact
    satisfies all nine thresholds. Closed form: ``m + 1`` for coverage and
    ``9 - latent`` for the other two families; computed here by scanning the
    curve so that the closed forms can be independently checked in tests.
    """
    curve = truth_curve(family, latent_level)
    for s, y in enumerate(curve):
        if y == 0:
            return s
    return NO_CROSSING_INDEX


# --------------------------------------------------------------------------
# Rubric rendering and normalisation
# --------------------------------------------------------------------------


def render_rubric(
    family: str,
    strictness: int,
    *,
    item_block: str | None = None,
    target: int | None = None,
) -> str:
    """Render the rubric text for one (artifact, threshold) pair."""
    _check_family(family)
    _check_strictness(strictness)
    threshold = raw_threshold(family, strictness)
    if family == FAMILY_NUMERIC_TOLERANCE:
        if target is None:
            raise ValueError("numeric_tolerance rubrics require a target value")
        return RUBRIC_TEMPLATES[family].format(target=target, threshold=threshold)
    if item_block is None:
        raise ValueError(f"{family} rubrics require an item_block")
    return RUBRIC_TEMPLATES[family].format(item_block=item_block, threshold=threshold)


def normalize_rubric(family: str, rubric_text: str) -> str:
    """Replace the one permitted threshold field with a fixed placeholder.

    Raises :class:`ValueError` unless the family's threshold pattern matches
    exactly once, which guards against a rubric whose threshold field is
    missing, duplicated, or ambiguous.
    """
    _check_family(family)
    pattern = THRESHOLD_PATTERNS[family]
    matches = pattern.findall(rubric_text)
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one threshold field in a {family} rubric, "
            f"found {len(matches)}"
        )
    return pattern.sub(THRESHOLD_PLACEHOLDER, rubric_text)


def extract_threshold(family: str, rubric_text: str) -> int:
    """Read the threshold back out of a rendered rubric string."""
    _check_family(family)
    matches = THRESHOLD_PATTERNS[family].findall(rubric_text)
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one threshold field in a {family} rubric, "
            f"found {len(matches)}"
        )
    return int(matches[0])


# --------------------------------------------------------------------------
# Internal validation
# --------------------------------------------------------------------------


def _check_family(family: str) -> None:
    if family not in FAMILIES:
        raise ValueError(f"unknown predicate family: {family!r}")


def _check_strictness(strictness: int) -> None:
    if not 0 <= strictness <= SCALE_MAX:
        raise ValueError(f"strictness index out of range: {strictness}")


def _check_latent(latent_level: int) -> None:
    if not 0 <= latent_level <= SCALE_MAX:
        raise ValueError(f"latent level out of range: {latent_level}")
