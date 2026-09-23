"""Deterministic construction of the frozen FormalCRRC Day-1 dataset.

The dataset is a full crossing of

    3 predicate families x 9 latent levels x 12 instances = 324 artifacts

with each artifact evaluated at all 9 strictness levels, giving 2916
rubric-response pairs. Construction is a pure function of
:data:`formalcrrc.config.SEED`; rerunning this module reproduces the file
byte-for-byte.

Two invariants are enforced by construction and re-checked by the test suite:

* the ``candidate_response`` of an artifact is byte-identical at all nine
  strictness levels -- only the rubric threshold moves;
* every family-A/B candidate response carries exactly
  :data:`~formalcrrc.config.MARKERS_PER_RESPONSE` markers, so response length
  carries no information about the latent level.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np
import pandas as pd

from formalcrrc import predicates
from formalcrrc.config import (
    FAMILIES,
    FAMILY_CODES,
    FAMILY_COVERAGE,
    FAMILY_MAX_VIOLATION,
    FAMILY_NUMERIC_TOLERANCE,
    LIST_SIZE,
    MARKER_ALPHABET,
    MARKER_CODE_LEN,
    MARKER_PREFIX,
    MARKERS_PER_RESPONSE,
    N_INSTANCES,
    N_LATENT_LEVELS,
    N_STRICTNESS,
    NUMERIC_TARGET_MAX,
    NUMERIC_TARGET_MIN,
    SEED,
)

# --------------------------------------------------------------------------
# Candidate response templates
# --------------------------------------------------------------------------
#
# Families A and B deliberately share a response format. The two families then
# differ only in the direction of the rubric threshold, not in the surface form
# of the artifact being judged.

RESPONSE_TEMPLATE_MARKERS: str = (
    "Submission record.\n"
    "The following markers are present in this submission:\n"
    "{marker_lines}\n"
    "End of submission record."
)

RESPONSE_TEMPLATE_NUMERIC: str = (
    "Measurement report.\n"
    "After completing the procedure, the value reported by this submission "
    "is {value}.\n"
    "End of measurement report."
)


#: Attempts allowed when redrawing a family-C target away from a forbidden pair.
MAX_TARGET_REDRAWS: int = 1000


def sha256_text(text: str) -> str:
    """Return the SHA-256 hex digest of ``text`` encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _artifact_id(
    family_code: str, latent_level: int, instance_index: int, partition: str | None
) -> str:
    """Artifact identifier, optionally namespaced by partition.

    Day 1 passes no partition and keeps its original ``A-L3-I07`` form. Day-2
    partitions prefix their tag so identifiers can never collide across
    experiments. The identifier is never shown to a model: prompts contain only
    the rubric and the candidate response.
    """
    stem = f"{family_code}-L{latent_level}-I{instance_index:02d}"
    return f"{partition}-{stem}" if partition else stem


# --------------------------------------------------------------------------
# Artifact construction
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Artifact:
    """One fixed evaluated artifact together with its formal construction data."""

    artifact_id: str
    family: str
    family_code: str
    latent_level: int
    instance_index: int
    candidate_response: str
    true_first_fail_index: int
    rng_provenance: str
    #: Family A/B only: the eight items of the Required / Prohibited list.
    list_items: tuple[str, ...] = ()
    #: Family A/B only: the markers actually emitted by the response.
    response_markers: tuple[str, ...] = ()
    #: Family C only.
    target_value: int | None = None
    reported_value: int | None = None
    error_sign: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def item_block(self) -> str:
        """The Required / Prohibited list as it appears inside the rubric."""
        return "\n".join(f"- {item}" for item in self.list_items)


def _rng(
    family: str, latent_level: int, instance_index: int, seed: int = SEED
) -> np.random.Generator:
    """Return the artifact-specific generator derived from the master seed.

    seed defaults to the Day-1 master seed, so the default call reproduces
    the Day-1 dataset exactly. Day-2 partitions pass their own seeds.
    """
    family_index = FAMILIES.index(family)
    return np.random.default_rng([seed, family_index, latent_level, instance_index])


def _rng_provenance(
    family: str, latent_level: int, instance_index: int, seed: int = SEED
) -> str:
    family_index = FAMILIES.index(family)
    return (
        f"numpy.default_rng([{seed}, {family_index}, {latent_level}, "
        f"{instance_index}]) / PCG64"
    )


def _draw_unique_markers(rng: np.random.Generator, count: int) -> list[str]:
    """Draw ``count`` distinct marker tokens of the form ``MK-XXXX``."""
    alphabet = np.frombuffer(MARKER_ALPHABET.encode("ascii"), dtype="S1")
    seen: set[str] = set()
    markers: list[str] = []
    while len(markers) < count:
        letters = rng.choice(alphabet, size=MARKER_CODE_LEN, replace=True)
        code = b"".join(letters.tolist()).decode("ascii")
        token = f"{MARKER_PREFIX}{code}"
        if token in seen:
            continue
        seen.add(token)
        markers.append(token)
    return markers


def _build_marker_artifact(
    family: str,
    latent_level: int,
    instance_index: int,
    seed: int = SEED,
    partition: str | None = None,
) -> Artifact:
    """Construct one family-A or family-B artifact.

    ``latent_level`` is the number of list members the response contains: ``m``
    for coverage, ``v`` for max-violation. The response always emits
    ``MARKERS_PER_RESPONSE`` markers in total, padded with tokens that are not
    on the list.
    """
    rng = _rng(family, latent_level, instance_index, seed)
    pool = _draw_unique_markers(rng, LIST_SIZE + MARKERS_PER_RESPONSE)
    list_items = pool[:LIST_SIZE]
    off_list = pool[LIST_SIZE:]

    # Which list members appear, and which off-list fillers pad the response.
    on_list_order = rng.permutation(LIST_SIZE)
    present = [list_items[i] for i in on_list_order[:latent_level]]
    filler = off_list[: MARKERS_PER_RESPONSE - latent_level]

    markers = present + filler
    order = rng.permutation(len(markers))
    markers = [markers[i] for i in order]

    marker_lines = "\n".join(f"- {m}" for m in markers)
    response = RESPONSE_TEMPLATE_MARKERS.format(marker_lines=marker_lines)

    code = FAMILY_CODES[family]
    return Artifact(
        artifact_id=_artifact_id(code, latent_level, instance_index, partition),
        family=family,
        family_code=code,
        latent_level=latent_level,
        instance_index=instance_index,
        candidate_response=response,
        true_first_fail_index=predicates.true_first_fail_index(family, latent_level),
        rng_provenance=_rng_provenance(family, latent_level, instance_index, seed),
        list_items=tuple(list_items),
        response_markers=tuple(markers),
    )


def _build_numeric_artifact(
    latent_level: int,
    instance_index: int,
    seed: int = SEED,
    partition: str | None = None,
    forbidden_pairs: frozenset[tuple[int, int]] = frozenset(),
) -> Artifact:
    """Construct one family-C artifact with absolute error ``e = latent_level``.

    Targets vary across the twelve instances, and the sign of ``V - T``
    alternates with the instance index so that the error is not always in one
    direction. For ``e = 0`` the sign is recorded as 0.
    """
    family = FAMILY_NUMERIC_TOLERANCE
    rng = _rng(family, latent_level, instance_index, seed)
    sign = 0 if latent_level == 0 else (1 if instance_index % 2 == 0 else -1)
    # Family C draws an integer target from a bounded range, so two independent
    # draws can coincide. Redraw while the (target, reported) pair is already
    # spoken for, which keeps partitions content-disjoint. The default empty
    # forbidden set reproduces Day-1 behaviour exactly.
    for _ in range(MAX_TARGET_REDRAWS):
        target = int(rng.integers(NUMERIC_TARGET_MIN, NUMERIC_TARGET_MAX + 1))
        reported = target + sign * latent_level
        if (target, reported) not in forbidden_pairs:
            break
    else:
        raise RuntimeError(
            f"could not draw an unused family-C target for L{latent_level} "
            f"I{instance_index} within {MAX_TARGET_REDRAWS} attempts"
        )

    response = RESPONSE_TEMPLATE_NUMERIC.format(value=reported)

    code = FAMILY_CODES[family]
    return Artifact(
        artifact_id=_artifact_id(code, latent_level, instance_index, partition),
        family=family,
        family_code=code,
        latent_level=latent_level,
        instance_index=instance_index,
        candidate_response=response,
        true_first_fail_index=predicates.true_first_fail_index(family, latent_level),
        rng_provenance=_rng_provenance(family, latent_level, instance_index, seed),
        target_value=target,
        reported_value=reported,
        error_sign=sign,
    )


def build_artifact(
    family: str,
    latent_level: int,
    instance_index: int,
    seed: int = SEED,
    partition: str | None = None,
    forbidden_pairs: frozenset[tuple[int, int]] = frozenset(),
) -> Artifact:
    """Construct a single artifact deterministically.

    The defaults reproduce Day 1 exactly. ``forbidden_pairs`` applies only to
    family C, whose bounded integer target range makes coincidental duplicates
    possible between independently seeded partitions.
    """
    if family in (FAMILY_COVERAGE, FAMILY_MAX_VIOLATION):
        return _build_marker_artifact(
            family, latent_level, instance_index, seed, partition
        )
    if family == FAMILY_NUMERIC_TOLERANCE:
        return _build_numeric_artifact(
            latent_level, instance_index, seed, partition, forbidden_pairs
        )
    raise ValueError(f"unknown predicate family: {family!r}")


def iter_artifacts(
    seed: int = SEED,
    partition: str | None = None,
    forbidden_pairs: frozenset[tuple[int, int]] = frozenset(),
    enforce_unique_targets: bool = False,
) -> Iterator[Artifact]:
    """Yield all 324 artifacts in canonical (family, latent, instance) order.

    With ``enforce_unique_targets`` the family-C (target, reported) pairs drawn
    so far are accumulated into the forbidden set as generation proceeds, so a
    partition contains no two artifacts with the same pair. Day 1 was generated
    without this rule and must keep being reproducible without it, so it is off
    by default.
    """
    used = set(forbidden_pairs)
    for family in FAMILIES:
        for latent_level in range(N_LATENT_LEVELS):
            for instance_index in range(N_INSTANCES):
                artifact = build_artifact(
                    family,
                    latent_level,
                    instance_index,
                    seed,
                    partition,
                    frozenset(used),
                )
                if enforce_unique_targets and artifact.target_value is not None:
                    used.add((artifact.target_value, artifact.reported_value))
                yield artifact


# --------------------------------------------------------------------------
# Dataset assembly
# --------------------------------------------------------------------------

DATASET_COLUMNS: tuple[str, ...] = (
    "prompt_id",
    "artifact_id",
    "family",
    "family_code",
    "latent_level",
    "instance_index",
    "strictness_index",
    "raw_threshold",
    "candidate_response",
    "rubric_text",
    "formal_truth",
    "true_first_fail_index",
    "seed",
    "rng_provenance",
    "candidate_sha256",
    "rubric_sha256",
    "rubric_normalized_sha256",
    "list_items_json",
    "response_markers_json",
    "target_value",
    "reported_value",
    "error_sign",
)


def build_dataset(
    seed: int = SEED,
    partition: str | None = None,
    forbidden_pairs: frozenset[tuple[int, int]] = frozenset(),
    enforce_unique_targets: bool = False,
) -> pd.DataFrame:
    """Build a complete 2916-row dataset as a DataFrame.

    Called with no arguments this reproduces the frozen Day-1 dataset exactly,
    including its content hash. Day-2 partitions pass their own seed, partition
    tag and forbidden-pair set.
    """
    rows: list[dict[str, Any]] = []
    for artifact in iter_artifacts(
        seed, partition, forbidden_pairs, enforce_unique_targets
    ):
        candidate_hash = sha256_text(artifact.candidate_response)
        for strictness in range(N_STRICTNESS):
            rubric = predicates.render_rubric(
                artifact.family,
                strictness,
                item_block=artifact.item_block if artifact.list_items else None,
                target=artifact.target_value,
            )
            normalized = predicates.normalize_rubric(artifact.family, rubric)
            rows.append(
                {
                    "prompt_id": f"{artifact.artifact_id}|s{strictness}",
                    "artifact_id": artifact.artifact_id,
                    "family": artifact.family,
                    "family_code": artifact.family_code,
                    "latent_level": artifact.latent_level,
                    "instance_index": artifact.instance_index,
                    "strictness_index": strictness,
                    "raw_threshold": predicates.raw_threshold(
                        artifact.family, strictness
                    ),
                    "candidate_response": artifact.candidate_response,
                    "rubric_text": rubric,
                    "formal_truth": predicates.formal_truth(
                        artifact.family, artifact.latent_level, strictness
                    ),
                    "true_first_fail_index": artifact.true_first_fail_index,
                    "seed": seed,
                    "rng_provenance": artifact.rng_provenance,
                    "candidate_sha256": candidate_hash,
                    "rubric_sha256": sha256_text(rubric),
                    "rubric_normalized_sha256": sha256_text(normalized),
                    "list_items_json": json.dumps(list(artifact.list_items)),
                    "response_markers_json": json.dumps(
                        list(artifact.response_markers)
                    ),
                    "target_value": artifact.target_value,
                    "reported_value": artifact.reported_value,
                    "error_sign": artifact.error_sign,
                }
            )

    frame = pd.DataFrame(rows, columns=list(DATASET_COLUMNS))
    for column in ("target_value", "reported_value", "error_sign"):
        frame[column] = frame[column].astype("Int64")
    for column in (
        "latent_level",
        "instance_index",
        "strictness_index",
        "raw_threshold",
        "formal_truth",
        "true_first_fail_index",
        "seed",
    ):
        frame[column] = frame[column].astype("int64")
    return frame.sort_values(["artifact_id", "strictness_index"]).reset_index(drop=True)


def dataset_content_hash(frame: pd.DataFrame) -> str:
    """Format-independent identity of the dataset's content.

    Hashes a canonical text serialisation of every row rather than the parquet
    file, so the identity survives a change of writer version while still
    detecting any change to a single character of any field.
    """
    ordered = frame.sort_values(["artifact_id", "strictness_index"])
    digest = hashlib.sha256()
    for column in DATASET_COLUMNS:
        digest.update(f"#{column}\n".encode("utf-8"))
    for row in ordered.itertuples(index=False):
        record = "\x1f".join(
            "" if value is None or value is pd.NA else str(value)
            for value in row
        )
        digest.update(record.encode("utf-8"))
        digest.update(b"\x1e")
    return digest.hexdigest()
