"""The label-swap intervention: prompt construction, scoring, pairing, metrics.

Exactly one conceptual factor moves relative to Day 3 -- which response token
carries which semantic class:

===============  ==========================  ==========================
condition        response instruction        canonical margin
===============  ==========================  ==========================
original         Return A if met / B if not  ``M_O(s) = S_A - S_B``
swapped          Return B if met / A if not  ``M_S(s) = S_B - S_A``
===============  ==========================  ==========================

In both conditions a **positive canonical margin means 'criterion met'**, so
every downstream convention -- the crossing rule, the decision rule, the
reachability rule -- applies unchanged and the two conditions are directly
comparable. Using ``S_A - S_B`` for the swapped condition would invert the
semantics; :func:`swapped_margin` exists so that mistake cannot be made silently,
and the test suite asserts the sign.

The swapped prompt is *derived from* the frozen Day-1/Day-3 template rather than
retyped, and :func:`prompt_diff` proves that the two templates differ at exactly
the two label characters.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from formalcrrc import day2, day3, predicates, prompts, scoring
from formalcrrc.config import (
    LABEL_MET,
    LABEL_NOT_MET,
    N_STRICTNESS,
    NO_CROSSING_INDEX,
    PROMPT_TEMPLATE,
)
from formalcrrc.day2_bootstrap import crossing_index
from formalcrrc.labelswap_config import PERMITTED_TEMPLATE_DIFF_POSITIONS

# --------------------------------------------------------------------------
# The prompt intervention
# --------------------------------------------------------------------------

#: The two response-instruction lines as the frozen template ends. Built from
#: the config label constants, not retyped, so a change to either label would
#: propagate rather than silently disagree.
ORIGINAL_RESPONSE_INSTRUCTION: str = (
    f"Return {LABEL_MET} if the criterion is met.\n"
    f"Return {LABEL_NOT_MET} if the criterion is not met."
)

#: The same two lines with the semantic assignment reversed.
SWAPPED_RESPONSE_INSTRUCTION: str = (
    f"Return {LABEL_NOT_MET} if the criterion is met.\n"
    f"Return {LABEL_MET} if the criterion is not met."
)


def swapped_prompt_template() -> str:
    """The Day-3 template with only the response-label assignment reversed.

    Derived from :data:`~formalcrrc.config.PROMPT_TEMPLATE` by replacing its
    final two lines, so every other character is carried over by construction
    rather than by transcription.
    """
    if not PROMPT_TEMPLATE.endswith(ORIGINAL_RESPONSE_INSTRUCTION):
        raise ValueError(
            "the frozen prompt template does not end with the expected response "
            "instruction; the swap cannot be derived safely"
        )
    head = PROMPT_TEMPLATE[: -len(ORIGINAL_RESPONSE_INSTRUCTION)]
    return head + SWAPPED_RESPONSE_INSTRUCTION


def build_swapped_user_message(rubric_text: str, candidate_response: str) -> str:
    """Render one swapped-condition user message."""
    return swapped_prompt_template().format(
        rubric_text=rubric_text, candidate_response=candidate_response
    )


def swapped_prompt_template_sha256() -> str:
    return scoring.sha256_text(swapped_prompt_template())


def swapped_label_reference_message() -> str:
    """Structurally identical reference prompt carrying no study content."""
    return build_swapped_user_message(
        prompts.LABEL_REFERENCE_RUBRIC, prompts.LABEL_REFERENCE_RESPONSE
    )


def prompt_diff(original: str | None = None, swapped: str | None = None) -> dict[str, Any]:
    """Character-level diff between the two templates, with a verdict.

    The intervention is admissible only if the templates have equal length and
    differ at exactly the two label characters -- one where ``met`` is assigned
    and one where ``not met`` is.
    """
    original = PROMPT_TEMPLATE if original is None else original
    swapped = swapped_prompt_template() if swapped is None else swapped

    same_length = len(original) == len(swapped)
    positions = (
        [
            {"index": i, "original": a, "swapped": b}
            for i, (a, b) in enumerate(zip(original, swapped))
            if a != b
        ]
        if same_length
        else []
    )
    only_labels = all(
        {entry["original"], entry["swapped"]} == {LABEL_MET, LABEL_NOT_MET}
        for entry in positions
    )
    expected_count = len(positions) == PERMITTED_TEMPLATE_DIFF_POSITIONS

    return {
        "original_sha256": scoring.sha256_text(original),
        "swapped_sha256": scoring.sha256_text(swapped),
        "same_length": same_length,
        "n_differing_positions": len(positions),
        "differing_positions": positions,
        "all_differences_are_label_characters": only_labels,
        "expected_difference_count": PERMITTED_TEMPLATE_DIFF_POSITIONS,
        "original_response_instruction": ORIGINAL_RESPONSE_INSTRUCTION,
        "swapped_response_instruction": SWAPPED_RESPONSE_INSTRUCTION,
        "non_label_text_identical": _non_label_text_identical(original, swapped),
        "status": "PASS"
        if (same_length and expected_count and only_labels)
        else "FAIL",
    }


def _non_label_text_identical(original: str, swapped: str) -> bool:
    """Everything outside the two response-instruction lines is byte-identical."""
    head_o = original[: -len(ORIGINAL_RESPONSE_INSTRUCTION)]
    head_s = swapped[: -len(SWAPPED_RESPONSE_INSTRUCTION)]
    return head_o == head_s


def build_swapped_prompt_manifest(dataset: pd.DataFrame) -> pd.DataFrame:
    """Swapped prompt manifest over the frozen Day-3 rows.

    Reuses the Day-3 inference order keys so the two conditions are scored in
    the same scrambled order; only ``user_message`` differs.
    """
    frame = dataset[
        ["prompt_id", "artifact_id", "family", "latent_level", "strictness_index"]
    ].copy()
    frame["user_message"] = [
        build_swapped_user_message(rubric, response)
        for rubric, response in zip(
            dataset["rubric_text"], dataset["candidate_response"], strict=True
        )
    ]
    frame["user_message_sha256"] = [
        scoring.sha256_text(text) for text in frame["user_message"]
    ]
    frame["order_key"] = [prompts.inference_order_key(p) for p in frame["prompt_id"]]
    frame = frame.sort_values("order_key").reset_index(drop=True)
    frame.insert(0, "order_index", range(len(frame)))
    return frame


def verify_row_equivalence(
    dataset: pd.DataFrame, swapped_manifest: pd.DataFrame
) -> dict[str, Any]:
    """Per-row proof that only the A/B mapping differs.

    For every artifact-threshold pair the original and swapped user messages are
    compared after removing the response instruction. Everything before it --
    criterion text, rubric threshold, candidate response, formatting -- must be
    byte-identical.
    """
    original = prompts.build_prompt_manifest(dataset).set_index("prompt_id")
    swapped = swapped_manifest.set_index("prompt_id")

    if set(original.index) != set(swapped.index):
        return {"status": "FAIL", "reason": "prompt id sets differ"}

    mismatches: list[str] = []
    tail_ok = 0
    for prompt_id in original.index:
        o_msg = original.loc[prompt_id, "user_message"]
        s_msg = swapped.loc[prompt_id, "user_message"]
        if not o_msg.endswith(ORIGINAL_RESPONSE_INSTRUCTION):
            mismatches.append(f"{prompt_id}: original tail unexpected")
            continue
        if not s_msg.endswith(SWAPPED_RESPONSE_INSTRUCTION):
            mismatches.append(f"{prompt_id}: swapped tail unexpected")
            continue
        head_o = o_msg[: -len(ORIGINAL_RESPONSE_INSTRUCTION)]
        head_s = s_msg[: -len(SWAPPED_RESPONSE_INSTRUCTION)]
        if head_o != head_s:
            mismatches.append(f"{prompt_id}: body differs outside the label lines")
            continue
        tail_ok += 1

    metadata_ok = all(
        original[column].equals(swapped.loc[original.index, column])
        for column in ("artifact_id", "family", "latent_level", "strictness_index")
    )

    return {
        "n_rows": int(len(original)),
        "n_bodies_identical": tail_ok,
        "metadata_identical": bool(metadata_ok),
        "mismatches": mismatches[:20],
        "n_mismatches": len(mismatches),
        "status": "PASS"
        if not mismatches and metadata_ok and tail_ok == len(original)
        else "FAIL",
    }


# --------------------------------------------------------------------------
# Swapped-condition tokenisation
# --------------------------------------------------------------------------


def resolve_swapped_label_tokenization(
    tokenizer: Any, rendered_reference_prompt: str
) -> scoring.LabelTokenization:
    """Resolve answer-label tokens with the semantic assignment reversed.

    The *tokens* are the same two the Day-3 panel used; what changes is which of
    them means "criterion met". Building a
    :class:`~formalcrrc.scoring.LabelTokenization` this way makes
    ``scoring.score_prompt`` return ``raw_score_met = S_B`` and
    ``raw_score_not_met = S_A``, so the recorded margin is ``S_B - S_A``: the
    canonical swapped margin, positive for "met", with no sign fix-up anywhere
    downstream.
    """
    met_ids = scoring.label_continuation_ids(
        tokenizer, rendered_reference_prompt, LABEL_NOT_MET
    )
    not_met_ids = scoring.label_continuation_ids(
        tokenizer, rendered_reference_prompt, LABEL_MET
    )
    return scoring.LabelTokenization(
        label_met=LABEL_NOT_MET,
        label_not_met=LABEL_MET,
        met_token_ids=met_ids,
        not_met_token_ids=not_met_ids,
        single_token=len(met_ids) == 1 and len(not_met_ids) == 1,
        naive_met_token_ids=tuple(
            tokenizer.encode(LABEL_NOT_MET, add_special_tokens=False)
        ),
        naive_not_met_token_ids=tuple(
            tokenizer.encode(LABEL_MET, add_special_tokens=False)
        ),
    )


def tokenization_is_day3_mirror(
    swapped: scoring.LabelTokenization, day3_record: dict[str, Any]
) -> dict[str, Any]:
    """Check the swapped tokenisation is the Day-3 one with the roles exchanged.

    Same two token ids, opposite semantic assignment. Anything else -- a
    different id, a lost single-token equivalence, a changed scoring rule --
    means the conditions are not comparable.
    """
    checks = {
        "met_is_day3_not_met_token": list(swapped.met_token_ids)
        == list(day3_record["not_met_token_ids"]),
        "not_met_is_day3_met_token": list(swapped.not_met_token_ids)
        == list(day3_record["met_token_ids"]),
        "labels_exchanged": (
            swapped.label_met == day3_record["label_not_met"]
            and swapped.label_not_met == day3_record["label_met"]
        ),
        "single_token_equivalence_preserved": bool(swapped.single_token)
        == bool(day3_record["single_token_equivalence"]),
        "scoring_method_unchanged": swapped.scoring_method
        == day3_record["scoring_method"],
    }
    return {"checks": checks, "status": "PASS" if all(checks.values()) else "FAIL"}


# --------------------------------------------------------------------------
# Canonical margins -- the sign convention that must not be got wrong
# --------------------------------------------------------------------------


def original_margin(logit_a: float | np.ndarray, logit_b: float | np.ndarray):
    """``M_O = S_A - S_B``. Positive means 'criterion met' under the original map."""
    return np.asarray(logit_a, dtype=float) - np.asarray(logit_b, dtype=float)


def swapped_margin(logit_a: float | np.ndarray, logit_b: float | np.ndarray):
    """``M_S = S_B - S_A``. Positive means 'criterion met' under the swapped map.

    Deliberately **not** ``S_A - S_B``. Under the swapped instruction the token
    ``B`` is the one the judge emits when it believes the criterion is met, so
    the canonical semantic margin reverses with the mapping.
    """
    return np.asarray(logit_b, dtype=float) - np.asarray(logit_a, dtype=float)


def canonical_margin(logit_a, logit_b, condition: str):
    """Dispatch on condition so the sign is never chosen by hand at a call site."""
    from formalcrrc.labelswap_config import CONDITION_ORIGINAL, CONDITION_SWAPPED

    if condition == CONDITION_ORIGINAL:
        return original_margin(logit_a, logit_b)
    if condition == CONDITION_SWAPPED:
        return swapped_margin(logit_a, logit_b)
    raise ValueError(f"unknown condition: {condition!r}")


def semantic_decision(margins) -> np.ndarray:
    """``D(s) = 1[M(s) >= 0]``, the Day-3 convention: margin 0 counts as met."""
    return (np.asarray(margins, dtype=float) >= 0.0).astype(int)


# --------------------------------------------------------------------------
# Curve assembly and paired metrics
# --------------------------------------------------------------------------


def curve_block(scores: pd.DataFrame, margin_column: str = "margin"):
    """Return ``(artifact_ids, margins)`` sorted by artifact then strictness."""
    ordered = scores.sort_values(["artifact_id", "strictness_index"])
    ids = ordered["artifact_id"].unique()
    block = (
        ordered.pivot(
            index="artifact_id", columns="strictness_index", values=margin_column
        )
        .loc[ids]
        .to_numpy(dtype=float)
    )
    if block.shape[1] != N_STRICTNESS:
        raise ValueError(f"expected {N_STRICTNESS} strictness levels")
    return np.asarray(ids), block


def curve_metrics(margins: Sequence[float], j_star: int) -> dict[str, Any]:
    """Every per-curve quantity, computed with the unchanged Day-3 machinery."""
    curve = np.asarray(margins, dtype=float)
    j_hat = int(crossing_index(curve[None, :])[0])
    reachable = day3.reachable_crossings(curve)
    nontrivial = 1 <= int(j_star) <= N_STRICTNESS - 1
    return {
        "j_hat": j_hat,
        "tce": abs(j_hat - int(j_star)),
        "reachable_crossings": ",".join(str(j) for j in reachable),
        "reachable_count": len(reachable),
        "prefix_record_rate": day3.prefix_record_rate(curve),
        "translation_reachable": bool(
            day3.translation_reachable(curve, int(j_star))
        ),
        "nontrivial_boundary": bool(nontrivial),
        "full_reachability": len(reachable) == NO_CROSSING_INDEX + 1,
        "mmvr": day2.margin_monotonicity_violation_rate(curve),
        "mmvm": day2.margin_reversal_magnitude(curve),
        "span": day2.response_span(curve),
        "margin_spearman": day2.margin_spearman(curve),
        "adjacent_equal_margins": int(np.sum(curve[1:] == curve[:-1])),
    }


def row_accuracy_and_brier(
    margins: Sequence[float], truth: Sequence[int]
) -> tuple[float, float]:
    """Row-level accuracy and Brier score from canonical margins."""
    curve = np.asarray(margins, dtype=float)
    labels = np.asarray(truth, dtype=int)
    probabilities = day2.sigmoid(curve)
    accuracy = float(np.mean(semantic_decision(curve) == labels))
    brier = float(np.mean((probabilities - labels) ** 2))
    return accuracy, brier


def reachable_set_agreement(a: str, b: str) -> tuple[bool, float]:
    """Exact-set agreement and Jaccard similarity between two reachable sets."""
    set_a = {int(v) for v in a.split(",") if v != ""}
    set_b = {int(v) for v in b.split(",") if v != ""}
    union = set_a | set_b
    jaccard = len(set_a & set_b) / len(union) if union else 1.0
    return set_a == set_b, jaccard


def true_first_fail(family: str, latent_level: int) -> int:
    """``j*`` recomputed from the predicate, never read from a stored column."""
    return predicates.true_first_fail_index(family, int(latent_level))
