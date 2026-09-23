"""Judge prompt construction and the frozen inference order.

One minimal user message per rubric-response pair. The semantic instructions are
byte-identical for every model; the rubric threshold is the only study-variable
text. Model-specific chat templates are applied at inference time and their
rendered form is hashed for provenance, but the message content built here does
not depend on the model.

No chain-of-thought, no few-shot examples, no persona, no model-specific
reasoning instructions, no temperature variation.
"""

from __future__ import annotations

import hashlib

import pandas as pd

from formalcrrc.config import (
    INFERENCE_ORDER_SALT,
    MODEL_IDS,
    MODEL_ORDER_SALT,
    PROMPT_TEMPLATE,
)


#: Placeholders used by the label-tokenization reference prompt.
LABEL_REFERENCE_RUBRIC: str = "<RUBRIC>"
LABEL_REFERENCE_RESPONSE: str = "<RESPONSE>"


def build_user_message(rubric_text: str, candidate_response: str) -> str:
    """Render the single user message shown to every judge."""
    return PROMPT_TEMPLATE.format(
        rubric_text=rubric_text, candidate_response=candidate_response
    )


def label_reference_message() -> str:
    """A structurally identical prompt carrying no study content.

    Answer-label tokens are resolved against this message, so the resolution
    depends on the prompt's chat-template structure rather than on any rubric,
    threshold or candidate response.
    """
    return build_user_message(LABEL_REFERENCE_RUBRIC, LABEL_REFERENCE_RESPONSE)


def prompt_template_sha256() -> str:
    """SHA-256 of the frozen prompt template itself."""
    return hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()


def _order_key(value: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}|{value}".encode("utf-8")).hexdigest()


def inference_order_key(prompt_id: str) -> str:
    """Deterministic hash key fixing where a prompt sits in the run order.

    Sorting by this key scrambles family, latent level and strictness so that no
    run proceeds in an outcome-grouped order.
    """
    return _order_key(prompt_id, INFERENCE_ORDER_SALT)


def model_run_order() -> list[str]:
    """Deterministic, hash-derived order in which the judge models are run."""
    return sorted(MODEL_IDS, key=lambda m: _order_key(m, MODEL_ORDER_SALT))


def build_prompt_manifest(dataset: pd.DataFrame) -> pd.DataFrame:
    """Build the frozen prompt manifest, including the fixed inference order.

    The manifest is written before any study inference and is the authority on
    both the prompt text and the order in which prompts are scored.
    """
    frame = dataset[
        [
            "prompt_id",
            "artifact_id",
            "family",
            "latent_level",
            "strictness_index",
        ]
    ].copy()
    frame["user_message"] = [
        build_user_message(rubric, response)
        for rubric, response in zip(
            dataset["rubric_text"], dataset["candidate_response"], strict=True
        )
    ]
    frame["user_message_sha256"] = [
        hashlib.sha256(text.encode("utf-8")).hexdigest()
        for text in frame["user_message"]
    ]
    frame["order_key"] = [inference_order_key(pid) for pid in frame["prompt_id"]]
    frame = frame.sort_values("order_key").reset_index(drop=True)
    frame.insert(0, "order_index", range(len(frame)))
    return frame
