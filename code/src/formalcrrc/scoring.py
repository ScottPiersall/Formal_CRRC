"""Probability extraction for the A/B judge decision.

The Day-1 endpoint is probabilistic. For each prompt the judge's probability that
the criterion is met is read off the model's own scores for the two answer
labels; nothing is sampled and nothing is generated.

The answer-label tokens are resolved as *continuations* of a rendered prompt --
``encode(prompt + label)`` minus ``encode(prompt)`` -- not by encoding the bare
label string. The two differ for SentencePiece tokenizers, which return a
word-boundary-marked token for a string encoded in isolation; scoring that token
would read the wrong logit.

Primary rule (``single_token_next_logit``), used when both ``"A"`` and ``"B"``
continue the prompt with exactly one token for a given model::

    P_met = exp(z_A) / (exp(z_A) + exp(z_B))

with ``z`` the raw next-token logits at the first generated position.

Preregistered fallback (``sequence_loglikelihood``), used for any model where
single-token equivalence fails::

    P_met = exp(l_A) / (exp(l_A) + exp(l_B))

with ``l`` the exact continuation log-likelihood of the label string. The two
rules coincide up to a constant when the label really is one token, so the
fallback is a strict generalisation rather than a different measurement.

The rule is chosen per model from tokenizer behaviour alone, before any study
prompt is scored, and is recorded alongside every score.

Torch and transformers are imported lazily so that dataset construction, metrics
and analysis run in environments without them.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

from formalcrrc.config import (
    LABEL_MET,
    LABEL_NOT_MET,
    SCORING_FALLBACK,
    SCORING_PRIMARY,
)


# --------------------------------------------------------------------------
# Pure numeric core (no torch)
# --------------------------------------------------------------------------


def normalized_probability(score_met: float, score_not_met: float) -> float:
    """Normalise two log-scale scores into ``P(met)``.

    Computed as a numerically stable two-way softmax; identical arithmetic is
    used for the primary logit rule and the log-likelihood fallback.
    """
    if not (math.isfinite(score_met) and math.isfinite(score_not_met)):
        raise ValueError(
            f"non-finite label scores: met={score_met!r} not_met={score_not_met!r}"
        )
    delta = score_met - score_not_met
    if delta >= 0.0:
        return 1.0 / (1.0 + math.exp(-delta))
    exp_delta = math.exp(delta)
    return exp_delta / (1.0 + exp_delta)


# --------------------------------------------------------------------------
# Label tokenisation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelTokenization:
    """The tokens a model must emit for each answer label.

    ``met_token_ids`` / ``not_met_token_ids`` are *continuation* ids: what
    appears when the label follows the rendered prompt. ``naive_*`` records what
    encoding the bare label string would have given. The two differ whenever a
    tokenizer attaches a word-boundary marker to a string encoded in isolation,
    and it is the continuation ids that index the right logit.
    """

    label_met: str
    label_not_met: str
    met_token_ids: tuple[int, ...]
    not_met_token_ids: tuple[int, ...]
    single_token: bool
    naive_met_token_ids: tuple[int, ...] = ()
    naive_not_met_token_ids: tuple[int, ...] = ()

    @property
    def differs_from_naive(self) -> bool:
        """True when encoding the bare label would have indexed a different token."""
        return (
            self.met_token_ids != self.naive_met_token_ids
            or self.not_met_token_ids != self.naive_not_met_token_ids
        )

    @property
    def scoring_method(self) -> str:
        """The preregistered scoring rule implied by this tokenisation."""
        return SCORING_PRIMARY if self.single_token else SCORING_FALLBACK

    def to_dict(self) -> dict[str, Any]:
        return {
            "label_met": self.label_met,
            "label_not_met": self.label_not_met,
            "met_token_ids": list(self.met_token_ids),
            "not_met_token_ids": list(self.not_met_token_ids),
            "naive_met_token_ids": list(self.naive_met_token_ids),
            "naive_not_met_token_ids": list(self.naive_not_met_token_ids),
            "differs_from_naive_encoding": self.differs_from_naive,
            "single_token_equivalence": self.single_token,
            "scoring_method": self.scoring_method,
        }


def label_continuation_ids(
    tokenizer: Any, rendered_prompt: str, label: str
) -> tuple[int, ...]:
    """Token ids the model must emit for ``label`` to follow ``rendered_prompt``.

    Encoding a label on its own is not enough: a SentencePiece tokenizer will
    happily return ``"_A"`` for ``"A"``, which is a different token from the one
    that actually continues a prompt. Taking the difference between the encoded
    prompt and the encoded prompt-plus-label gives the token the model is really
    being asked about.
    """
    prompt_ids = list(tokenizer.encode(rendered_prompt, add_special_tokens=False))
    full_ids = list(
        tokenizer.encode(rendered_prompt + label, add_special_tokens=False)
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "appending the answer label changed the prompt's own tokenisation; "
            "the continuation cannot be isolated"
        )
    continuation = tuple(full_ids[len(prompt_ids) :])
    if not continuation:
        raise ValueError(f"answer label {label!r} produced an empty continuation")
    return continuation


def resolve_label_tokenization(
    tokenizer: Any, rendered_reference_prompt: str
) -> LabelTokenization:
    """Resolve the answer-label tokens against a rendered reference prompt.

    Single-token equivalence requires *both* labels to continue the prompt with
    exactly one token. The reference prompt shares the chat-template structure of
    every study prompt but carries no study content; stability of the result
    across real study prompts is verified separately, by
    ``scripts/verify_label_tokens.py``, using tokenizers alone.
    """
    met_ids = label_continuation_ids(tokenizer, rendered_reference_prompt, LABEL_MET)
    not_met_ids = label_continuation_ids(
        tokenizer, rendered_reference_prompt, LABEL_NOT_MET
    )
    return LabelTokenization(
        label_met=LABEL_MET,
        label_not_met=LABEL_NOT_MET,
        met_token_ids=met_ids,
        not_met_token_ids=not_met_ids,
        single_token=len(met_ids) == 1 and len(not_met_ids) == 1,
        naive_met_token_ids=tuple(
            tokenizer.encode(LABEL_MET, add_special_tokens=False)
        ),
        naive_not_met_token_ids=tuple(
            tokenizer.encode(LABEL_NOT_MET, add_special_tokens=False)
        ),
    )


# --------------------------------------------------------------------------
# Chat template rendering
# --------------------------------------------------------------------------


def render_chat_prompt(tokenizer: Any, user_message: str) -> str:
    """Apply the model's native chat template to the single user message."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message}],
        tokenize=False,
        add_generation_prompt=True,
    )


def sha256_text(text: str) -> str:
    """SHA-256 hex digest of ``text`` encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Model scoring (torch)
# --------------------------------------------------------------------------


def score_single_token(
    model: Any, tokenizer: Any, rendered_prompt: str, tokenization: LabelTokenization
) -> tuple[float, float, int]:
    """Return ``(z_met, z_not_met, n_prompt_tokens)`` from next-token logits."""
    import torch

    ids = tokenizer(rendered_prompt, return_tensors="pt", add_special_tokens=False)
    ids = {k: v.to(model.device) for k, v in ids.items()}
    with torch.no_grad():
        logits = model(**ids).logits[0, -1, :].float()
    z_met = float(logits[tokenization.met_token_ids[0]].item())
    z_not_met = float(logits[tokenization.not_met_token_ids[0]].item())
    return z_met, z_not_met, int(ids["input_ids"].shape[1])


def _sequence_loglikelihood(
    model: Any, tokenizer: Any, prompt_ids: Any, continuation_ids: tuple[int, ...]
) -> float:
    import torch

    device = model.device
    cont = torch.tensor([list(continuation_ids)], dtype=prompt_ids.dtype, device=device)
    full = torch.cat([prompt_ids, cont], dim=1)
    with torch.no_grad():
        logits = model(input_ids=full).logits[0].float()
    n_prompt = prompt_ids.shape[1]
    total = 0.0
    for offset, token_id in enumerate(continuation_ids):
        step_logits = logits[n_prompt - 1 + offset]
        log_probs = torch.log_softmax(step_logits, dim=-1)
        total += float(log_probs[token_id].item())
    return total


def score_sequence_loglikelihood(
    model: Any, tokenizer: Any, rendered_prompt: str, tokenization: LabelTokenization
) -> tuple[float, float, int]:
    """Return ``(l_met, l_not_met, n_prompt_tokens)`` from continuation log-likelihoods."""
    import torch

    encoded = tokenizer(
        rendered_prompt, return_tensors="pt", add_special_tokens=False
    )
    prompt_ids = encoded["input_ids"].to(model.device)
    with torch.no_grad():
        l_met = _sequence_loglikelihood(
            model, tokenizer, prompt_ids, tokenization.met_token_ids
        )
        l_not_met = _sequence_loglikelihood(
            model, tokenizer, prompt_ids, tokenization.not_met_token_ids
        )
    return l_met, l_not_met, int(prompt_ids.shape[1])


def score_prompt(
    model: Any, tokenizer: Any, rendered_prompt: str, tokenization: LabelTokenization
) -> dict[str, Any]:
    """Score one rendered prompt under the rule implied by ``tokenization``."""
    if tokenization.single_token:
        raw_met, raw_not_met, n_tokens = score_single_token(
            model, tokenizer, rendered_prompt, tokenization
        )
    else:
        raw_met, raw_not_met, n_tokens = score_sequence_loglikelihood(
            model, tokenizer, rendered_prompt, tokenization
        )
    return {
        "raw_score_met": raw_met,
        "raw_score_not_met": raw_not_met,
        "p_met": normalized_probability(raw_met, raw_not_met),
        "scoring_method": tokenization.scoring_method,
        "n_prompt_tokens": n_tokens,
        "rendered_prompt_sha256": sha256_text(rendered_prompt),
    }
