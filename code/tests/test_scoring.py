"""A/B probability extraction against manually constructed logits.

No model is loaded here. The tokenizer is a stub whose behaviour is chosen to
exercise both the primary single-token rule and the preregistered fallback.
"""

from __future__ import annotations

import math

import pytest

from formalcrrc import scoring
from formalcrrc.config import (
    LABEL_MET,
    LABEL_NOT_MET,
    SCORING_FALLBACK,
    SCORING_PRIMARY,
)


PROMPT = "<user>probe</user><assistant>"
PROMPT_IDS = [7, 8, 9]


class StubTokenizer:
    """Minimal tokenizer stand-in.

    ``naive`` is what encoding a bare label returns; ``continuation`` is what
    really follows the prompt. Setting them differently reproduces the
    SentencePiece word-boundary case.
    """

    def __init__(
        self,
        continuation: dict[str, list[int]],
        naive: dict[str, list[int]] | None = None,
        chat_template: str = "T",
        prompt_ids: list[int] | None = None,
    ):
        self._continuation = continuation
        self._naive = naive if naive is not None else continuation
        self.chat_template = chat_template
        self._prompt_ids = list(prompt_ids if prompt_ids is not None else PROMPT_IDS)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        if text in self._naive:
            return list(self._naive[text])
        for label, ids in self._continuation.items():
            if text.endswith(label) and text[: -len(label)] == PROMPT:
                return self._prompt_ids + list(ids)
        if text == PROMPT:
            return list(self._prompt_ids)
        raise KeyError(text)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        body = messages[0]["content"]
        suffix = "<assistant>" if add_generation_prompt else ""
        return f"<user>{body}</user>{suffix}"


class TestNormalizedProbability:
    def test_equal_scores_give_one_half(self):
        assert scoring.normalized_probability(0.0, 0.0) == pytest.approx(0.5)
        assert scoring.normalized_probability(7.5, 7.5) == pytest.approx(0.5)

    def test_matches_the_softmax_definition(self):
        z_a, z_b = 2.0, 0.5
        expected = math.exp(z_a) / (math.exp(z_a) + math.exp(z_b))
        assert scoring.normalized_probability(z_a, z_b) == pytest.approx(expected)

    def test_hand_computed_case(self):
        # exp(1) / (exp(1) + exp(0)) = 2.71828.. / 3.71828.. = 0.7310585786
        assert scoring.normalized_probability(1.0, 0.0) == pytest.approx(
            0.7310585786, abs=1e-9
        )

    def test_is_invariant_to_a_shared_shift(self):
        base = scoring.normalized_probability(3.0, -1.0)
        shifted = scoring.normalized_probability(103.0, 99.0)
        assert base == pytest.approx(shifted)

    def test_ordering_is_preserved(self):
        assert scoring.normalized_probability(5.0, 0.0) > 0.5
        assert scoring.normalized_probability(0.0, 5.0) < 0.5

    def test_extreme_values_do_not_overflow(self):
        assert scoring.normalized_probability(1000.0, -1000.0) == pytest.approx(1.0)
        assert scoring.normalized_probability(-1000.0, 1000.0) == pytest.approx(0.0)

    def test_non_finite_scores_are_rejected(self):
        with pytest.raises(ValueError):
            scoring.normalized_probability(float("nan"), 0.0)
        with pytest.raises(ValueError):
            scoring.normalized_probability(0.0, float("inf"))

    def test_logits_and_loglikelihoods_use_identical_arithmetic(self):
        """The fallback is a generalisation, not a different measurement."""
        z_a, z_b = 1.25, -0.75
        log_z = math.log(math.exp(z_a) + math.exp(z_b))
        assert scoring.normalized_probability(z_a, z_b) == pytest.approx(
            scoring.normalized_probability(z_a - log_z, z_b - log_z)
        )


class TestLabelTokenization:
    def test_single_token_labels_select_the_primary_rule(self):
        tokenizer = StubTokenizer({LABEL_MET: [32], LABEL_NOT_MET: [33]})
        resolved = scoring.resolve_label_tokenization(tokenizer, PROMPT)
        assert resolved.single_token is True
        assert resolved.scoring_method == SCORING_PRIMARY
        assert resolved.met_token_ids == (32,)
        assert resolved.not_met_token_ids == (33,)

    def test_multi_token_label_selects_the_fallback(self):
        tokenizer = StubTokenizer({LABEL_MET: [1, 2], LABEL_NOT_MET: [33]})
        resolved = scoring.resolve_label_tokenization(tokenizer, PROMPT)
        assert resolved.single_token is False
        assert resolved.scoring_method == SCORING_FALLBACK

    def test_either_label_being_multi_token_triggers_the_fallback(self):
        tokenizer = StubTokenizer({LABEL_MET: [32], LABEL_NOT_MET: [5, 6, 7]})
        resolved = scoring.resolve_label_tokenization(tokenizer, PROMPT)
        assert resolved.single_token is False

    def test_record_round_trips_to_a_dict(self):
        tokenizer = StubTokenizer({LABEL_MET: [32], LABEL_NOT_MET: [33]})
        record = scoring.resolve_label_tokenization(tokenizer, PROMPT).to_dict()
        assert record["label_met"] == "A"
        assert record["label_not_met"] == "B"
        assert record["scoring_method"] == SCORING_PRIMARY
        assert record["met_token_ids"] == [32]
        assert record["differs_from_naive_encoding"] is False

    def test_everything_is_encoded_without_special_tokens(self):
        calls: list[bool] = []

        class Recorder(StubTokenizer):
            def encode(self, text, add_special_tokens=True):
                calls.append(add_special_tokens)
                return super().encode(text, add_special_tokens)

        scoring.resolve_label_tokenization(
            Recorder({LABEL_MET: [32], LABEL_NOT_MET: [33]}), PROMPT
        )
        assert calls and all(flag is False for flag in calls)


class TestContinuationResolution:
    """The scored token must be what the model emits, not what encode() returns.

    Mistral-7B-Instruct-v0.3 encodes the bare string "A" as the
    word-boundary-marked token "_A", while the token that actually continues its
    prompt is a different id. Indexing the former would read the wrong logit.
    """

    def test_word_boundary_marker_case_is_detected(self):
        tokenizer = StubTokenizer(
            continuation={LABEL_MET: [29509], LABEL_NOT_MET: [29528]},
            naive={LABEL_MET: [1098], LABEL_NOT_MET: [1133]},
        )
        resolved = scoring.resolve_label_tokenization(tokenizer, PROMPT)
        assert resolved.met_token_ids == (29509,)
        assert resolved.not_met_token_ids == (29528,)
        assert resolved.naive_met_token_ids == (1098,)
        assert resolved.differs_from_naive is True
        assert resolved.single_token is True

    def test_agreement_is_reported_when_the_two_definitions_coincide(self):
        tokenizer = StubTokenizer({LABEL_MET: [32], LABEL_NOT_MET: [33]})
        resolved = scoring.resolve_label_tokenization(tokenizer, PROMPT)
        assert resolved.differs_from_naive is False

    def test_continuation_ids_isolate_the_appended_label(self):
        tokenizer = StubTokenizer({LABEL_MET: [42], LABEL_NOT_MET: [43]})
        assert scoring.label_continuation_ids(tokenizer, PROMPT, LABEL_MET) == (42,)

    def test_a_prompt_whose_tokenisation_shifts_is_rejected(self):
        class Shifting(StubTokenizer):
            def encode(self, text, add_special_tokens=True):
                if text == PROMPT:
                    return [1, 2, 3]
                return [9, 9, 9, 42]

        with pytest.raises(ValueError):
            scoring.label_continuation_ids(Shifting({}), PROMPT, LABEL_MET)

    def test_an_empty_continuation_is_rejected(self):
        class Empty(StubTokenizer):
            def encode(self, text, add_special_tokens=True):
                return [1, 2, 3]

        with pytest.raises(ValueError):
            scoring.label_continuation_ids(Empty({}), PROMPT, LABEL_MET)


class TestChatTemplateRendering:
    def test_single_user_message_with_generation_prompt(self):
        tokenizer = StubTokenizer({LABEL_MET: [32], LABEL_NOT_MET: [33]})
        rendered = scoring.render_chat_prompt(tokenizer, "hello")
        assert rendered == "<user>hello</user><assistant>"

    def test_rendered_hash_changes_with_the_message(self):
        tokenizer = StubTokenizer({LABEL_MET: [32], LABEL_NOT_MET: [33]})
        a = scoring.sha256_text(scoring.render_chat_prompt(tokenizer, "one"))
        b = scoring.sha256_text(scoring.render_chat_prompt(tokenizer, "two"))
        assert a != b

    def test_hash_is_stable_for_identical_input(self):
        assert scoring.sha256_text("x") == scoring.sha256_text("x")
