"""Judge prompt construction and the frozen inference order."""

from __future__ import annotations

import pytest

from formalcrrc import prompts
from formalcrrc.config import MODEL_IDS, N_ROWS, PROMPT_TEMPLATE


class TestPromptConstruction:
    def test_message_contains_rubric_and_response_verbatim(self):
        message = prompts.build_user_message("RUBRIC-X", "RESPONSE-Y")
        assert "RUBRIC-X" in message
        assert "RESPONSE-Y" in message

    def test_message_declares_both_answer_labels(self):
        message = prompts.build_user_message("r", "c")
        assert "Return A if the criterion is met." in message
        assert "Return B if the criterion is not met." in message

    def test_prompt_carries_no_chain_of_thought_or_few_shot_cues(self):
        lowered = PROMPT_TEMPLATE.lower()
        for banned in (
            "step by step",
            "step-by-step",
            "think",
            "reason",
            "explain",
            "example",
            "for instance",
            "you are an expert",
            "as a helpful",
        ):
            assert banned not in lowered, banned

    def test_template_has_exactly_two_substitution_slots(self):
        assert PROMPT_TEMPLATE.count("{") == 2
        assert "{rubric_text}" in PROMPT_TEMPLATE
        assert "{candidate_response}" in PROMPT_TEMPLATE

    def test_template_hash_is_stable(self):
        assert prompts.prompt_template_sha256() == prompts.prompt_template_sha256()
        assert len(prompts.prompt_template_sha256()) == 64


class TestInferenceOrder:
    def test_order_key_is_deterministic(self):
        assert prompts.inference_order_key("A-L0-I00|s0") == (
            prompts.inference_order_key("A-L0-I00|s0")
        )

    def test_distinct_prompts_get_distinct_keys(self):
        a = prompts.inference_order_key("A-L0-I00|s0")
        b = prompts.inference_order_key("A-L0-I00|s1")
        assert a != b

    def test_model_run_order_is_a_deterministic_permutation(self):
        order = prompts.model_run_order()
        assert sorted(order) == sorted(MODEL_IDS)
        assert order == prompts.model_run_order()


class TestPromptManifest:
    @pytest.fixture(scope="class")
    @classmethod
    def manifest(cls, formal_dataset):
        return prompts.build_prompt_manifest(formal_dataset)

    def test_one_row_per_rubric_response_pair(self, manifest):
        assert len(manifest) == N_ROWS
        assert manifest["prompt_id"].is_unique

    def test_order_index_is_a_dense_permutation(self, manifest):
        assert manifest["order_index"].tolist() == list(range(N_ROWS))

    def test_run_order_is_not_grouped_by_family(self, manifest):
        """Consecutive prompts must not march through one family at a time."""
        families = manifest["family"].tolist()
        switches = sum(1 for a, b in zip(families, families[1:]) if a != b)
        assert switches > N_ROWS * 0.5

    def test_run_order_is_not_grouped_by_strictness(self, manifest):
        strictness = manifest["strictness_index"].tolist()
        switches = sum(1 for a, b in zip(strictness, strictness[1:]) if a != b)
        assert switches > N_ROWS * 0.5

    def test_run_order_is_not_grouped_by_latent_level(self, manifest):
        latent = manifest["latent_level"].tolist()
        switches = sum(1 for a, b in zip(latent, latent[1:]) if a != b)
        assert switches > N_ROWS * 0.5

    def test_order_is_reproducible(self, formal_dataset, manifest):
        rebuilt = prompts.build_prompt_manifest(formal_dataset)
        assert rebuilt["prompt_id"].tolist() == manifest["prompt_id"].tolist()

    def test_message_hashes_are_correct(self, manifest):
        import hashlib

        for row in manifest.head(50).itertuples(index=False):
            expected = hashlib.sha256(row.user_message.encode("utf-8")).hexdigest()
            assert row.user_message_sha256 == expected

    def test_messages_differ_across_thresholds_of_one_artifact(self, manifest):
        group = manifest[manifest["artifact_id"] == "A-L4-I00"]
        assert group["user_message_sha256"].nunique() == 9
