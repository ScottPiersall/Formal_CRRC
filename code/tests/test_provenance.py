"""Checksums, source manifests, and the model-provenance gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from formalcrrc import dataset as ds
from formalcrrc import prompts, provenance
from formalcrrc.config import (
    DATASET_MANIFEST_PATH,
    DATASET_PATH,
    MODEL_IDS,
    PREREG_JSON_PATH,
    PREREG_SHA_PATH,
)


def _good_record() -> dict:
    return {
        "model_id": "Qwen/Qwen2.5-14B-Instruct",
        "revision": "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
        "tokenizer_revision": "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
        "dtype": "bfloat16",
        "chat_template_sha256": "0" * 64,
        "label_tokenization": {
            "met_token_ids": [32],
            "not_met_token_ids": [33],
            "scoring_method": "single_token_next_logit",
        },
        "transformers_version": "5.16.1",
        "torch_version": "2.13.0+cu130",
    }


class TestHashing:
    def test_text_hash_is_stable_and_sensitive(self):
        assert provenance.sha256_text("abc") == provenance.sha256_text("abc")
        assert provenance.sha256_text("abc") != provenance.sha256_text("abd")

    def test_canonical_json_is_key_order_independent(self):
        a = {"x": 1, "y": {"b": 2, "a": 3}}
        b = {"y": {"a": 3, "b": 2}, "x": 1}
        assert provenance.sha256_json(a) == provenance.sha256_json(b)

    def test_canonical_json_detects_a_value_change(self):
        assert provenance.sha256_json({"x": 1}) != provenance.sha256_json({"x": 2})

    def test_file_hash_matches_text_hash(self, tmp_path: Path):
        target = tmp_path / "f.txt"
        target.write_text("hello", encoding="utf-8")
        assert provenance.sha256_file(target) == provenance.sha256_text("hello")


class TestChecksumFiles:
    def test_written_checksum_verifies(self, tmp_path: Path):
        target = tmp_path / "doc.json"
        provenance.write_json(target, {"a": 1})
        sha_path = tmp_path / "doc.sha256"
        digest = provenance.write_checksum_file(target, sha_path)
        assert len(digest) == 64
        assert provenance.verify_checksum_file(target, sha_path) is True

    def test_tampering_breaks_verification(self, tmp_path: Path):
        target = tmp_path / "doc.json"
        provenance.write_json(target, {"a": 1})
        sha_path = tmp_path / "doc.sha256"
        provenance.write_checksum_file(target, sha_path)
        provenance.write_json(target, {"a": 2})
        assert provenance.verify_checksum_file(target, sha_path) is False

    def test_checksum_file_names_its_target(self, tmp_path: Path):
        target = tmp_path / "preregistration.json"
        provenance.write_json(target, {"a": 1})
        sha_path = tmp_path / "preregistration.sha256"
        provenance.write_checksum_file(target, sha_path)
        assert "preregistration.json" in sha_path.read_text(encoding="utf-8")


class TestSourceManifest:
    def test_hashes_present_files_and_marks_absent_ones(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
        manifest = provenance.source_manifest(["a.py", "missing.py"], root=tmp_path)
        assert manifest["files"]["a.py"] is not None
        assert manifest["files"]["missing.py"] is None
        assert len(manifest["tree_sha256"]) == 64

    def test_tree_hash_changes_when_a_file_changes(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
        before = provenance.source_manifest(["a.py"], root=tmp_path)["tree_sha256"]
        (tmp_path / "a.py").write_text("x = 2\n", encoding="utf-8")
        after = provenance.source_manifest(["a.py"], root=tmp_path)["tree_sha256"]
        assert before != after

    def test_analysis_code_paths_all_exist(self, repo_root: Path):
        from formalcrrc.config import ANALYSIS_CODE_PATHS

        missing = [p for p in ANALYSIS_CODE_PATHS if not (repo_root / p).is_file()]
        assert missing == []


class TestModelProvenanceGate:
    def test_a_complete_record_passes(self):
        assert provenance.validate_model_provenance(_good_record()) == []

    @pytest.mark.parametrize(
        "field", list(provenance.REQUIRED_MODEL_PROVENANCE_FIELDS)
    )
    def test_every_required_field_is_enforced(self, field):
        record = _good_record()
        record.pop(field)
        problems = provenance.validate_model_provenance(record)
        assert any(field in p for p in problems)

    @pytest.mark.parametrize("revision", ["main", "master", "HEAD", "abc"])
    def test_unresolved_revision_is_rejected(self, revision):
        record = _good_record()
        record["revision"] = revision
        problems = provenance.validate_model_provenance(record)
        assert any("revision" in p for p in problems)

    def test_missing_scoring_method_is_rejected(self):
        record = _good_record()
        record["label_tokenization"] = {
            "met_token_ids": [32],
            "not_met_token_ids": [33],
        }
        problems = provenance.validate_model_provenance(record)
        assert any("scoring_method" in p for p in problems)

    def test_missing_label_token_ids_are_rejected(self):
        record = _good_record()
        record["label_tokenization"]["met_token_ids"] = []
        problems = provenance.validate_model_provenance(record)
        assert any("met_token_ids" in p for p in problems)


class TestEnvironmentSnapshot:
    def test_snapshot_records_the_interpreter_and_libraries(self):
        snapshot = provenance.environment_snapshot(include_gpu=False)
        assert snapshot["python_version"]
        assert "numpy" in snapshot["packages"]
        assert "gpu" not in snapshot

    def test_gpu_snapshot_reports_a_reason_when_absent(self):
        gpu = provenance.gpu_snapshot()
        assert "available" in gpu
        if not gpu["available"]:
            assert gpu["reason"]

    def test_snapshot_contains_no_secret_looking_keys(self):
        snapshot = provenance.environment_snapshot(include_gpu=True)
        # "tokenizers" is a recorded package name, not a credential.
        snapshot["packages"].pop("tokenizers", None)
        text = json.dumps(snapshot).lower()
        for banned in (
            "token",
            "password",
            "api_key",
            "apikey",
            "secret",
            "credential",
            "bearer",
            "authorization",
            "private key",
            "ssh-rsa",
            "begin openssh",
        ):
            assert banned not in text, banned


class TestFrozenArtifacts:
    """Checks against the on-disk artifacts, skipped until they are generated."""

    def test_dataset_manifest_matches_the_rebuilt_dataset(self, repo_root: Path):
        manifest_path = repo_root / DATASET_MANIFEST_PATH
        if not manifest_path.is_file():
            pytest.skip("dataset not generated yet")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rebuilt = ds.dataset_content_hash(ds.build_dataset())
        assert manifest["hashes"]["dataset_content_sha256"] == rebuilt

    def test_dataset_file_hash_matches_the_manifest(self, repo_root: Path):
        manifest_path = repo_root / DATASET_MANIFEST_PATH
        dataset_path = repo_root / DATASET_PATH
        if not (manifest_path.is_file() and dataset_path.is_file()):
            pytest.skip("dataset not generated yet")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["hashes"]["dataset_file_sha256"] == provenance.sha256_file(
            dataset_path
        )

    def test_prompt_template_hash_matches_the_manifest(self, repo_root: Path):
        manifest_path = repo_root / DATASET_MANIFEST_PATH
        if not manifest_path.is_file():
            pytest.skip("dataset not generated yet")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["hashes"]["prompt_template_sha256"] == (
            prompts.prompt_template_sha256()
        )

    def test_preregistration_checksum_verifies(self, repo_root: Path):
        prereg = repo_root / PREREG_JSON_PATH
        sha_path = repo_root / PREREG_SHA_PATH
        if not (prereg.is_file() and sha_path.is_file()):
            pytest.skip("preregistration not frozen yet")
        assert provenance.verify_checksum_file(prereg, sha_path) is True

    def test_preregistration_pins_the_designed_model_panel(self, repo_root: Path):
        prereg = repo_root / PREREG_JSON_PATH
        if not prereg.is_file():
            pytest.skip("preregistration not frozen yet")
        document = json.loads(prereg.read_text(encoding="utf-8"))
        assert document["judge_models"]["model_ids"] == list(MODEL_IDS)

    def test_preregistration_pins_the_dataset_hash(self, repo_root: Path):
        prereg = repo_root / PREREG_JSON_PATH
        manifest_path = repo_root / DATASET_MANIFEST_PATH
        if not (prereg.is_file() and manifest_path.is_file()):
            pytest.skip("preregistration not frozen yet")
        document = json.loads(prereg.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert document["dataset"]["content_sha256"] == (
            manifest["hashes"]["dataset_content_sha256"]
        )
