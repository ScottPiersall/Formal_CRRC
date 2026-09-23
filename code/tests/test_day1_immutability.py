"""Day 1 must survive Day 2 untouched.

Two separate guarantees are tested: that the recorded Day-1 outputs still hash
to what they hashed to, and that no Day-2 code path is even capable of writing
into a Day-1 output location.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from formalcrrc import dataset as ds
from formalcrrc import day1_audit, day2_config, prompts

BASELINE = "artifacts/day2/day1_baseline_manifest.json"

#: Day-2 code that must never write into a Day-1 location.
DAY2_SCRIPTS = (
    "scripts/generate_day2_datasets.py",
    "scripts/freeze_day2_preregistration.py",
    "scripts/verify_day2_ready.py",
    "scripts/run_day2_inference.py",
    "scripts/fit_day2_bias.py",
    "scripts/analyze_day2.py",
    "scripts/figures_day2.py",
    "scripts/verify_day2_integrity.py",
    "src/formalcrrc/day2.py",
    "src/formalcrrc/day2_bootstrap.py",
    "src/formalcrrc/day2_config.py",
)

#: The one permitted addition under artifacts/day1 -- additive, overwrites nothing.
PERMITTED_DAY1_WRITE = "artifacts/day1/integrity_addendum.json"


class TestBaselineManifest:
    def test_baseline_exists(self, repo_root: Path):
        assert (repo_root / BASELINE).is_file(), "run scripts/audit_day1_for_day2.py"

    def test_baseline_covers_every_day1_artifact(self, repo_root: Path):
        baseline = json.loads((repo_root / BASELINE).read_text(encoding="utf-8"))
        recorded = set(baseline["files"])
        for tree in day1_audit.DAY1_IMMUTABLE_TREES:
            for path in (repo_root / tree).rglob("*"):
                if not path.is_file():
                    continue
                relative = path.relative_to(repo_root).as_posix()
                if relative == PERMITTED_DAY1_WRITE:
                    continue
                assert relative in recorded, f"{relative} is not fingerprinted"

    def test_baseline_covers_the_day1_documents(self, repo_root: Path):
        baseline = json.loads((repo_root / BASELINE).read_text(encoding="utf-8"))
        for relative in day1_audit.DAY1_IMMUTABLE_DOCS:
            assert relative in baseline["files"], relative

    def test_day1_files_are_unchanged(self, repo_root: Path):
        baseline = json.loads((repo_root / BASELINE).read_text(encoding="utf-8"))
        verification = day1_audit.verify_day1_baseline(baseline, repo_root)
        assert verification["changed"] == [], verification["changed"]
        assert verification["missing"] == [], verification["missing"]
        assert verification["status"] == "PASS"

    def test_only_the_addendum_was_added(self, repo_root: Path):
        baseline = json.loads((repo_root / BASELINE).read_text(encoding="utf-8"))
        added = day1_audit.verify_day1_baseline(baseline, repo_root)["added"]
        assert set(added) <= {PERMITTED_DAY1_WRITE}, added


class TestDay1BehaviourPreserved:
    """Extending the shared modules must not move a single Day-1 byte."""

    def test_day1_dataset_still_reproduces_its_content_hash(self, formal_dataset):
        assert (
            ds.dataset_content_hash(formal_dataset)
            == day2_config.DAY1_DATASET_CONTENT_SHA256
        )

    def test_default_build_matches_the_frozen_manifest(self, repo_root: Path):
        manifest = json.loads(
            (repo_root / "artifacts/day1/dataset_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert (
            ds.dataset_content_hash(ds.build_dataset())
            == manifest["hashes"]["dataset_content_sha256"]
        )

    def test_default_artifact_ids_carry_no_partition_prefix(self, formal_dataset):
        assert formal_dataset["artifact_id"].str.match(r"^[ABC]-L\d-I\d\d$").all()

    def test_day1_seed_column_unchanged(self, formal_dataset):
        assert set(formal_dataset["seed"]) == {42}

    def test_prompt_template_hash_unchanged(self, repo_root: Path):
        prereg = json.loads(
            (repo_root / "artifacts/day1/preregistration.json").read_text(
                encoding="utf-8"
            )
        )
        assert prompts.prompt_template_sha256() == prereg["prompt"]["template_sha256"]

    def test_day1_default_generation_ignores_the_uniqueness_rule(self):
        """Day 1 contains a duplicate family-C pair; the default must keep it."""
        frame = ds.build_dataset()
        subset = frame[frame["family"] == "numeric_tolerance"].drop_duplicates(
            "artifact_id"
        )
        pairs = list(
            zip(subset["target_value"].astype(int), subset["reported_value"].astype(int))
        )
        assert len(pairs) - len(set(pairs)) == 1


class TestDay2CannotWriteDay1:
    @pytest.mark.parametrize("relative", DAY2_SCRIPTS)
    def test_no_day1_output_path_appears_in_day2_code(
        self, repo_root: Path, relative: str
    ):
        path = repo_root / relative
        if not path.is_file():
            pytest.skip(f"{relative} not present yet")
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"(artifacts/day1|figures/day1)[\w./]*", text):
            target = match.group(0)
            assert target in (
                "artifacts/day1",
                "figures/day1",
                PERMITTED_DAY1_WRITE,
            ) or target.startswith("artifacts/day1/"), target

    @pytest.mark.parametrize("relative", DAY2_SCRIPTS)
    def test_day2_code_writes_only_to_day2_locations(
        self, repo_root: Path, relative: str
    ):
        """Any write target named in Day-2 code must live under a Day-2 path."""
        path = repo_root / relative
        if not path.is_file():
            pytest.skip(f"{relative} not present yet")
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if "write_json" not in line and "to_parquet" not in line:
                continue
            if "day1" in line:
                assert PERMITTED_DAY1_WRITE in line or "day1_baseline" in line, line

    def test_day2_config_paths_are_all_under_day2(self):
        for name in dir(day2_config):
            if not name.endswith("_PATH"):
                continue
            value = getattr(day2_config, name)
            if not isinstance(value, str):
                continue
            assert value.startswith(("artifacts/day2", "figures/day2")), (name, value)
