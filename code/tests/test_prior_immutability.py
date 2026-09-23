"""Days 1 and 2 must survive Day 3 untouched.

Two guarantees: their recorded outputs still hash to what they hashed to, and no
Day-3 code path is capable of writing into a prior experiment's location.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from formalcrrc import dataset as ds
from formalcrrc import day3, day3_config, prompts

MANIFEST = "artifacts/day3/prior_experiments_manifest.json"

#: Day-3 code that must never write into a Day-1 or Day-2 location.
DAY3_SOURCES = (
    "scripts/generate_day3_dataset.py",
    "scripts/freeze_day3_preregistration.py",
    "scripts/verify_day3_ready.py",
    "scripts/run_day3_inference.py",
    "scripts/compute_day3_oracle.py",
    "scripts/analyze_day3.py",
    "scripts/figures_day3.py",
    "scripts/verify_day3_integrity.py",
    "scripts/audit_day3_priors.py",
    "src/formalcrrc/day3.py",
    "src/formalcrrc/day3_bootstrap.py",
    "src/formalcrrc/day3_config.py",
)

PRIOR_WRITE_TARGETS = ("artifacts/day1", "artifacts/day2", "figures/day1", "figures/day2")


class TestPriorManifest:
    def test_manifest_exists(self, repo_root: Path):
        assert (repo_root / MANIFEST).is_file(), "run scripts/audit_day3_priors.py"

    def test_manifest_covers_every_prior_file(self, repo_root: Path):
        manifest = json.loads((repo_root / MANIFEST).read_text(encoding="utf-8"))
        recorded = set(manifest["files"])
        for tree in day3_config.PRIOR_TREES:
            for path in (repo_root / tree).rglob("*"):
                if path.is_file():
                    assert path.relative_to(repo_root).as_posix() in recorded

    def test_manifest_covers_the_prior_documents(self, repo_root: Path):
        manifest = json.loads((repo_root / MANIFEST).read_text(encoding="utf-8"))
        for relative in day3_config.PRIOR_DOCS:
            if (repo_root / relative).is_file():
                assert relative in manifest["files"], relative

    def test_both_experiments_are_separately_recorded(self, repo_root: Path):
        manifest = json.loads((repo_root / MANIFEST).read_text(encoding="utf-8"))
        assert manifest["by_experiment"]["day1"]["n_files"] > 0
        assert manifest["by_experiment"]["day2"]["n_files"] > 0

    def test_nothing_changed(self, repo_root: Path):
        manifest = json.loads((repo_root / MANIFEST).read_text(encoding="utf-8"))
        verification = day3.verify_prior_manifest(manifest, repo_root)
        assert verification["day1"]["changed"] == []
        assert verification["day1"]["missing"] == []
        assert verification["day2"]["changed"] == []
        assert verification["day2"]["missing"] == []
        assert verification["status"] == "PASS"

    def test_combined_hash_refuses_an_empty_list(self):
        with pytest.raises(ValueError):
            day3._combined({})


class TestPriorBehaviourPreserved:
    def test_day1_dataset_still_reproduces(self, formal_dataset):
        assert (
            ds.dataset_content_hash(formal_dataset)
            == day3_config.DAY1_DATASET_CONTENT_SHA256
        )

    def test_day2_partitions_still_reproduce(
        self, repo_root: Path, day2_calibration, day2_test
    ):
        for path, frame in (
            ("artifacts/day2/calibration_manifest.json", day2_calibration),
            ("artifacts/day2/test_manifest.json", day2_test),
        ):
            manifest = json.loads((repo_root / path).read_text(encoding="utf-8"))
            assert manifest["hashes"]["dataset_content_sha256"] == (
                ds.dataset_content_hash(frame)
            )

    def test_prompt_template_unchanged_across_all_three_days(self, repo_root: Path):
        for path in (
            "artifacts/day1/preregistration.json",
            "artifacts/day2/preregistration.json",
        ):
            document = json.loads((repo_root / path).read_text(encoding="utf-8"))
            assert prompts.prompt_template_sha256() == (
                document["prompt"]["template_sha256"]
            )

    def test_day2_pinned_source_files_unchanged(self, repo_root: Path):
        """Day 3 must not have edited any file Day 2 pinned by hash."""
        from formalcrrc import provenance

        manifest = json.loads(
            (repo_root / "artifacts/day2/integrity_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        for relative, recorded in manifest["source_manifest"]["files"].items():
            if recorded is None:
                continue
            candidate = repo_root / relative
            assert candidate.is_file(), relative
            assert provenance.sha256_file(candidate) == recorded, relative

    def test_day1_pinned_sources_changed_only_by_the_documented_extension(
        self, repo_root: Path
    ):
        """One Day-1 source hash is stale, and it was Day 2 that made it so.

        Day 2 extended ``dataset.py`` backward-compatibly and recorded the
        change in its own preregistration. That file must therefore match Day
        2's recorded hash rather than Day 1's, and nothing else may differ.
        """
        from formalcrrc import day2_config, provenance

        day1 = json.loads(
            (repo_root / "artifacts/day1/integrity_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        day2 = json.loads(
            (repo_root / "artifacts/day2/integrity_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        documented = set(day2_config.DAY1_SOURCE_EXTENSIONS)

        differing = []
        for relative, recorded in day1["source_manifest"]["files"].items():
            if recorded is None:
                continue
            candidate = repo_root / relative
            assert candidate.is_file(), relative
            if provenance.sha256_file(candidate) != recorded:
                differing.append(relative)

        assert set(differing) == documented, (
            f"unexpected divergence from the Day-1 source manifest: "
            f"{sorted(set(differing) - documented)}"
        )
        for relative in differing:
            assert provenance.sha256_file(repo_root / relative) == (
                day2["source_manifest"]["files"][relative]
            ), relative

    def test_day3_introduced_no_new_source_extension(self):
        """The documented extension list is Day 2's and Day 3 adds nothing."""
        from formalcrrc import day2_config

        assert set(day2_config.DAY1_SOURCE_EXTENSIONS) == {
            "src/formalcrrc/dataset.py"
        }


class TestDay3CannotWritePriors:
    @pytest.mark.parametrize("relative", DAY3_SOURCES)
    def test_no_prior_write_target_in_day3_code(self, repo_root: Path, relative: str):
        path = repo_root / relative
        if not path.is_file():
            pytest.skip(f"{relative} not present yet")
        for line in path.read_text(encoding="utf-8").splitlines():
            if "write_json" not in line and "to_parquet" not in line:
                continue
            for target in PRIOR_WRITE_TARGETS:
                assert target not in line, f"{relative}: {line.strip()}"

    def test_day3_config_paths_are_all_under_day3(self):
        for name in dir(day3_config):
            if not name.endswith("_PATH"):
                continue
            value = getattr(day3_config, name)
            if isinstance(value, str):
                assert value.startswith(("artifacts/day3", "figures/day3")), (
                    name,
                    value,
                )

    def test_day3_artifact_and_figure_dirs_are_namespaced(self):
        assert day3_config.ARTIFACT_DIR == "artifacts/day3"
        assert day3_config.FIGURE_DIR == "figures/day3"
