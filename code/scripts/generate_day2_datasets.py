#!/usr/bin/env python
"""Generate and freeze the Day-2 CALIBRATION and TEST partitions (Phase C).

Two new partitions of 324 artifacts x 9 thresholds each, drawn from the same
data-generating process as Day 1 with master seeds 43 and 44. Neither reuses a
Day-1 artifact.

Family C draws an integer target from a bounded range, so independent draws can
coincide. Generation therefore carries a forbidden set of (target, reported)
pairs -- Day 1's, then Day 1's plus calibration's -- so that no evaluated item is
shared between Day 1, CALIBRATION and TEST. Families A and B need no such rule:
their marker pools make a collision astronomically unlikely, and the observed
overlap is zero.

Writes the two datasets, their prompt manifests with frozen inference orders,
and a manifest per partition. Runs before any Day-2 model is loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from formalcrrc import dataset as ds  # noqa: E402
from formalcrrc import day2_config as d2c  # noqa: E402
from formalcrrc import predicates, prompts, provenance  # noqa: E402
from formalcrrc.config import (  # noqa: E402
    FAMILIES,
    FAMILY_NUMERIC_TOLERANCE,
    LIST_SIZE,
    MARKERS_PER_RESPONSE,
    N_ARTIFACTS,
    N_INSTANCES,
    N_LATENT_LEVELS,
    N_ROWS,
    N_STRICTNESS,
)


def family_c_pairs(frame: pd.DataFrame) -> frozenset[tuple[int, int]]:
    """The (target, reported) pairs used by a dataset's family-C artifacts."""
    subset = frame[frame["family"] == FAMILY_NUMERIC_TOLERANCE].drop_duplicates(
        "artifact_id"
    )
    return frozenset(
        zip(
            subset["target_value"].astype(int),
            subset["reported_value"].astype(int),
        )
    )


def assert_partition_structure(frame: pd.DataFrame, partition: str) -> None:
    """Every structural invariant a Day-2 partition must satisfy on its own."""
    label = f"[{partition}]"
    assert len(frame) == N_ROWS, f"{label} expected {N_ROWS} rows, got {len(frame)}"
    assert frame["artifact_id"].nunique() == N_ARTIFACTS, f"{label} artifact count"
    assert frame["prompt_id"].is_unique, f"{label} prompt ids not unique"
    assert set(frame["seed"]) == {d2c.SEEDS[partition]}, f"{label} seed column"
    assert frame["artifact_id"].str.startswith(f"{partition}-").all(), (
        f"{label} artifact ids are not namespaced"
    )

    for artifact_id, group in frame.groupby("artifact_id", sort=False):
        levels = sorted(group["strictness_index"].tolist())
        assert levels == list(range(N_STRICTNESS)), f"{label} {artifact_id} levels"
        assert group["candidate_response"].nunique() == 1, (
            f"{label} {artifact_id}: candidate response is not byte-identical"
        )
        family = group["family"].iloc[0]
        normalized = {
            predicates.normalize_rubric(family, text) for text in group["rubric_text"]
        }
        assert len(normalized) == 1, (
            f"{label} {artifact_id}: rubrics differ outside the threshold field"
        )
        assert group["rubric_sha256"].nunique() == N_STRICTNESS, (
            f"{label} {artifact_id}: rubric text does not vary with the threshold"
        )
        ordered = group.sort_values("strictness_index")
        truth = ordered["formal_truth"].tolist()
        assert truth == sorted(truth, reverse=True), (
            f"{label} {artifact_id}: formal labels are not monotone"
        )
        latent = int(group["latent_level"].iloc[0])
        recomputed = [
            predicates.formal_truth(family, latent, s) for s in range(N_STRICTNESS)
        ]
        assert truth == recomputed, f"{label} {artifact_id}: truth not reproducible"
        zeros = [s for s, y in enumerate(truth) if y == 0]
        expected = zeros[0] if zeros else N_STRICTNESS
        assert set(ordered["true_first_fail_index"]) == {expected}, (
            f"{label} {artifact_id}: first-fail index"
        )

    cells = Counter(zip(frame["family"], frame["latent_level"]))
    for family in FAMILIES:
        for latent in range(N_LATENT_LEVELS):
            assert cells[(family, latent)] == N_INSTANCES * N_STRICTNESS, (
                f"{label} {family}/L{latent}: wrong row count"
            )

    markers = frame[frame["family"] != FAMILY_NUMERIC_TOLERANCE]
    for payload in markers["response_markers_json"].unique():
        assert payload.count("MK-") == MARKERS_PER_RESPONSE, f"{label} marker count"
    for payload in markers["list_items_json"].unique():
        assert payload.count("MK-") == LIST_SIZE, f"{label} item list size"


def rendered_prompts(frame: pd.DataFrame) -> set[str]:
    return {
        prompts.build_user_message(rubric, response)
        for rubric, response in zip(
            frame["rubric_text"], frame["candidate_response"], strict=True
        )
    }


def cross_partition_report(
    calibration: pd.DataFrame, test: pd.DataFrame, day1: pd.DataFrame
) -> dict:
    """Overlap between the three partitions, asserted where it must be zero."""

    def artifacts(frame: pd.DataFrame) -> set[str]:
        return set(frame["artifact_id"])

    def candidates(frame: pd.DataFrame, family: str | None = None) -> set[str]:
        subset = frame if family is None else frame[frame["family"] == family]
        return set(subset.drop_duplicates("artifact_id")["candidate_sha256"])

    pairs = {
        "day1": family_c_pairs(day1),
        "cal": family_c_pairs(calibration),
        "test": family_c_pairs(test),
    }
    prompt_sets = {
        "day1": rendered_prompts(day1),
        "cal": rendered_prompts(calibration),
        "test": rendered_prompts(test),
    }

    report = {
        "artifact_id_overlap": {
            "cal_vs_test": len(artifacts(calibration) & artifacts(test)),
            "cal_vs_day1": len(artifacts(calibration) & artifacts(day1)),
            "test_vs_day1": len(artifacts(test) & artifacts(day1)),
        },
        "rendered_prompt_overlap": {
            "cal_vs_test": len(prompt_sets["cal"] & prompt_sets["test"]),
            "cal_vs_day1": len(prompt_sets["cal"] & prompt_sets["day1"]),
            "test_vs_day1": len(prompt_sets["test"] & prompt_sets["day1"]),
        },
        "family_c_target_pair_overlap": {
            "cal_vs_test": len(pairs["cal"] & pairs["test"]),
            "cal_vs_day1": len(pairs["cal"] & pairs["day1"]),
            "test_vs_day1": len(pairs["test"] & pairs["day1"]),
        },
        "candidate_response_overlap_by_family": {
            family: {
                "cal_vs_test": len(
                    candidates(calibration, family) & candidates(test, family)
                ),
                "cal_vs_day1": len(
                    candidates(calibration, family) & candidates(day1, family)
                ),
                "test_vs_day1": len(
                    candidates(test, family) & candidates(day1, family)
                ),
            }
            for family in FAMILIES
        },
    }

    # Hard requirements.
    for key, value in report["artifact_id_overlap"].items():
        assert value == 0, f"artifact_id overlap {key} = {value}"
    for key, value in report["rendered_prompt_overlap"].items():
        assert value == 0, f"rendered prompt overlap {key} = {value}"
    for key, value in report["family_c_target_pair_overlap"].items():
        assert value == 0, f"family-C target pair overlap {key} = {value}"
    for family in (f for f in FAMILIES if f != FAMILY_NUMERIC_TOLERANCE):
        for key, value in report["candidate_response_overlap_by_family"][
            family
        ].items():
            assert value == 0, f"{family} candidate overlap {key} = {value}"

    # Family-C candidate responses depend only on the reported value, which is a
    # bounded integer, so coincidences are expected. They are not reuse: the
    # (target, reported) pairs and the rendered prompts are disjoint.
    report["family_c_candidate_coincidence_note"] = (
        "A family-C candidate response is determined by the reported value "
        "alone, an integer in a bounded range, so byte-identical candidate "
        "responses across partitions are an expected property of the shared "
        "data-generating process. They are not reused items: the (target, "
        "reported) pairs are disjoint by construction and no rendered prompt is "
        "shared."
    )
    return report


def leakage_checks(frame: pd.DataFrame, partition: str) -> dict:
    """Confirm the partition label never reaches a model-facing prompt."""
    offenders_id: list[str] = []
    offenders_word: list[str] = []
    for row in frame.itertuples(index=False):
        message = prompts.build_user_message(row.rubric_text, row.candidate_response)
        if row.artifact_id in message:
            offenders_id.append(row.prompt_id)
        lowered = message.lower()
        if "calibration" in lowered or "partition" in lowered or "held-out" in lowered:
            offenders_word.append(row.prompt_id)
    assert not offenders_id, f"artifact id leaked into {len(offenders_id)} prompts"
    assert not offenders_word, f"partition wording leaked into {len(offenders_word)}"
    return {
        "artifact_id_in_prompt": len(offenders_id),
        "partition_wording_in_prompt": len(offenders_word),
        "note": (
            "Prompts contain only the rubric and the candidate response. The "
            "partition tag lives in artifact_id and prompt_id, which are never "
            "rendered into a model-facing message."
        ),
    }


def build_manifest(
    frame: pd.DataFrame,
    prompt_frame: pd.DataFrame,
    partition: str,
    dataset_path: Path,
    prompt_path: Path,
    root: Path,
    cross: dict | None,
    leakage: dict,
) -> dict:
    return {
        "created_at": provenance.utc_now(),
        "experiment": "FormalCRRC Day 2",
        "partition": partition,
        "role": (
            "intercept fitting only"
            if partition == d2c.PARTITION_CALIBRATION
            else "held-out evaluation only"
        ),
        "seed": d2c.SEEDS[partition],
        "dimensions": {
            "families": list(FAMILIES),
            "n_latent_levels": N_LATENT_LEVELS,
            "n_instances_per_cell": N_INSTANCES,
            "n_artifacts": int(frame["artifact_id"].nunique()),
            "n_strictness_levels": N_STRICTNESS,
            "n_rows": int(len(frame)),
        },
        "hashes": {
            "dataset_content_sha256": ds.dataset_content_hash(frame),
            "dataset_file_sha256": provenance.sha256_file(dataset_path),
            "prompt_manifest_file_sha256": provenance.sha256_file(prompt_path),
            "prompt_template_sha256": prompts.prompt_template_sha256(),
            "prompt_set_sha256": hashlib.sha256(
                "\n".join(sorted(prompt_frame["user_message_sha256"])).encode("utf-8")
            ).hexdigest(),
        },
        "inference_order": {
            "salt": d2c.ORDER_SALTS[partition],
            "rule": "sort by sha256(salt|prompt_id), frozen before inference",
        },
        "leakage_checks": leakage,
        "cross_partition": cross,
        "source": provenance.source_manifest(
            [
                "src/formalcrrc/config.py",
                "src/formalcrrc/predicates.py",
                "src/formalcrrc/dataset.py",
                "src/formalcrrc/prompts.py",
                "src/formalcrrc/day2_config.py",
                "scripts/generate_day2_datasets.py",
            ],
            root=root,
        ),
        "git": provenance.git_state(root),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(provenance.repo_root()))
    args = parser.parse_args()
    root = Path(args.root)

    print("rebuilding Day-1 dataset for the forbidden-pair set ...", flush=True)
    day1 = ds.build_dataset()
    day1_hash = ds.dataset_content_hash(day1)
    if day1_hash != d2c.DAY1_DATASET_CONTENT_SHA256:
        print(
            "ABORT: Day-1 dataset no longer reproduces its frozen content hash\n"
            f"  expected {d2c.DAY1_DATASET_CONTENT_SHA256}\n"
            f"  got      {day1_hash}",
            flush=True,
        )
        return 2
    print(f"Day-1 content hash reproduced: {day1_hash}", flush=True)

    forbidden = family_c_pairs(day1)
    frames: dict[str, pd.DataFrame] = {}
    prompt_frames: dict[str, pd.DataFrame] = {}

    for partition in d2c.GENERATION_ORDER:
        print(f"building {partition} (seed {d2c.SEEDS[partition]}) ...", flush=True)
        frame = ds.build_dataset(
            seed=d2c.SEEDS[partition],
            partition=partition,
            forbidden_pairs=forbidden,
            enforce_unique_targets=True,
        )
        assert_partition_structure(frame, partition)
        frames[partition] = frame
        forbidden = forbidden | family_c_pairs(frame)

        prompt_frame = prompts.build_prompt_manifest(frame)
        # Re-derive the order under this partition's own salt.
        prompt_frame = prompt_frame.drop(columns=["order_index", "order_key"])
        prompt_frame["order_key"] = [
            provenance.sha256_text(f"{d2c.ORDER_SALTS[partition]}|{pid}")
            for pid in prompt_frame["prompt_id"]
        ]
        prompt_frame = prompt_frame.sort_values("order_key").reset_index(drop=True)
        prompt_frame.insert(0, "order_index", range(len(prompt_frame)))
        prompt_frames[partition] = prompt_frame

    calibration = frames[d2c.PARTITION_CALIBRATION]
    test = frames[d2c.PARTITION_TEST]
    cross = cross_partition_report(calibration, test, day1)

    for partition, frame in frames.items():
        leakage = leakage_checks(frame, partition)
        dataset_path = root / d2c.DATASET_PATHS[partition]
        prompt_path = root / d2c.PROMPT_PATHS[partition]
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(dataset_path, index=False)
        prompt_frames[partition].to_parquet(prompt_path, index=False)

        manifest = build_manifest(
            frame,
            prompt_frames[partition],
            partition,
            dataset_path,
            prompt_path,
            root,
            cross if partition == d2c.PARTITION_CALIBRATION else None,
            leakage,
        )
        provenance.write_json(root / d2c.MANIFEST_PATHS[partition], manifest)
        print(
            f"{partition}: {manifest['dimensions']['n_artifacts']} artifacts, "
            f"{manifest['dimensions']['n_rows']} rows, "
            f"content {manifest['hashes']['dataset_content_sha256'][:16]}",
            flush=True,
        )

    print("", flush=True)
    print("cross-partition overlap (all must be zero):", flush=True)
    for group in (
        "artifact_id_overlap",
        "rendered_prompt_overlap",
        "family_c_target_pair_overlap",
    ):
        print(f"  {group}: {cross[group]}", flush=True)
    print(
        f"  family-C candidate coincidences: "
        f"{cross['candidate_response_overlap_by_family'][FAMILY_NUMERIC_TOLERANCE]}",
        flush=True,
    )
    print("", flush=True)
    for partition in d2c.GENERATION_ORDER:
        print(f"wrote {d2c.DATASET_PATHS[partition]}", flush=True)
        print(f"wrote {d2c.PROMPT_PATHS[partition]}", flush=True)
        print(f"wrote {d2c.MANIFEST_PATHS[partition]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
