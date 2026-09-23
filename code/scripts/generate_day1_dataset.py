#!/usr/bin/env python
"""Generate and freeze the FormalCRRC Day-1 formal dataset.

Writes, under ``artifacts/day1/``:

* ``formal_dataset.parquet``  -- 2916 rubric-response pairs with formal labels
* ``prompt_manifest.parquet`` -- judge prompts plus the frozen inference order
* ``dataset_manifest.json``   -- dimensions, seeds and checksums

Structural invariants are asserted here as well as in the test suite, so a
corrupt dataset can never be written silently. No model is involved.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from formalcrrc import dataset as ds  # noqa: E402
from formalcrrc import predicates, prompts, provenance  # noqa: E402
from formalcrrc.config import (  # noqa: E402
    DATASET_MANIFEST_PATH,
    DATASET_PATH,
    FAMILIES,
    LIST_SIZE,
    MARKERS_PER_RESPONSE,
    N_ARTIFACTS,
    N_INSTANCES,
    N_LATENT_LEVELS,
    N_ROWS,
    N_STRICTNESS,
    PROMPT_MANIFEST_PATH,
    SEED,
)


def assert_structure(frame: pd.DataFrame) -> None:
    """Fail loudly if any construction invariant is violated."""
    assert len(frame) == N_ROWS, f"expected {N_ROWS} rows, got {len(frame)}"
    assert frame["artifact_id"].nunique() == N_ARTIFACTS, "artifact count mismatch"

    for artifact_id, group in frame.groupby("artifact_id", sort=False):
        levels = sorted(group["strictness_index"].tolist())
        assert levels == list(range(N_STRICTNESS)), (
            f"{artifact_id}: strictness levels are {levels}"
        )
        assert group["candidate_response"].nunique() == 1, (
            f"{artifact_id}: candidate response is not byte-identical"
        )
        family = group["family"].iloc[0]
        normalized = {
            predicates.normalize_rubric(family, text)
            for text in group["rubric_text"]
        }
        assert len(normalized) == 1, (
            f"{artifact_id}: rubrics differ outside the threshold field"
        )
        truth = group.sort_values("strictness_index")["formal_truth"].tolist()
        assert truth == sorted(truth, reverse=True), (
            f"{artifact_id}: formal labels are not monotone in strictness"
        )
        recomputed = [
            predicates.formal_truth(family, int(group["latent_level"].iloc[0]), s)
            for s in range(N_STRICTNESS)
        ]
        assert truth == recomputed, f"{artifact_id}: formal truth not reproducible"

    cells = Counter(zip(frame["family"], frame["latent_level"]))
    expected = N_INSTANCES * N_STRICTNESS
    for family in FAMILIES:
        for latent in range(N_LATENT_LEVELS):
            assert cells[(family, latent)] == expected, (
                f"{family}/L{latent}: expected {expected} rows, "
                f"got {cells[(family, latent)]}"
            )

    markers = frame[frame["family"] != "numeric_tolerance"]
    for payload in markers["response_markers_json"].unique():
        assert payload.count("MK-") == MARKERS_PER_RESPONSE, (
            "marker responses must all carry the same number of markers"
        )
    for payload in markers["list_items_json"].unique():
        assert payload.count("MK-") == LIST_SIZE, "item lists must hold eight items"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=str(provenance.repo_root()),
        help="repository root under which artifacts are written",
    )
    args = parser.parse_args()
    root = Path(args.root)

    print("building formal dataset ...", flush=True)
    frame = ds.build_dataset()
    assert_structure(frame)

    manifest_frame = prompts.build_prompt_manifest(frame)
    assert len(manifest_frame) == N_ROWS
    assert manifest_frame["order_index"].is_unique

    dataset_path = root / DATASET_PATH
    prompt_path = root / PROMPT_MANIFEST_PATH
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(dataset_path, index=False)
    manifest_frame.to_parquet(prompt_path, index=False)

    content_hash = ds.dataset_content_hash(frame)
    manifest = {
        "created_at": provenance.utc_now(),
        "seed": SEED,
        "dimensions": {
            "families": list(FAMILIES),
            "n_families": len(FAMILIES),
            "n_latent_levels": N_LATENT_LEVELS,
            "n_instances_per_cell": N_INSTANCES,
            "n_artifacts": N_ARTIFACTS,
            "n_strictness_levels": N_STRICTNESS,
            "n_rows": N_ROWS,
            "list_size": LIST_SIZE,
            "markers_per_response": MARKERS_PER_RESPONSE,
        },
        "columns": list(ds.DATASET_COLUMNS),
        "hashes": {
            "dataset_content_sha256": content_hash,
            "dataset_file_sha256": provenance.sha256_file(dataset_path),
            "prompt_manifest_file_sha256": provenance.sha256_file(prompt_path),
            "prompt_template_sha256": prompts.prompt_template_sha256(),
            "rubric_template_sha256": {
                family: provenance.sha256_text(predicates.RUBRIC_TEMPLATES[family])
                for family in FAMILIES
            },
        },
        "true_first_fail_index_by_cell": {
            family: {
                str(latent): predicates.true_first_fail_index(family, latent)
                for latent in range(N_LATENT_LEVELS)
            }
            for family in FAMILIES
        },
        "source": provenance.source_manifest(
            ["src/formalcrrc/config.py",
             "src/formalcrrc/predicates.py",
             "src/formalcrrc/dataset.py",
             "src/formalcrrc/prompts.py"],
            root=root,
        ),
        "git": provenance.git_state(root),
    }
    provenance.write_json(root / DATASET_MANIFEST_PATH, manifest)

    print(f"artifacts:            {N_ARTIFACTS}", flush=True)
    print(f"rubric-response rows: {len(frame)}", flush=True)
    print(f"dataset content hash: {content_hash}", flush=True)
    print(f"wrote {DATASET_PATH}", flush=True)
    print(f"wrote {PROMPT_MANIFEST_PATH}", flush=True)
    print(f"wrote {DATASET_MANIFEST_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
