# Unified package validation

The unified package was constructed from research archives using the manuscript as the inclusion specification.

## Scientific reproduction evidence carried forward

`UPSTREAM_VALIDATION_RESULTS.json` records successful saved-score reproductions from the compact numerical-evidence source package:

- MBPP: 10 result tables, 19,872 threshold conditions, 276 tasks.
- Historical analysis: 111,888 exported score rows and 12,312 complete curves.
- Synthetic RTS controls: 324 artifacts and 3,888 metric comparisons.
- Historical code controls: 1,357 complete full-threshold curves.
- Fourteen-task extension: 224 complete curve variants.
- Held-out calibration: saved expected tables reproduced in the source package.

No new model inference is represented by those checks.

## Unified-package checks

The compact synthetic/control runner was re-executed after the merge and passed: 324 synthetic artifacts, 3,888 synthetic metric checks, 1,357 complete historical code-control curves, and 224 complete 14-task extension curve variants.

The final package is additionally checked for:

- Python syntax compilation for every shipped `.py` file;
- absence of the known author/workstation/cluster account identifiers found in the input archives;
- absence of private-key headers and common API/token key patterns;
- absence of original Git metadata, caches, and credential files.

Because this build environment does not have the exact pinned NumPy/Python combinations required by both saved-score runners, the full numerical runners were not re-executed after curation. Curation changes do not modify the saved score/truth datasets or statistical definitions; they remove operational material, replace one SLURM account string with a placeholder, and add the missing synthetic RTS source implementation.
