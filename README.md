# FormalCRRC Anonymous Reproducibility Package


## 1. Verify the download

```bash
python verify.py
```

`MANIFEST.json` authenticates the files in this anonymous export. `MERGE_PROVENANCE.json` records the two input-archive hashes without exposing author or machine identity. This export is a derivative package: sanitized source files do not retain original freeze hashes. It does not claim external preregistration. See `SCOPE.md` for the exact manuscript-to-artifact mapping.

## 2. Reproduce the MBPP extension

Use Python 3.14 and the original NumPy version in a separate environment:

```bash
python -m venv .venv-mbpp
# Linux/macOS: source .venv-mbpp/bin/activate
# Windows: .venv-mbpp\Scripts\Activate.ps1
python -m pip install -r requirements-mbpp.txt
python reproduce_mbpp.py --out outputs/mbpp
```

This recomputes ten result tables from 19,872 threshold-condition receipts and 1,104 count outcomes: task-level metrics, stratified estimates and bounds, paired effects, the six primary tests, secondary interactions, count summaries, and visible-prefix diagnostics. Results are compared with the saved analysis at a numerical tolerance of 1e-12. It preserves all 276 tasks, including missing scores and protocol failures. Only 33 tasks are partially correct.

## 3. Reproduce historical and control analyses

Use Python 3.12 with a separate environment:

```bash
python -m venv .venv-analysis
# Activate this environment using the platform-specific command above.
python -m pip install -r requirements-analysis.txt
python reproduce_analysis.py --out outputs/analysis
```

The runner verifies the package, rebuilds the historical panel from 111,888 threshold rows, recomputes synthetic immediate/RTS metrics and bootstrap intervals, checks code-control curve metrics, and rebuilds held-out calibration. It compares the main historical and calibration tables against saved results and exits nonzero on a discrepancy. Historical post-hoc intervals remain distinct from the original study intervals.

## Package map

| Path | Contents |
|---|---|
| `code/src`, `code/scripts`, `code/tests` | Research implementation, original pipeline scripts, and tests |
| `studies/legacy` | Synthetic studies and historical five-model code-panel exports |
| `studies/mbpp` | Independent extension receipts, statistical code, and expected results |
| `studies/synthetic_rts` | Within-model synthetic immediate/RTS score pairs |
| `studies/code_controls` | Original/corrected RTS and immediate control exports |
| `studies/calibration` | Frozen folds, normalized scores, calibration code, expected outputs |
| `studies/replication14` | Fourteen-task independent extension score exports and curve checks |
| `inference`, `protocols` | Inference implementation and exported protocol/model settings |
| `SCOPE.md` | Manuscript-to-artifact inclusion map and explicit exclusions |
| `VALIDATION.md` | Checks and inherited scientific reproduction receipts for this release |
| `REPRODUCIBILITY.md` | Scope, exclusions, and inference prerequisites |
| `UPLOAD.md` | Anonymous GitHub upload instructions |

The code-control exports contain historical protocol versions; these are paired representations or repeated protocols on shared tasks, not additional independent samples. Preserve study, model, template, readout, split, and version identifiers when aggregating.

See `SCOPE.md`, `LICENSE-REVIEW.md`, and `THIRD_PARTY.md` before redistribution. This package does not include model weights or enhanced benchmark test bodies.
