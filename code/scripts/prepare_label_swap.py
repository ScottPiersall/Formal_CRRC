#!/usr/bin/env python
"""Freeze the supplementary 81-artifact sample before scoring (idempotent)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from formalcrrc import label_swap as ls, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=provenance.repo_root())
    parser.add_argument("--experiment-id", default=ls.DEFAULT_EXPERIMENT)
    parser.add_argument("--lf-root", type=Path, help="independent historical checkout with raw LF bytes")
    args = parser.parse_args()
    print(ls.prepare(args.root, args.experiment_id, args.lf_root))


if __name__ == "__main__":
    main()
