"""Verify every shipped file using only the Python standard library."""
from pathlib import Path
import hashlib, json, sys

ROOT = Path(__file__).resolve().parent

def verify():
    manifest = json.loads((ROOT/'MANIFEST.json').read_text(encoding='utf-8'))
    failures = []
    for rel, expected in manifest['files'].items():
        p = (ROOT/rel).resolve()
        if not p.is_relative_to(ROOT) or not p.is_file():
            failures.append(rel + ': missing or unsafe')
        elif hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            failures.append(rel + ': hash mismatch')
    if failures:
        print('\n'.join(failures)); raise SystemExit(1)
    print(f"PASS: {len(manifest['files'])} package files verified.")

if __name__ == '__main__': verify()
