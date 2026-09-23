"""Acquire the explicitly pinned upstream release; never import or execute it."""
import gzip
import hashlib
import json
import pathlib
import shutil
import urllib.request
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/code_pilot_v1/upstream'
COMMIT = 'e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2'
URL = 'https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus.jsonl.gz'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'HumanEvalPlus-v0.1.10.jsonl.gz'
    if not target.exists():
        with urllib.request.urlopen(URL, timeout=120) as response:
            target.write_bytes(response.read())
    data = OUT / 'HumanEvalPlus-v0.1.10.jsonl'
    payload = gzip.decompress(target.read_bytes())
    if data.exists():
        assert data.read_bytes() == payload
    else:
        data.write_bytes(payload)
    source = ROOT / '.remote/code_pilot_v1_upstream'
    for path in ['LICENSE', 'evalplus/data/humaneval.py', 'evalplus/data/utils.py', 'evalplus/eval/__init__.py', 'setup.cfg']:
        dest = OUT / 'evalplus_v0.3.1' / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copyfile(source / path, dest)
    license_url = 'https://raw.githubusercontent.com/evalplus/humanevalplus_release/v0.1.10/LICENSE'
    # Release repository may have no LICENSE at this tag. HumanEval's source
    # license is fetched at a resolved commit, and provenance distinguishes them.
    record = dict(captured_at=datetime.now(timezone.utc).isoformat(), package='evalplus',
                  package_version='0.3.1', source_commit=COMMIT, dataset_version='v0.1.10',
                  url=URL, source_license='Apache-2.0 (vendored LICENSE)',
                  dataset_license_note='HumanEval derives from MIT-licensed openai/human-eval; EvalPlus repository Apache-2.0. No separate license in the dataset payload.',
                  files={str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'source_manifest.json'})
    dest = OUT / 'source_manifest.json'
    if not dest.exists():
        dest.write_text(json.dumps(record, indent=2) + '\n')
    print('Pinned dataset:', len(payload.splitlines()), 'tasks', 'SHA256', record['files'][data.name])

if __name__ == '__main__':
    main()
