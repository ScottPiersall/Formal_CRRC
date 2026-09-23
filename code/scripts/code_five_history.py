"""Record exactly which historical Q3 evidence was verified without rewriting it."""
import pathlib,sys,json,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from formalcrrc import code_extension as cp
o=ROOT/'artifacts/code_extension_v3_five_models';folder=ROOT/'artifacts/reasoning_anchor'
raw=(folder/'preregistration.json').read_bytes();expected=(folder/'preregistration.sha256').read_text().split()[0]
config=json.loads(raw);historical=cp.load(folder/'run.json')
assert config['model']['revision']=='144afc2f379b542fdd4e85a1fcd5e1f79112d95d'
source_checks={}
for name,value in historical['source_manifest']['files'].items():
    expected_hash=value['sha256'] if isinstance(value,dict) else value
    p=ROOT/name
    if p.exists():
        b=p.read_bytes();source_checks[name]=dict(expected_sha256=expected_hash,raw_sha256=cp.sha(b),lf_normalized_sha256=cp.sha(b.replace(b'\r\n',b'\n')),
            raw_matches=cp.sha(b)==expected_hash,lf_normalized_matches=cp.sha(b.replace(b'\r\n',b'\n'))==expected_hash)
record=dict(created_at=cp.now(),preregistration_expected_sha256=expected,preregistration_raw_sha256=cp.sha(raw),
    preregistration_lf_normalized_sha256=cp.sha(raw.replace(b'\r\n',b'\n')),raw_checksum_matches=cp.sha(raw)==expected,
    lf_normalized_checksum_matches=cp.sha(raw.replace(b'\r\n',b'\n'))==expected,
    frozen_model=config['model'],frozen_protocol=config['protocol'],historical_sources=source_checks,
    deviation='Do not copy historical runner fallback that appends a missing stop ID; new protocol requires raw native completion tokens.',
    scope='Local historical files preserved byte-for-byte. LF comparison is explicit transport-line-ending diagnosis, not an assertion that raw checksum passed.')
cp.immutable(o/'historical_q3_review.json',record)
print(json.dumps({k:record[k] for k in ['raw_checksum_matches','lf_normalized_checksum_matches']}))
print('SOURCE_VERIFICATION',json.dumps({k:(v['raw_matches'],v['lf_normalized_matches']) for k,v in source_checks.items() if k in ['scripts/run_reasoning_anchor_inference.py','src/formalcrrc/reasoning_anchor.py','src/formalcrrc/reasoning_anchor_config.py']}))
