"""Freeze the local protocol and code before any new inference; attest gates."""
import ast,subprocess,xml.etree.ElementTree as ET
from protocol import *
def main():
 if (OUT/'preregistration/freeze.json').exists():
  frozen=load(OUT/'preregistration/freeze.json')
  for rel,h in frozen['files'].items():assert sha((OUT/rel).read_bytes())==h,rel
  print('FROZEN FILES VERIFIED');return
 assert len(rows(OUT/'preregistration/inference_inputs.jsonl'))==1660
 assert not list((OUT/'traces').glob('A/**/*.json')) and not list((OUT/'traces').glob('B/**/*.json'))
 root=OUT.parents[1];dependencies=load(OUT/'preregistration/dependencies.json')['files']
 for rel,h in dependencies.items():assert sha((root/rel).read_bytes())==h,'Prior evidence changed: '+rel
 reused={}
 for p in (OUT/'preregistration/reused_source/formalcrrc').glob('*.py'):
  original=root/'src/formalcrrc'/p.name;assert p.read_bytes()==original.read_bytes()
  reused[original.relative_to(root).as_posix()]=sha(p.read_bytes())
 write(OUT/'preregistration/reused_source_manifest.json',dict(files=reused,scope='Entire source-only package import closure, copied byte for byte; no unrelated datasets'),True)
 test=ET.parse(OUT/'analysis/cpu_tests_final.xml').getroot();suites=list(test.iter('testsuite'))
 assert sum(int(s.attrib.get('failures',0))+int(s.attrib.get('errors',0)) for s in suites)==0
 assert sum(int(s.attrib['tests']) for s in suites)==7
 compiled=[]
 for p in list((OUT/'src').glob('*.py'))+list((OUT/'analysis').glob('*.py')):
  ast.parse(p.read_text());compiled.append(p.relative_to(OUT).as_posix())
 write(OUT/'preregistration/local_cpu_gate.json',dict(created_at=now(),status='passed',pytest_tests=7,
  tests_file='analysis/cpu_tests_final.xml',tests_sha256=sha((OUT/'analysis/cpu_tests_final.xml').read_bytes()),syntax_checked=compiled,
  tokenizer_probe_contexts=1494,independent_existing_curve_checks=640,
  checks_not_run=['Native Transformers tokenizer equality on CLUSTER','GPU smoke and numeric prefix scoring','Throughput and loading','Institutional quota/remaining balance'],
  no_new_model_inference=True),True)
 paths=[p for directory in ['preregistration','src'] for p in (OUT/directory).rglob('*') if p.is_file()]
 paths += list((OUT/'analysis').glob('*.py'))+[OUT/'data/truth.jsonl',OUT/'data/immediate_scores.jsonl',OUT/'analysis/cpu_tests_final.xml']
 files={p.relative_to(OUT).as_posix():sha(p.read_bytes()) for p in sorted(paths)}
 write(OUT/'preregistration/freeze.json',dict(created_at=now(),study='Post-observation code supplement; not original preregistration',stage='Before smoke and main inference',
  state='Frozen local protocol; remote native-tokenizer, GPU, quota and smoke gates pending',files=files),True)
 print('FROZEN',len(files),'files',sha((OUT/'preregistration/freeze.json').read_bytes()))
if __name__=='__main__':main()
