"""Derive only lifecycle/output-gate changes from the archived Table 8 runner."""
import difflib,pathlib
O=pathlib.Path(__file__).resolve().parents[1]
for n in ('protocol.py','cost_gate.py','test_protocol.py'):
 p=O/'src'/n;b=(O/'reference/table8_v2/src'/n).read_bytes()
 if p.exists():assert p.read_bytes()==b
 else:p.write_bytes(b)
p=O/'reference/table8_v2/src/run_inference.py';old=p.read_text();s=old
s=s.replace("assert gate['job_id']==self.job and gate['stage']==stage and gate['quota_gate_passed'] and gate['project_budget_gate_passed']", "assert gate['job_id']==self.job and gate['stage'] in (stage,'combined') and gate['quota_gate_passed'] and gate['project_budget_gate_passed']")
s=s.replace("if stage=='main':assert load(OUT/'costs/smoke_gate.json')['technical_gate_passed']", "if stage=='main':assert load(OUT/'preregistration/cpu_engineering_gate.json')['passed']")
s=s.replace("gate['reserved_physical_gpu_seconds']<=180000", "gate['reserved_physical_gpu_seconds']<=14400")
assert s!=old
p=O/'src/run_inference.py'
if p.exists():assert p.read_text()==s
else:p.write_text(s)
(O/'preregistration/runner_adapter.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='Table8_v2/run_inference.py',tofile='replication/run_inference.py')))
print('Runner derived; scorer, seeds, stop parsing and generation unchanged.')
