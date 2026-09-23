"""Read-only v2 audit and additive five-model freeze preparation."""
import hashlib
import json
import pathlib
import subprocess
import sys
from datetime import datetime, timezone

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from formalcrrc import code_extension as cp
OUT=ROOT/'artifacts/code_extension_v3_five_models'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'v2_reuse_audit.json').exists(): print('V2 audit already saved'); return
    (OUT/'git_status_before.txt').write_bytes(subprocess.check_output(['git','status','--short'],cwd=ROOT))
    (OUT/'git_diff_before.patch').write_bytes(subprocess.check_output(['git','diff','--binary'],cwd=ROOT))
    # Redirect the audit's sole output. No frozen v2 source or result is written.
    original_snapshot=cp.snapshot
    def redirected(path,data):
        path=pathlib.Path(path)
        if path.is_relative_to(ROOT/'artifacts/code_extension_v2'):
            path=OUT/'prior_audit_outputs'/path.relative_to(ROOT/'artifacts/code_extension_v2')
        original_snapshot(path,data)
    cp.snapshot=redirected
    from formalcrrc import code_extension_analysis as analysis
    from formalcrrc import code_extension_answer_analysis as forms
    items,truths,raw,curves,audit=analysis.audit(ROOT)
    ar,ac,aa=forms.audit(ROOT,items,truths)
    _,_,numeric=forms.audit_precision(ROOT,truths,ac,ar)
    audit.update(answer_forms=aa,numeric_audit_retained=True,partial_main_tasks=sum(i['split']=='main' and 1<=truths[i['task_id']]['z']<=7 for i in items))
    cp.immutable(OUT/'v2_reuse_audit.json',audit)
    deps=[p for p in (ROOT/'artifacts/code_extension_v2').rglob('*') if p.is_file()]
    deps+=cp.source_files(ROOT)
    deps += [ROOT/p for p in ['src/formalcrrc/reasoning_anchor.py','src/formalcrrc/reasoning_anchor_config.py','scripts/run_reasoning_anchor_inference.py','artifacts/reasoning_anchor/preregistration.json','artifacts/reasoning_anchor/preregistration.sha256','artifacts/reasoning_anchor/model_provenance.json','docs/PREREGISTRATION_REASONING_ANCHOR.md']]
    cp.immutable(OUT/'v2_dependency_manifest.json',dict(created_at=cp.now(),files=cp.manifest(ROOT,deps),scope='Read-only v2 evidence and verified shared sources; post-v2 extension, not original v2 preregistration'))
    # Copy only the frozen input messages, not a generator or rewritten template.
    prompts=cp.load(ROOT/'artifacts/code_extension_v2/prompts/qwen.json')['rows']
    rows=[]
    for p in prompts:
        r={k:p[k] for k in ['task_id','split','variant','k','user_message','threshold_span','normalized_sha256']}
        r['z']=truths[r['task_id']]['z']
        r['user_message_sha256']=cp.sha(r['user_message'])
        payload=f"42|code_extension_v3_five_models|{r['task_id']}|{r['variant']}|{r['k']}"
        r['generation_seed']=int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8],'big')%2147483647
        r['seed_payload']=payload
        rows.append(r)
    assert len(rows)==1494 and len({r['generation_seed'] for r in rows})==1494
    rows.sort(key=lambda r:(r['split']!='smoke',r['task_id'],r['variant'],r['k']))
    cp.immutable(OUT/'inputs.json',dict(created_at=cp.now(),rows=rows))
    old=cp.load(ROOT/'artifacts/code_extension_v2/execution_status.json')
    cp.immutable(OUT/'budget.json',dict(created_at=cp.now(),cumulative_cap_gpu_seconds=28800,prior_actual_gpu_seconds=6824,
        remaining_actual_gpu_seconds=21976,prior_conservative_reservations_gpu_seconds=18000,
        conservative_new_reservation_cap_gpu_seconds=10800,
        policy='No fresh eight-hour authorization. Use at most the smaller conservative remaining reservation (3 h); retain actual remaining allocation (6.1044 h) separately. All v3 allocations including failed jobs and loading count. Release no historical reservation implicitly.',
        prior_status_sha256=cp.sha((ROOT/'artifacts/code_extension_v2/execution_status.json').read_bytes())))
    print(json.dumps(audit))

if __name__=='__main__': main()
