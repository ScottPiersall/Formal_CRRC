"""Independent evidence audit, supplemental report, and portable handoff checks.

This is post-inference auditing/packaging only. Frozen scoring and analysis source
files remain byte-identical; no additional model requests are issued.
"""
import ast
import csv
import datetime
import json
import math
import pathlib
import sys
import tempfile
import zipfile
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from formalcrrc import code_pilot as cp
from formalcrrc import code_pilot_analysis as analysis

ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=cp.outdir(ROOT)

def audit():
    assert cp.integrity(ROOT)['passed']
    a=cp.load(OUT/'freeze_A.json'); b=cp.load(OUT/'freeze_B.json')
    stamp=lambda x:datetime.datetime.fromisoformat(x)
    instances=cp.load(OUT/'instances.json')
    source={r['task_id']:r for r in (json.loads(x) for x in (OUT/'upstream/HumanEvalPlus-v0.1.10.jsonl').read_text().splitlines())}
    valid_truth={}; observations=0
    compact=lambda v:json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'))
    for item in instances:
        task=item['task_id']; g=cp.load(OUT/'generation'/f'{cp.slug(task)}.json'); t=cp.load(OUT/'truth'/f'{cp.slug(task)}.json')
        assert stamp(a['created_at'])<stamp(g['created_at'])<=stamp(b['created_at'])
        assert stamp(t['created_at'])<=stamp(b['created_at']) and t['stable']
        assert t['code_sha256']==cp.sha(g['code'])==g['code_sha256']
        inp=[compact(x['args']) for x in item['tests']]
        assert len(set(inp))==8 and all(len(x.encode())<=512 for x in inp)
        assert not set(inp)&{compact(x) for x in source[task]['base_input']}
        assert set(inp)<={compact(x) for x in source[task]['plus_input']}
        expected=[ast.literal_eval(x['expected_repr']) for x in item['tests']]
        assert all(len(x['expected_repr'].encode())<=512 for x in item['tests'])
        for role in ['reference_execution','execution']:
            rows=t[role]['tests']; assert len(rows)==16
            for r in rows:
                assert r['category']!='infrastructure_error'
                if r['category']=='ok':
                    try: match=bool(ast.literal_eval(r['observed_repr'])==expected[r['test_index']])
                    except (SyntaxError,ValueError): match=False
                else: match=False
                assert match==r['passed']
                if role=='reference_execution': assert match
                observations+=1
            assert [r['passed'] for r in rows[:8]]==[r['passed'] for r in rows[8:]]
        bits=[int(r['passed']) for r in t['execution']['tests'][:8]]
        assert t['pass_bits']==bits and t['z']==sum(bits)
        valid_truth[task]=t
    metrics=list(csv.DictReader((OUT/'analysis/per_instance_metrics.csv').open()))
    checked=0; normalized=0; endpoints={}; magnitudes=[]
    for key in cp.MODELS:
        p=cp.load(OUT/'prompts'/f'{key}.json'); grouped={}
        for r in p['rows']: grouped.setdefault(r['task_id'],[]).append(r)
        for task,rows in grouped.items():
            messages=[]
            for r in rows:
                data=r['user_message'].encode(); start,end=r['threshold_span']
                assert data[start:end]==str(r['k']).encode()
                messages.append(data[:start]+b'{threshold}'+data[end:])
            assert len(rows)==9 and len(set(messages))==1
            normalized+=1
        for item in [i for i in instances if i['split']=='main']:
            task=item['task_id']; z=valid_truth[task]['z']; j=z+1
            score=[cp.load(OUT/'scores/main'/key/f'{cp.score_key(task,k)}.json') for k in range(9)]
            assert all(stamp(s['created_at'])>=stamp(b['created_at']) for s in score)
            m=[s['score_met']-s['score_not_met'] for s in score]
            truth=set(range(j)); cuts=[set()]; active=set()
            for value in sorted(set(m),reverse=True):
                active=active|{i for i,x in enumerate(m) if x==value}; cuts.append(active)
            raw={i for i,x in enumerate(m) if x>=0}
            error=len(raw^truth); first=next((k for k in range(9) if k not in raw),9)
            r=next(r for r in metrics if r['model_key']==key and r['task_id']==task)
            assert int(r['errors'])==error and abs(float(r['accuracy'])-(9-error)/9)<1e-14
            assert int(r['oracle_min_errors'])==min(len(c^truth) for c in cuts)
            assert int(r['tce'])==abs(first-j)
            if z<8:
                trr=any(next((k for k in range(9) if k not in c),9)==j for c in cuts)
                fsrr=any(c==truth for c in cuts)
                assert r['trr']==str(trr) and r['fsrr']==str(fsrr)
                if 1<=z<=7 and not fsrr: magnitudes.append(abs(float(r['fsrr_gap'])))
            else: assert r['trr']=='' and r['fsrr']==''
            endpoints.setdefault(key,[]).append(0 in raw)
            checked+=1
    result=dict(created_at=cp.now(),passed=True,independently_recomputed_main_curves=checked,
                reference_and_candidate_observations_rechecked=observations,
                normalized_prompt_groups=normalized,freeze_chronology_passed=True,
                source_test_membership_and_size_limits_passed=True,
                k0_accuracy={k:sum(v)/len(v) for k,v in endpoints.items()},
                partial_fsrr_failure_abs_gap_range=[min(magnitudes),max(magnitudes)] if magnitudes else None,
                method='Independent descending tied-score group enumeration; direct set disagreement and first-failure positions; no reuse of curve_diagnostics for the independent calculations')
    cp.immutable(OUT/'independent_evidence_audit.json',result)
    return result

def supplement(result):
    s=cp.load(OUT/'analysis/summary.json')
    lines=['# Pilot v1: additional auditable detail',
       '\nThis supplement reports frozen-run evidence. It does not amend the protocol or add model calls.',
       f"\nIndependent audit: {result['independently_recomputed_main_curves']} curves, {result['reference_and_candidate_observations_rechecked']} execution observations, {result['normalized_prompt_groups']} prompt-invariance groups; all passed.",
       '\n## Partial-correct candidates (z=1,…,7; n=10)',
       '\n| Model | Accuracy | TRR | FSRR | Oracle minimum errors |',
       '|---|---|---|---|---|']
    for key,g in s['model_summaries'].items():
        p=g['partial']; lines.append('| '+key+' | '+' | '.join(analysis.fmt(p[m]) for m in ['accuracy','trr','fsrr','oracle_min_errors'])+' |')
    lines+=['\nIntervals are task-bootstrap percentile 95% intervals. A [0,0] interval when every sampled task fails is a degenerate empirical bootstrap interval, not proof that the population recovery rate is exactly zero.',
       '\n## Paired Qwen minus Mistral differences',
       '\n| Metric | Paired difference and 95% CI |', '|---|---|']
    for metric,stat in s['paired']['all']['qwen_minus_mistral'].items(): lines.append(f'| {metric} | {analysis.fmt(stat)} |')
    lines+=[f"\nThe universally true k=0 criterion was correctly classified in {result['k0_accuracy']['mistral']:.1%} of Mistral cases and {result['k0_accuracy']['qwen']:.1%} of Qwen cases. Thus the strict-threshold accuracy is essential here; the easy endpoint did not inflate observed accuracy.",
      f"\nAbsolute FSRR separation-failure gaps on partial-correct tasks ranged from {result['partial_fsrr_failure_abs_gap_range'][0]:.6g} to {result['partial_fsrr_failure_abs_gap_range'][1]:.6g}; none was within the predefined 0.01 near-zero diagnostic range. This is not a substitute for independent numerical/template robustness experiments.",
      '\nNo TRR-recoverable but FSRR-unrecoverable case was observed in either model. Equal-error/different-recovery examples do exist; see summary.json. The GO rule was met through FSRR-unrecoverable partial-correct tasks, not through the absent second pattern.',
      '\nThe decision threshold of ten partial-correct candidates was met exactly. A redesigned or expanded study must use new tasks. Mistral generated all candidates and was also a judge. These exploratory findings do not demonstrate population-wide judge unreliability or a reasoning-model benefit.',
      '\n## Sources',
      '\n- [Pinned EvalPlus source](https://github.com/evalplus/evalplus/tree/e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2)',
      '- [HumanEval+ v0.1.10 release](https://github.com/evalplus/humanevalplus_release/releases/tag/v0.1.10)',
      '- [HumanEval MIT license at resolved source commit](https://github.com/openai/human-eval/blob/6d43fb980f9fee3c892a914eda09951f772ad10d/LICENSE)']
    (OUT/'analysis/ADDITIONAL_DETAIL.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def build_portable():
    analysis.package(ROOT)
    target=ROOT.parent/'FormalCRRC_Code_Pilot_v1_Handoff.zip'
    with zipfile.ZipFile(target) as z:
        payload={name:z.read(name) for name in z.namelist()}
    for name in ['tests/conftest.py','tests/test_scoring.py','pyproject.toml']:
        payload[name]=(ROOT/name).read_bytes()
    # Keep only the pilot modules and the original package's import closure;
    # unrelated older study modules are not part of the handoff.
    needed={'__init__','scoring','provenance','sequence_recovery'}
    needed.update(p.stem for p in (ROOT/'src/formalcrrc').glob('code_pilot*.py'))
    todo=list(needed)
    while todo:
        module=todo.pop(); tree=ast.parse((ROOT/'src/formalcrrc'/f'{module}.py').read_text())
        imports=set()
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom) and node.module=='formalcrrc': imports.update(a.name for a in node.names)
            elif isinstance(node,ast.ImportFrom) and (node.module or '').startswith('formalcrrc.'):
                imports.add(node.module.split('.')[1])
            elif isinstance(node,ast.Import):
                imports.update(a.name.split('.')[1] for a in node.names if a.name.startswith('formalcrrc.'))
        for name in imports:
            if name not in needed and (ROOT/'src/formalcrrc'/f'{name}.py').is_file():
                needed.add(name); todo.append(name)
    for name in list(payload):
        if name.startswith('src/formalcrrc/') and name.endswith('.py') and pathlib.PurePosixPath(name).stem not in needed:
            del payload[name]
    manifest_name='artifacts/code_pilot_v1/handoff_manifest.json'
    manifest=json.loads(payload[manifest_name])
    manifest['files']={n:h for n,h in manifest['files'].items() if n in payload}
    manifest['dependency_modules']=sorted(needed)
    payload[manifest_name]=cp.canonical(manifest).encode()
    (ROOT/manifest_name).write_bytes(payload[manifest_name])
    payload['README.md']=('''# FormalCRRC Code Pilot v1 handoff

See `docs/CODE_PILOT_V1.md` for verified environment and commands.
Start with `artifacts/code_pilot_v1/execution_status.json`, `analysis/REPORT_ZH.md`,
`analysis/METHODS_RESULTS_EN.md`, and `analysis/ADDITIONAL_DETAIL.md` beneath that directory.
The original Day 1–3 result files and model weights are intentionally not bundled.
Raw pilot replies, truth, prompts, scores, freeze checksums, code patch and tests are included.
Install the pinned local packages documented in the runbook before running:

```bash
python -m pytest tests/test_code_pilot.py tests/test_scoring.py
python scripts/code_pilot.py check-integrity
```

The archive manifest hashes every member except itself; the adjacent .sha256 hashes this zip.
''').encode()
    payload['archive_manifest.json']=cp.canonical(dict(created_at=cp.now(),files={n:cp.sha(data) for n,data in sorted(payload.items())})).encode()
    temp=target.with_suffix('.building.zip')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,data in sorted(payload.items()): z.writestr(name,data)
    temp.replace(target)
    with zipfile.ZipFile(target) as z:
        assert len(z.namelist())==len(set(z.namelist())) and z.testzip() is None
        m=json.loads(z.read('archive_manifest.json'))
        assert all(cp.sha(z.read(n))==digest for n,digest in m['files'].items())
        assert not any(n.endswith(('.safetensors','.bin','.pt','.pem','.key')) for n in z.namelist())
        print('Verified archive entries:',len(z.namelist()))
    digest=cp.sha(target.read_bytes()); target.with_suffix('.zip.sha256').write_text(f'{digest}  {target.name}\n')
    print('FINAL_HANDOFF',target,'SHA256',digest)

if __name__=='__main__':
    audit_path=OUT/'independent_evidence_audit.json'
    result=cp.load(audit_path) if audit_path.exists() else audit()
    supplement(result)
    build_portable()
