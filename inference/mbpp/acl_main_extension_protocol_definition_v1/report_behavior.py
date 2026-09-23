"""Read existing RTS receipts and report observable behavior, never scientific scores.

This review command is deliberately bound to the completed development work and
its CPU decode audit. Formal receipt integration remains a Stage B deliverable.
"""
from collections import Counter
import json
import math
from build_review import ROOT, RETRY, read, save, sha


def classify(text, technically_valid):
    marker = text.find('FINAL:') if isinstance(text, str) else -1
    if technically_valid and marker < 0:
        raise ValueError('A valid RTS context must have an observed FINAL boundary')
    prefix = text[:marker] if marker >= 0 else None
    visible = bool(prefix.strip()) if prefix is not None else None
    return {'marker_observed': marker >= 0,
            'visible_prefix_observed': visible,
            'pre_marker_characters': len(prefix) if prefix is not None else None,
            'pre_marker_whitespace_words': len(prefix.split()) if prefix is not None else None,
            'valid_scored_context': technically_valid,
            'valid_marker_only_context': technically_valid and visible is False,
            'valid_visible_prefix_context': technically_valid and visible is True}


def aggregate(rows):
    planned = len(rows)
    valid = sum(r['valid_scored_context'] for r in rows)
    marker = sum(r['valid_marker_only_context'] for r in rows)
    visible = sum(r['valid_visible_prefix_context'] for r in rows)
    assert marker + visible == valid
    return {'planned_contexts':planned, 'observed_generation_requests':sum(r['request_observed'] for r in rows),
            'valid_scored_contexts':valid, 'valid_marker_only':marker, 'valid_visible_prefix':visible,
            'missing_contexts':planned-valid, 'failure_reasons':dict(Counter(r['failure_reason'] for r in rows if r['failure_reason'])),
            'marker_only_over_valid':marker/valid if valid else None,
            'observed_marker_only_over_planned':marker/planned if planned else None,
            'valid_over_planned':valid/planned if planned else None,
            'generated_tokens_including_marker_and_extras':sum(r['generated_tokens'] or 0 for r in rows)}


def main():
    inputs = [r for r in read(RETRY/'work/judge_inputs.json') if r['kind']=='corrected_RTS_v2']
    checks = {r['request_id']:r for r in read(ROOT/'audit/CLUSTER_cpu_refresh.json')['decode_checks']}
    rows=[]
    for judge in ['qwen','mistral']:
        for info in inputs:
            key=judge+'_'+info['condition_id'];rp=RETRY/'work/requests'/(key+'_generation.json');sp=RETRY/'work/judge_results'/(key+'.json')
            request=read(rp) if rp.exists() else None;result=read(sp) if sp.exists() else None
            valid=bool(result and result.get('status')=='ok')
            if valid:
                if request is None: raise ValueError('Valid result without generation receipt: '+key)
                audit=checks[request['request_id']]
                if not (audit['full_decode_matches_returned_text'] and audit['actual_accepted_prefix_verified'] and sha(rp)==audit['request_file_sha256'] and sha(sp)==audit['result_file_sha256']):
                    raise ValueError('Missing or stale actual-token decode audit: '+key)
                if request['finish_reason']!='stop' or request['stop_reason']!='FINAL:' or request['generated_tokens']>=8192:
                    raise ValueError('Valid result conflicts with existing stop rule: '+key)
                if not all(math.isfinite(result['scores'][k]) for k in ['bare_margin','union_margin']):
                    raise ValueError('Nonfinite score in purportedly valid context')
            row={'judge':judge,'task_id':info['task_id'],'template':info['template'],'k':info['k'],
                 'request_observed':request is not None,'generated_tokens':request['generated_tokens'] if request else None,
                 'failure_reason':None if valid else ('request_missing' if request is None else (result or {}).get('failure','result_missing_or_invalid')),
                 **classify(request['returned_text'] if request else None,valid)}
            if valid:
                assert row['visible_prefix_observed']==audit['visible_prefix']
                assert row['pre_marker_characters']==audit['pre_marker_characters']
            rows.append(row)
    assert len(rows)==216
    tasks=[]
    for judge in ['qwen','mistral']:
        for template in ['original','explicit']:
            for task in dict.fromkeys(r['task_id'] for r in inputs):
                rs=[r for r in rows if (r['judge'],r['template'],r['task_id'])==(judge,template,task)]
                assert {r['k'] for r in rs}==set(range(9))
                tasks.append({'judge':judge,'template':template,'task_id':task,**aggregate(rs)})
    groups=[]
    for judge in ['qwen','mistral']:
        for template in ['original','explicit']:
            rs=[r for r in rows if (r['judge'],r['template'])==(judge,template)]
            groups.append({'judge':judge,'template':template,'independent_tasks':len({r['task_id'] for r in rs}),**aggregate(rs)})
    save('results/development_behavior_report.json', {'scope':'Reanalysis of already completed excluded development smoke; not a new experiment', 'old_smoke_diagnostic_was_post_hoc':True,'formal_rules_now_predefined_before_formal_samples':True,'development_inferential_tests':False,'groups':groups,'task_rows':tasks,'request_rows':rows,'new_inference_requests':0})
    print(json.dumps(groups,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
