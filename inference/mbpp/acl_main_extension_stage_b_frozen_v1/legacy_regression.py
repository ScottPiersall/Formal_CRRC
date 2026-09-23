"""Read-only replay of already completed development smoke likelihood receipts."""
import argparse
from common import *
import sys
import math
sys.path.insert(0,str(ROOT/'runtime'))
from scoring_core import prefix_requests,derive_margins,parse_count,ProtocolFailure

def replay(source):
    source=Path(source);work=source/'work';score_count=0;count_rows=0;parsed=0;hashes={};union_differences=[]
    manifest=source/'completed_package_manifest.sha256.json';verify_manifest(source,manifest)
    decoded={r['judge']+'_'+r['condition_id']:r for r in read(ROOT/'audit/CLUSTER_cpu_check.json').get('historical_counts_decoded_and_reparsed',[])}
    token_decode_pending=[];failed_parses=0
    for p in sorted((work/'judge_results').glob('*.json')):
        row=read(p);hashes[p.relative_to(source).as_posix()]=sha(p)
        if row['kind']=='count':
            raw_path=work/'requests'/(row['judge']+'_'+row['condition_id']+'_generation.json');raw=read(raw_path);hashes[raw_path.relative_to(source).as_posix()]=sha(raw_path)
            old=decoded.get(row['judge']+'_'+row['condition_id'])
            if old is not None and old['raw_sha256']!=sha(raw_path):raise RuntimeError('Historical tokenizer replay binding differs')
            try:
                result=parse_count(old['decoded_without_terminal_eos'] if old else raw['returned_text'],raw['generated_tokens'],raw['finish_reason'],raw['stop_reason'],eos_token_ids=old['eos_token_ids'] if old else [2,151645])
                if row['status']!='ok' or result['predicted_count']!=row['predicted_count'] or result['verdicts']!=row['verdicts']:raise RuntimeError('Historical count parse differs')
                parsed+=1
            except ProtocolFailure:
                if row['status']=='ok':
                    if old:raise RuntimeError('Previously decoded count no longer parses')
                    token_decode_pending.append({'judge':row['judge'],'condition_id':row['condition_id'],'reason':'Historical returned_text differs from token decoding; cached native decoder requires the disconnected CLUSTER environment. Do not strip text to manufacture agreement.','old_reported_predicted_count':row['predicted_count'],'raw_sha256':sha(raw_path)})
                else:failed_parses+=1
            count_rows+=1;continue
        if row['status']!='ok':raise RuntimeError('Unexpected historical score failure')
        paths={k:tuple(v) for k,v in row['answer_paths'].items()};nodes={};expected=prefix_requests(row['context_token_ids'],paths)
        if len(expected)!=len(row['prefix_requests']):raise RuntimeError('Historical prefix coverage')
        for spec,rid in zip(expected,row['prefix_requests']):
            raw_path=work/'requests'/(rid+'.json');raw=read(raw_path);hashes[raw_path.relative_to(source).as_posix()]=sha(raw_path)
            if raw['prompt_token_ids']!=list(spec['context_ids']):raise RuntimeError('Historical conditional context drift')
            nodes[spec['suffix_prefix']]={int(k):v for k,v in raw['conditional_logprobs'].items()}
        actual=derive_margins(paths,nodes);saved=row['scores'];delta=abs(actual['union_margin']-saved['union_margin']);union_differences.append(delta)
        if actual['bare_margin']!=saved['bare_margin'] or actual['sequence_logprobs']!=saved['sequence_logprobs'] or delta>1e-12:raise RuntimeError('Historical margin mismatch')
        score_count+=1
    parity=[read(p) for p in sorted((work/'parity').glob('*.json'))]
    if (score_count,count_rows,parsed+len(token_decode_pending),failed_parses,len(parity))!=(432,24,20,4,48) or not all(r['passed'] and r['absolute_difference']<=r['tolerance']==.02 for r in parity):raise RuntimeError('Historical regression counts differ')
    return {'created_utc':now(),'source':str(source),'source_manifest_sha256':sha(manifest),'source_manifest_files_verified':len(read(manifest)['files']),
       'scored_conditions_recomputed':score_count,'bare_and_sequence_logprobs_exact':True,'maximum_union_cross_platform_absolute_difference':max(union_differences),
       'union_audit_tolerance':1e-12,'audit_tolerance_is_not_metric_tie_epsilon':True,'count_requests_inspected':count_rows,'count_requests_reparsed':parsed+failed_parses,'parsed_counts_reverified':parsed,'parse_failures_reverified':failed_parses,'token_decode_pending':token_decode_pending,
       'existing_teacher_forced_parity_receipts_checked':len(parity),'new_teacher_forced_probes':0,'new_model_requests':0,'models_loaded':False,
       'count_replay':'Native token decoding when explicitly available; otherwise returned_text parser comparison is a limited check. Any disagreement with the original token-based count is retained as pending, never repaired by text stripping.',
       'CPU_decoder_evidence_sha256':sha(ROOT/'audit/CLUSTER_cpu_check.json'),
       'source_file_hashes':hashes}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=replay(a.source);save(a.output,result);print(json.dumps({k:v for k,v in result.items() if k!='source_file_hashes'}))
if __name__=='__main__':main()
