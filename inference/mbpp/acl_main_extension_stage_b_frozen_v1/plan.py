"""Side-effect-free Stage B validation: standard library and pure scoring only."""
import argparse
import collections
import sys
from common import *
sys.path.insert(0,str(ROOT/'runtime'))
from scoring_core import prefix_requests

FIXED={'qwen':('Qwen/Qwen2.5-14B-Instruct','cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8'),
 'mistral':('mistralai/Mistral-7B-Instruct-v0.3','c170c708c41dac9275d15a8fff4eca08d52bab71')}

def require(value,message):
    if not value:raise RuntimeError(message)

def validate_content(root=ROOT,require_frozen=True):
    root=Path(root)
    if require_frozen:
        from gates import verify_bundle
        c=verify_bundle(root)
    else:c=read(root/'configs/experiment.json')
    source=read(root/'audit/source_bindings.json')
    for rel,expected in source['copied_unchanged'].items():require(sha(root/rel)==expected,'Inherited content changed: '+rel)
    tasks=read(root/'data/tasks.json');inputs=read(root/'data/truth_inputs.json');bindings=read(root/'data/candidate_truth_bindings.json')
    order=c['task_ids'];require(len(order)==len(set(order))==276,'Independent task count')
    for rows in [tasks,inputs,bindings['bindings']]:require([r['task_id'] for r in rows]==order,'Frozen task order differs')
    require(bindings['task_order']==order and bindings['all_truth_reliable'] is True,'Truth reliability')
    partial=[];zs=collections.Counter()
    for task,inp,b in zip(tasks,inputs,bindings['bindings']):
        require(len(inp['inputs_repr'])==len(inp['expected_repr'])==len(b['pass_bits'])==8,'Eight tests required')
        require(all(type(x) is bool for x in b['pass_bits']) and type(b['z']) is int and b['z']==sum(b['pass_bits']),'Invalid truth bits')
        require(task['inputs_repr']==inp['inputs_repr'] and task['expected_repr']==inp['expected_repr'],'Test selection changed')
        require(hashlib.sha256(inp['candidate_code'].encode()).hexdigest()==inp['candidate_sha256']==b['candidate_sha256'],'Candidate binding changed')
        zs[b['z']]+=1
        if 1<=b['z']<=7:partial.append(b['task_id'])
    require(c['primary_task_ids']==partial and len(partial)==33 and zs[0]==144 and zs[8]==99,'Realized population changed')
    require(not set(order)&{'Mbpp/'+str(i) for i in [410,4,763,222,568,425]},'Development task contamination')
    users=read(root/'data/judge_inputs.json');u={r['condition_id']:r for r in users}
    require(len(u)==len(users)==10488,'Shared condition count')
    planned={(t,template,kind,k) for t in order for template in ['original','explicit'] for kind in ['same_engine_immediate','corrected_RTS_v2'] for k in range(9)}
    planned|={(t,template,'count',None) for t in order for template in ['original','explicit']}
    require({(r['task_id'],r['template'],r['kind'],r.get('k')) for r in users}==planned,'Shared factorial coverage')
    require(all(not ({'z','pass_bits'}&r.keys()) for r in users),'Execution truth leaked into judge payload')
    summary={}
    for judge,(model,revision) in FIXED.items():
        j=c['judges'][judge];native=read(root/c['native_inputs'][judge]);rows=native['rows']
        require(j['model_id']==native['model_id']==model and j['revision']==native['revision']==revision,'Fixed judge revision')
        require(native['shared_inputs_sha256']==sha(root/'data/judge_inputs.json'),'Native shared input binding')
        require(len(rows)==10488 and {r['condition_id'] for r in rows}==set(u),'Native condition coverage')
        require(native['native_chat_template_sha256']==j['native_chat_template_sha256'],'Native chat template')
        nodes=0;counts=collections.Counter()
        for r in rows:
            user=u[r['condition_id']];kind=r['kind'];counts[kind]+=1
            require(all(r.get(k)==user.get(k) for k in ['task_id','kind','template','k']),'Condition metadata drift')
            tokens=r['prompt_token_ids'];require(tokens and all(type(x) is int and x>=0 for x in tokens),'Invalid native token IDs')
            require(len(tokens)+8192+32<32768,'Context overflow')
            seed_text='|'.join(['20260914',c['study_id'],judge,kind,r['template'],r['task_id'],str(r['k']) if r.get('k') is not None else 'NA'])
            expected=int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:8],'big')%2147483647
            require(r['seed']==expected,'Row seed drift')
            if kind=='same_engine_immediate':
                paths={k:tuple(v) for k,v in r['answer_paths'].items()};probes=prefix_requests(tokens,paths)
                require(set(paths)=={'A','B',' A',' B','\nA','\nB'} and len(probes)==2,'Frozen immediate path trie')
                nodes+=len(probes)
        require(dict(counts)=={'same_engine_immediate':4968,'corrected_RTS_v2':4968,'count':552},'Per-judge counts')
        summary[judge]={'revision':revision,'conditions':len(rows),'by_kind':dict(counts),'max_input_tokens':max(len(r['prompt_token_ids']) for r in rows),'immediate_prefix_probes':nodes}
    chunks=read(root/c['resources']['chunks']);require(len(chunks)==24 and len({r['chunk_id'] for r in chunks})==24,'Fixed chunks')
    for judge in FIXED:
        selected=[r for r in chunks if r['judge']==judge]
        require([t for r in selected for t in r['task_ids']]==order and all(len(r['task_ids'])==23 for r in selected),'Chunk order/coverage')
    require(all(r['GPU_count']==1 and r['walltime']=='08:00:00' and r['reserved_GPU_hours']==8 for r in chunks),'Per-slot resources')
    require(c['resources']['proposed_total_physical_GPU_hours_cap']==192 and c['resources']['extra_slots']==0,'Global budget')
    require(c['analysis']['bootstrap_replicates']==9999 and c['analysis']['seed']==20260914 and c['parity']['task_id']==order[0],'Analysis/parity settings')
    require(all(v is False for v in c['authorization'].values()),'Frozen config must not embed live authorization')
    return {'mode':'CPU_static_validation','Stage_B_frozen':c['freeze']['Stage_B'],'tasks':276,'partial':33,'z0':144,'z8':99,
      'core_contexts':19872,'RTS_generations_planned':9936,'count_generations_planned':1104,'conditions_total':20976,'judges':summary,
      'maximum_submission_slots':24,'maximum_GPU_hours':192,'model_modules_imported':[m for m in ['torch','vllm','transformers','tokenizers'] if m in sys.modules],
      'model_loaded':False,'network_calls':0,'GPU_submissions':0,'new_model_requests':0,'candidate_code_executed':False}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['validate','dry-run','run','generate'],nargs='?',default='validate');a=p.parse_args()
    if a.action in ['run','generate']:raise PermissionError('This entry point never loads models or generates candidates; explicit authorized judge submission uses control.py')
    print(json.dumps(validate_content(),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
