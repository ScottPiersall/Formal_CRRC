"""Pure protocol orchestration with an injected backend; no GPU imports."""
import math
import sys
from common import *
sys.path.insert(0,str(ROOT/'runtime'))
from scoring_core import exact_answer_paths,prefix_requests,derive_margins,rts_context,parse_count,ProtocolFailure
from ledger import Checkpoint

def load_rows(root,judge,chunk=None):
    root=Path(root);users={r['condition_id']:r for r in read(root/'data/judge_inputs.json')}
    native=read(root/'data/native'/(judge+'.json'))
    truths={r['task_id']:r for r in read(root/'data/candidate_truth_bindings.json')['bindings']}
    selected=set(chunk['task_ids']) if chunk else None;rows=[]
    for row in native['rows']:
        if selected is not None and row['task_id'] not in selected:continue
        u=users[row['condition_id']];t=truths[row['task_id']]
        binding={'judge':judge,'model_revision':native['revision'],'condition_id':row['condition_id'],'task_id':row['task_id'],
           'candidate_sha256':t['candidate_sha256'],'truth_receipt_sha256':t['truth_receipt_sha256'],'native_row_sha256':digest(row),
           'user_message_sha256':digest(u['user_message'])}
        # Count has no threshold in its immutable source manifest. Normalize only
        # the runtime metadata; do not introduce k into the model prompt.
        normalized=dict(row);normalized.setdefault('k',None)
        rows.append(dict(normalized,user_message=u['user_message'],binding=binding,result_id=judge+'_'+row['condition_id']))
    return rows

def validate_scored_result(row,requests):
    """Independently recompute stored margins from referenced conditional logits."""
    if row['kind']=='count' or row['status']!='ok':return
    paths={k:tuple(v) for k,v in row['answer_paths'].items()};context=row['context_token_ids'];nodes={}
    for ref in row['score_request_refs']:
        record=requests[ref['path']];spec=record['spec'];response=record['response'];prefix=tuple(spec['suffix_prefix'])
        if spec['context_token_ids']!=context+list(prefix):raise RuntimeError('Score probe context drift')
        nodes[prefix]={int(k):v for k,v in response['conditional_logprobs'].items()}
    expected=prefix_requests(context,paths)
    if len(nodes)!=len(row['score_request_refs']) or set(nodes)!={x['suffix_prefix'] for x in expected}:raise RuntimeError('Score prefix coverage differs')
    for node,ref in zip(expected,row['score_request_refs']):
        spec=requests[ref['path']]['spec']
        if spec['needed_token_ids']!=list(node['requested_token_ids']) or spec['suffix_prefix']!=list(node['suffix_prefix']):raise RuntimeError('Score path request changed')
    derived=derive_margins(paths,nodes)
    saved=row['scores']
    # Windows/Python 3.14 and CLUSTER/Linux libm can differ at the last bits of
    # logsumexp. This is an audit tolerance only, never a cutoff/tie epsilon.
    if derived['bare_margin']!=saved['bare_margin'] or derived['sequence_logprobs']!=saved['sequence_logprobs'] or not math.isclose(derived['union_margin'],saved['union_margin'],rel_tol=0,abs_tol=1e-12):raise RuntimeError('Scores disagree with raw conditional likelihoods')

class Runner:
    def __init__(self,config,judge,backend,store):
        self.config=config;self.judge=judge;self.backend=backend;self.store=store
    def run_one(self,row):
        found=self.store.completed(row['result_id'],row['binding'])
        if found is not None:return found
        refs=[];ids=row['prompt_token_ids'];kind=row['kind']
        result={k:row[k] for k in ['result_id','condition_id','task_id','template','kind','k','binding']}
        result.update(judge=self.judge,stage_B_sha256=self.store.stage_hash,request_refs=refs,status='pending')
        def request(suffix,mode,context,params,**extra):
            spec={'request_id':row['result_id']+'_'+suffix,'result_id':row['result_id'],'binding':row['binding'],
                'mode':mode,'kind':kind,'context_token_ids':list(context),'params':params,'seed':row['seed'],**extra}
            response,ref=self.store.request(spec,self.backend.invoke);refs.append(ref);return response,ref
        try:
            if kind=='same_engine_immediate':context=ids
            else:
                params={'n':1,'max_tokens':8192,'temperature':0,'top_p':1,'top_k':-1,'min_p':0,'repetition_penalty':1,'seed':row['seed'],
                   'stop':['FINAL:'] if kind=='corrected_RTS_v2' else None,'include_stop_str_in_output':True}
                generated,trace_ref=request('generation','generation',ids,params)
                result['generation_ref']=trace_ref
                if kind=='count':
                    clean=list(generated['generated_token_ids'])
                    if clean and clean[-1] in self.backend.eos:clean.pop()
                    decoded=self.backend.decode(clean)
                    result['count_text']=decoded
                    result.update(parse_count(decoded,generated['generated_tokens'],generated['finish_reason'],generated['stop_reason'],eos_token_ids=self.backend.eos))
                else:
                    context=list(rts_context(ids,generated['generated_token_ids'],self.backend.decode,generated['finish_reason'],generated['stop_reason']))
                    n=len(context)-len(ids);before=self.backend.decode(ids);accepted=self.backend.decode(context)[len(before):]
                    if not accepted.endswith('FINAL:'):raise RuntimeError('Accepted context does not end at marker')
                    prefix=accepted[:-len('FINAL:')]
                    result['rts']={'valid_boundary':True,'accepted_generated_prefix_length':n,'accepted_trace_text':accepted,
                       'visible_prefix':bool(prefix.strip()),'marker_only':not bool(prefix.strip()),'prefix_characters':len(prefix),'prefix_whitespace_words':len(prefix.split()),
                       'extra_returned_token_ids':generated['generated_token_ids'][n:]}
            if kind!='count':
                paths=exact_answer_paths(context,self.backend.encode,self.backend.decode)
                if kind=='same_engine_immediate' and row.get('answer_paths')!={k:list(v) for k,v in paths.items()}:raise RuntimeError('Frozen immediate answer paths differ')
                nodes={};score_refs=[]
                for i,node in enumerate(prefix_requests(context,paths)):
                    needed=list(node['requested_token_ids']);params={'n':1,'max_tokens':1,'temperature':0,'seed':row['seed'],'logprob_token_ids':needed}
                    response,ref=request('prefix_'+str(i),'score',node['context_ids'],params,suffix_prefix=list(node['suffix_prefix']),needed_token_ids=needed)
                    nodes[node['suffix_prefix']]={int(k):v for k,v in response.get('conditional_logprobs',{}).items()};score_refs.append(ref)
                scores=derive_margins(paths,nodes)
                result.update(scores=scores,context_token_ids=list(context),answer_paths={k:list(v) for k,v in paths.items()},score_request_refs=score_refs)
                parity=[]
                if row['task_id']==self.config['parity']['task_id'] and row['k']==0:
                    for i,(form,path) in enumerate(paths.items()):
                        params={'n':1,'max_tokens':1,'temperature':0,'seed':row['seed'],'prompt_logprobs':1}
                        response,ref=request('teacher_'+str(i),'parity',tuple(context)+path,params)
                        values=response['teacher_forced_actual_token_logprobs'][len(context):]
                        if not all(v is not None and math.isfinite(v) for v in values):raise RuntimeError('Invalid teacher-forced parity values')
                        teacher=sum(values);delta=abs(teacher-scores['sequence_logprobs'][form]);limit=self.config['parity']['max_abs_sequence_logprob_difference']
                        parity.append({'form':form,'teacher_logprob':teacher,'sequence_logprob':scores['sequence_logprobs'][form],'absolute_difference':delta,'tolerance':limit,'passed':delta<=limit,'request_ref':ref})
                    result['parity']=parity
                    if not all(x['passed'] for x in parity):
                        save(self.store.work/'parity_failures'/(row['result_id']+'.json'),result)
                        raise RuntimeError('Engineering parity failure; stop without replacing trace or scores')
            result['status']='ok'
        except ProtocolFailure as exc:
            result.update(status='terminal_protocol_failure',reason=str(exc))
        result['completed_utc']=now();self.store.finish(result);return result
    def run(self,rows,progress=lambda x:None):
        for row in rows:
            result=self.run_one(row);progress(result)
