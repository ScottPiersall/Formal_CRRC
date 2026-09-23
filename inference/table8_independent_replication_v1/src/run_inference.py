"""Gated offline vLLM runner. Never reads truth, old scores or candidate executors.

Requires a separately verified CLUSTER submission receipt and finite allocation.
No scheduler submission, credential access, download or corrective model call.
"""
import argparse, importlib.metadata, os, platform, signal, sys, time, traceback
from protocol import *
sys.dont_write_bytecode=True
sys.path.insert(0,str(OUT/'preregistration/reused_source'))
from formalcrrc import code_q3_repair as shared
STOP=False
def stop(*_):
 global STOP;STOP=True
def append(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a',encoding='utf-8') as h:h.write(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n');h.flush();os.fsync(h.fileno())
def verify_freeze():
 p=OUT/'preregistration/freeze.json';freeze=load(p)
 for rel,h in freeze['files'].items():assert sha((OUT/rel).read_bytes())==h,rel
 return sha(p.read_bytes())
def preflight():
 from transformers import AutoTokenizer,AutoConfig
 freeze_hash=verify_freeze();model=load(OUT/'preregistration/model.json');old=model['old_model_inventory'];snap=pathlib.Path(old['snapshot'])
 assert snap.is_dir() and snap.name==REVISION
 for name,meta in old['files'].items():assert sha((snap/name).read_bytes())==meta['sha256'],name
 for shard in old['shards']:
  p=snap/shard['name'];assert p.is_file() and p.stat().st_size==shard['size'] and p.resolve().name==shard['blob']
 cfg=AutoConfig.from_pretrained(MODEL,revision=REVISION,cache_dir=str(snap.parents[2]),local_files_only=True,trust_remote_code=False)
 assert cfg._commit_hash==REVISION and not getattr(cfg,'quantization_config',None)
 tok=AutoTokenizer.from_pretrained(str(snap),local_files_only=True,trust_remote_code=False)
 assert sha(tok.chat_template)==old['chat_template_sha256']
 inputs=rows(OUT/'preregistration/inference_inputs.jsonl')
 for r in inputs:
  native=tok.apply_chat_template([dict(role='user',content=r['user_message'])],tokenize=False,add_generation_prompt=True)
  assert native==r['rendered_prompt'] and tok.encode(native,add_special_tokens=False)==r['prompt_token_ids']
 versions={k:importlib.metadata.version(k) for k in ['torch','transformers','vllm','tokenizers','numpy']}
 assert all(versions[k]==v for k,v in {'torch':'2.13.0+cu130','transformers':'5.16.1','vllm':'0.28.0','tokenizers':'0.23.2'}.items())
 import vllm
 sampler=pathlib.Path(vllm.__file__).parent/'v1/sample/sampler.py';s=sampler.read_text()
 assert 'log_softmax(dim=-1, dtype=torch.float32)' in s and 'gather_specific_token_logprobs' in s and 'raw_logprobs' in s
 write(OUT/'preregistration/remote_preflight.json',dict(checked_at=now(),freeze_sha256=freeze_hash,status='passed',rows=len(inputs),
  snapshot=str(snap),versions=versions,sampler_sha256=sha(sampler.read_bytes()),hostname=platform.node(),model_weights_verified='Historical shard sizes/blob IDs plus pinned metadata; no new weight download'),True)
 print('REMOTE CPU PREFLIGHT PASSED',flush=True)

class Runner:
 def __init__(self,stage):
  self.freeze=verify_freeze();attest=load(OUT/'preregistration/remote_preflight.json');assert attest['freeze_sha256']==self.freeze
  self.job=os.environ['SLURM_JOB_ID'];gate=load(OUT/'costs/gates'/(self.job+'.json'));self.deadline=float(os.environ['FOLLOWUP_DEADLINE_EPOCH'])
  assert gate['job_id']==self.job and gate['stage'] in (stage,'combined') and gate['quota_gate_passed'] and gate['project_budget_gate_passed']
  assert gate['reserved_physical_gpu_seconds']<=14400 and 0<self.deadline-time.time()<=gate['reserved_physical_gpu_seconds']
  assert gate['account_remaining_basis'] and gate['evidence_files']
  for rel,h in gate['evidence_files'].items():assert sha((OUT/rel).read_bytes())==h
  if stage=='main':assert load(OUT/'preregistration/cpu_engineering_gate.json')['passed']
  from transformers import AutoTokenizer
  from vllm import LLM,SamplingParams
  import torch
  assert torch.cuda.is_available() and torch.cuda.is_bf16_supported() and torch.cuda.device_count()==1
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
  snap=load(OUT/'preregistration/model.json')['old_model_inventory']['snapshot'];self.tok=AutoTokenizer.from_pretrained(snap,local_files_only=True,trust_remote_code=False)
  self.params=SamplingParams;self.stage=stage;self.log=OUT/'costs/runtime'/(self.job+'.jsonl');self.active=None
  start=time.monotonic();cfg=load(OUT/'preregistration/generation_config.json')['engine']
  self.llm=LLM(model=snap,tokenizer=snap,trust_remote_code=False,**cfg);load_seconds=time.monotonic()-start
  write(OUT/'costs/runtime'/(self.job+'.json'),dict(job_id=self.job,stage=stage,created_at=now(),load_seconds=load_seconds,
   gpu=torch.cuda.get_device_name(),driver_capture='See scheduler job stdout nvidia-smi',cuda=torch.version.cuda,
   versions=attest['versions'],engine=cfg,deadline_epoch=self.deadline,freeze_sha256=self.freeze),True)
  original=self.llm.llm_engine.step
  def observe(*a,**kw):
   outputs=original(*a,**kw)
   for output in outputs:
    if getattr(output,'finished',False) and self.active is not None:self.save(output)
   return outputs
  self.llm.llm_engine.step=observe
 def remaining(self):return self.deadline-time.time()
 def save(self,output):
  r,dest,prefix,needed,started=self.active
  if dest.exists():return
  c=output.outputs[0];ids=list(c.token_ids)
  assert list(output.prompt_token_ids)==r['prompt_token_ids']+(prefix or [])
  result=dict(condition_id=r['condition_id'],seed=r['seed'],job_id=self.job,freeze_sha256=self.freeze,created_at=now(),
   prompt_ids_sha256=sha(canonical(r['prompt_token_ids'])),raw_generated_token_ids=ids,raw_returned_text=c.text,finish_reason=c.finish_reason,stop_reason=c.stop_reason,
   elapsed_seconds=time.monotonic()-started,generated_tokens=len(ids),status='ok')
  if needed is not None:
   result.update(prefix=prefix,next_log_probabilities={str(i):float(c.logprobs[0][i].logprob) for i in needed},probability_read_context_token_ids=r['prompt_token_ids']+(prefix or []),probability_read_context_text=self.tok.decode(r['prompt_token_ids']+(prefix or []),skip_special_tokens=False,clean_up_tokenization_spaces=False))
  elif r['experiment']=='A':
   context,status=marker_context(self.tok,r['prompt_token_ids'],ids,c.finish_reason,c.stop_reason)
   result.update(status=status,final_context_token_ids=context,final_context_text=self.tok.decode(context,skip_special_tokens=False,clean_up_tokenization_spaces=False) if context else None)
  else:
   eos=self.llm.llm_engine.model_config.hf_config.eos_token_id
   eos={eos} if isinstance(eos,int) else set(eos)
   clean=list(ids)
   while clean and clean[-1] in eos:clean.pop()
   text=self.tok.decode(clean,skip_special_tokens=False,clean_up_tokenization_spaces=False)
   n,status=parse_count(text,c.finish_reason,len(ids))
   if c.stop_reason is not None and c.stop_reason not in eos:n,status=None,'unexpected_stop'
   result.update(status=status,predicted_count=n,decoded_answer_text=text)
  result['raw_decoded_text']=self.tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
  if needed is None:
   if r['experiment']=='A':
    ctx=result.get('final_context_token_ids');n=len(ctx)-len(r['prompt_token_ids']) if ctx is not None else None
    result.update(accepted_generated_prefix_length=n,extra_returned_token_ids=ids[n:] if n is not None else None,v1_boundary_rule_status=result['status'],boundary_rule_version='v1_byte_identical')
  write(dest,result,True)
 def request(self,r,dest,prefix=None,needed=None):
  if dest.exists():return load(dest)
  request_key=str(dest.relative_to(OUT));history=[x for p in (OUT/'costs/runtime').glob('*.jsonl') for x in rows(p)]
  assert sum(x.get('event')=='submitted' and x.get('request_key')==request_key for x in history)<2,'Attempt limit reached'
  phase='score' if needed is not None else r['experiment'];start=time.monotonic()
  self.active=(r,dest,prefix,needed,start)
  if needed is not None:param=self.params(n=1,max_tokens=1,temperature=0,logprob_token_ids=needed,seed=r['seed'])
  else:param=self.params(n=1,max_tokens=CAP,temperature=0,top_p=1,top_k=-1,min_p=0,repetition_penalty=1,seed=r['seed'],
   stop=['FINAL:'] if r['experiment']=='A' else None,include_stop_str_in_output=True)
  ids=r['prompt_token_ids']+(prefix or [])
  append(self.log,dict(event='submitted',request_key=request_key,phase=phase,stage=self.stage,job_id=self.job,at=now(),input_tokens=len(ids)))
  try:
   output=self.llm.generate([dict(prompt_token_ids=ids)],param,use_tqdm=False)[0];self.save(output)
  except Exception:
   append(self.log,dict(event='infrastructure_failure',request_key=request_key,phase=phase,stage=self.stage,at=now(),error=traceback.format_exc(),elapsed_seconds=time.monotonic()-start));raise
  finally:self.active=None
  result=load(dest)
  append(self.log,dict(event='completed',request_key=request_key,phase=phase,stage=self.stage,at=now(),elapsed_seconds=time.monotonic()-start,generated_tokens=len(result['raw_generated_token_ids']),input_tokens=len(ids)))
  return result
 def score(self,r,t):
  dest=OUT/'data/scored'/(r['condition_id']+'.json')
  if dest.exists():return
  try:
   ctx=t['final_context_token_ids'];paths,forms=answer_paths(self.tok,t['final_context_text'],ctx);trie=shared.trie(paths);nodes={};seconds=0.;eos_probs={}
   eos=self.llm.llm_engine.model_config.hf_config.eos_token_id;eos=[eos] if isinstance(eos,int) else list(eos)
   for prefix,needed in trie.items():
    tag='root' if not prefix else '_'.join(map(str,prefix));p=OUT/'traces/prefix_nodes'/r['condition_id']/(tag+'.json')
    requested=sorted(set(needed+eos)) if not prefix else needed
    sr=dict(r,prompt_token_ids=ctx);n=self.request(sr,p,list(prefix),requested)
    nodes[prefix]={i:n['next_log_probabilities'][str(i)] for i in needed};seconds+=n['elapsed_seconds']
    if not prefix:eos_probs={str(i):math.exp(n['next_log_probabilities'][str(i)]) for i in eos}
   scores,events=shared.combine(forms,nodes)
   result=dict(status='ok',scores=scores,answer_events=events,prefix_nodes=[dict(token_ids=list(k),next_log_probabilities=v) for k,v in nodes.items()],scoring_seconds=seconds,extra_scoring_generated_tokens=len(nodes),eos_probabilities=eos_probs,eos_probability=sum(eos_probs.values()),bare_answer_mass=scores['bare']['total_probability_mass'],union_answer_mass=scores['whitespace_union']['total_probability_mass'],final_context_ids_sha256=sha(canonical(ctx)),scoring_generated_tokens_excluded_from_context=True)
  except (ValueError,AssertionError) as e:result=dict(status='score_validation_failure',error=str(e))
  write(dest,dict(condition_id=r['condition_id'],trace_sha256=sha((OUT/'traces'/(r['condition_id']+'.json')).read_bytes()),**result),True)
 def run(self):
  todo=[r for r in rows(OUT/'preregistration/inference_inputs.jsonl') if r['split']==self.stage]
  for r in todo:
   if STOP or self.remaining()<600:break
   dest=OUT/'traces'/(r['condition_id']+'.json')
   if r['experiment']=='I':
    if not dest.exists():
     write(dest,dict(condition_id=r['condition_id'],seed=r['seed'],job_id=self.job,freeze_sha256=self.freeze,created_at=now(),prompt_ids_sha256=sha(canonical(r['prompt_token_ids'])),status='ok',raw_generated_token_ids=[],raw_returned_text='',raw_decoded_text='',finish_reason=None,stop_reason=None,generated_tokens=0,elapsed_seconds=0.,final_context_token_ids=r['prompt_token_ids'],final_context_text=r['rendered_prompt'],immediate_no_trajectory=True),True)
    t=load(dest)
   else:t=self.request(r,dest)
   if t['status']=='ok':self.score(r,t)
   print(r['condition_id'],t['status'],t['generated_tokens'],round(self.remaining()),flush=True)
  write(OUT/'costs/runtime'/(self.job+'_finished.json'),dict(finished_at=now(),stage=self.stage,job_id=self.job,remaining_seconds=self.remaining()),True)

def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','smoke','main']);a=p.parse_args()
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGUSR1,stop)
 if a.stage=='preflight':preflight()
 else:Runner(a.stage).run()
if __name__=='__main__':main()
