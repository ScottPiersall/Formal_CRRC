"""Generate auditable minimal modifications of the archived v1 engine."""
import pathlib,difflib
O=pathlib.Path(__file__).resolve().parents[1];V=O.with_name('code_protocol_followup_v1')
source=(V/'src/run_inference.py').read_text();s=source
s=s.replace("gate=load(OUT/'costs/submission_gate.json');self.job=os.environ['SLURM_JOB_ID'];self.deadline=float(os.environ['FOLLOWUP_DEADLINE_EPOCH'])", "self.job=os.environ['SLURM_JOB_ID'];gate=load(OUT/'costs/gates'/(self.job+'.json'));self.deadline=float(os.environ['FOLLOWUP_DEADLINE_EPOCH'])")
s=s.replace('<=360000','<=180000')
s=s.replace("  write(dest,result,True)\n def request", "  if needed is None:\n   if r['experiment']=='A':\n    ctx=result.get('final_context_token_ids');n=len(ctx)-len(r['prompt_token_ids']) if ctx is not None else None\n    result.update(accepted_generated_prefix_length=n,extra_returned_token_ids=ids[n:] if n is not None else None,v1_boundary_rule_status=result['status'],boundary_rule_version='v1_byte_identical')\n  write(dest,result,True)\n def request")
s=s.replace("   result.update(prefix=prefix,next_log_probabilities={str(i):float(c.logprobs[0][i].logprob) for i in needed})", "   result.update(prefix=prefix,next_log_probabilities={str(i):float(c.logprobs[0][i].logprob) for i in needed},probability_read_context_token_ids=r['prompt_token_ids']+(prefix or []),probability_read_context_text=self.tok.decode(r['prompt_token_ids']+(prefix or []),skip_special_tokens=False,clean_up_tokenization_spaces=False))")
start=s.index(' def score(self,r,t):');end=s.index('\ndef main():',start)
replacement=''' def score(self,r,t):
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
'''
s=s[:start]+replacement+s[end:]
(O/'src/run_inference.py').write_text(s)
(O/'preregistration/engine_v1_v2.diff').write_text(''.join(difflib.unified_diff(source.splitlines(True),s.splitlines(True),fromfile='v1/run_inference.py',tofile='v2/run_inference.py')))
for name in ('test_protocol.py','cost_gate.py'):(O/'src'/name).write_bytes((V/'src'/name).read_bytes())
s=(V/'operations/run_managed_v2.py').read_text();(O/'operations/run_managed.py').write_text(s)
s=(V/'operations/gpu_managed_v2.sbatch').read_text().replace('FormalCRRC_code_protocol_followup_v1/artifacts/code_protocol_followup_v1','FormalCRRC_code_protocol_followup_v2').replace('FormalCRRC_code_protocol_followup_v1','FormalCRRC_code_protocol_followup_v2').replace('artifacts/code_protocol_followup_v1/','').replace('run_managed_v2.py','run_managed.py').replace('fc_followup_gpu','fc_protocol_v2')
(O/'operations/gpu.sbatch').write_text(s)
s=(V/'operations/preflight.sbatch').read_text().replace('FormalCRRC_code_protocol_followup_v1/artifacts/code_protocol_followup_v1','FormalCRRC_code_protocol_followup_v2').replace('FormalCRRC_code_protocol_followup_v1','FormalCRRC_code_protocol_followup_v2').replace('artifacts/code_protocol_followup_v1/','').replace('fc_followup_cpu','fc_protocol_v2_cpu')
(O/'operations/preflight.sbatch').write_text(s)
print('Engine derived; full diff saved')
