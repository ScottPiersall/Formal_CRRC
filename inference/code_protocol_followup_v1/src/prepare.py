"""Read-only audit of v2 dependencies, then isolate and freeze model inputs."""
import collections, difflib, importlib.metadata, json, pathlib, shutil, subprocess, sys
from protocol import *
ROOT=OUT.parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'src'))
from formalcrrc import code_extension as cp,code_five as f,code_q3_repair as shared
from formalcrrc.code_extension_answer_tokens import BackendTokenizer
OLD=ROOT/'artifacts/code_extension_v2'
def copy(p,dest):
 dest.parent.mkdir(parents=True,exist_ok=True)
 raw=p.read_bytes()
 if dest.exists(): assert dest.read_bytes()==raw,str(dest)
 else: dest.write_bytes(raw)
def main():
 if (OUT/'preregistration/freeze.json').exists():
  freeze=load(OUT/'preregistration/freeze.json')
  for p,h in freeze['files'].items():assert sha((OUT/p).read_bytes())==h,p
  print('ALREADY FROZEN: local files verified; remote gates remain required');return
 dependencies={}; originals={}
 def dep(p):
  h=sha(p.read_bytes()); dependencies[p.relative_to(ROOT).as_posix()]=h;return h
 git=subprocess.run(['git','status','--short'],cwd=ROOT,capture_output=True,text=True,check=True).stdout
 write(OUT/'preregistration/git_status_before.txt',git,True)
 write(OUT/'preregistration/git_head.txt',subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True),True)
 inv=load(OLD/'model_inventory.json');q=inv['models']['qwen']
 assert q['revision']==q['resolved_config_commit']==REVISION and q['model_id']==MODEL
 inputs=load(OLD/'prompts/qwen.json');assert inputs['revision']==REVISION
 items=load(OLD/'instances.json');assert collections.Counter(x['split'] for x in items)=={'main':80,'smoke':3}
 assert len({i['task_id'] for i in items})==83
 old_index={(r['task_id'],r['variant'],r['k']):r for r in inputs['rows']};assert len(old_index)==1494
 freeze_b=load(OLD/'freeze_B.json')['files']; freeze_checks=[]
 for p in [OLD/'instances.json',OLD/'model_inventory.json',OLD/'pilot_protocol.json',OLD/'freeze_A.json',OLD/'freeze_B.json',OLD/'prompts/qwen.json',OLD/'prompts/qwen__answer_tokens.json',OLD/'answer_form_protocol.json']:
  h=dep(p);copy(p,OUT/'preregistration/old_metadata'/p.relative_to(OLD))
 # Preserve the package's original import closure as source only, no old datasets.
 for p in sorted((ROOT/'src/formalcrrc').glob('*.py')):
  dep(p);copy(p,OUT/'preregistration/reused_source/formalcrrc'/p.name)
 tokfile=OLD/'upstream/tokenizers/qwen.json';assert dep(tokfile)==q['files']['tokenizer.json']['sha256']
 copy(tokfile,OUT/'preregistration/tokenizer/tokenizer.json')
 cfgpath=ROOT/'.remote/label_swap_tokenizers/models--Qwen--Qwen2.5-14B-Instruct/snapshots'/REVISION/'tokenizer_config.json'
 assert dep(cfgpath)==q['files']['tokenizer_config.json']['sha256']
 cfg=load(cfgpath);assert sha(cfg['chat_template'])==q['chat_template_sha256']==inputs['chat_template_sha256']
 copy(cfgpath,OUT/'preregistration/tokenizer/tokenizer_config.json');write(OUT/'preregistration/chat_template.jinja',cfg['chat_template'],True)
 tok=BackendTokenizer(tokfile);prepared=[];truth=[];baseline=[];diffs=[];wrappers=set();normalized=set();path_checks=0
 for item in items:
  task=item['task_id'];slug=task.replace('/','_');split=item['split'];assert len(item['tests'])==8
  gp=OLD/'generation'/(slug+'.json');tp=OLD/'truth'/(slug+'.json');g=load(gp);t=load(tp)
  assert g['extraction_status']=='ok' and g['model_id']=='mistralai/Mistral-7B-Instruct-v0.3'
  assert sha(g['code'])==g['code_sha256']==t['code_sha256']
  assert t['stable'] and len(t['pass_bits'])==8 and sum(t['pass_bits'])==t['z'] and t['split']==split
  assert t['execution']['status']=='complete'
  for repeat in (0,1):
   tests=[x for x in t['execution']['tests'] if x['repeat']==repeat]
   assert len(tests)==8 and [int(x['passed']) for x in sorted(tests,key=lambda x:x['test_index'])]==t['pass_bits']
  for p in [gp,tp]:
   h=dep(p);rel=p.relative_to(ROOT).as_posix();assert freeze_b[rel]==h,rel
   freeze_checks.append(rel);copy(p,OUT/'preregistration/old_evidence'/p.relative_to(OLD))
  truth.append(dict(task_id=task,split=split,z=t['z'],pass_bits=t['pass_bits'],candidate_sha256=g['code_sha256'],truth_source=tp.relative_to(ROOT).as_posix(),truth_sha256=dep(tp)))
  for variant in ('original','explicit'):
   normalized_group=[];count_versions=[]
   for k in range(9):
    r=old_index[task,variant,k];assert r['split']==split
    regenerated=cp.prompt(item,g['code'],k,variant)
    assert regenerated['user_message']==r['user_message'] and regenerated['threshold_span']==r['threshold_span']
    u=r['user_message'];rendered=r['rendered_prompt'];assert sha(rendered)==r['rendered_prompt_sha256']
    assert rendered.count(u)==1
    left,right=rendered.split(u);wrappers.add((left,right))
    assert len(tok.encode(rendered))==r['n_prompt_tokens']
    u_a=u+A_SUFFIX;ra=left+u_a+right;ids=tok.encode(ra)
    assert len(ids)+CAP+4<=32768
    a,b=r['threshold_span'];raw=u_a.encode();normalized_group.append(raw[:a]+b'{threshold}'+raw[b:]);assert raw[a:b]==str(k).encode()
    condition=key('A',split,variant,task,k)
    prepared.append(dict(condition_id=condition,experiment='A',split=split,task_id=task,variant=variant,k=k,seed=seed(condition),
     user_message=u_a,user_message_sha256=sha(u_a),rendered_prompt=ra,rendered_prompt_sha256=sha(ra),prompt_token_ids=ids,
     threshold_span=[a,b],normalized_sha256=sha(normalized_group[-1]),old_rendered_prompt_sha256=sha(rendered)))
    paths,forms=answer_paths(tok,ra+'Tokenizer-only marker probe.\nFINAL:',tok.encode(ra+'Tokenizer-only marker probe.\nFINAL:'))
    shared.trie(paths);path_checks+=1
    count_versions.append(count_prompt(u,variant))
    for representation,folder in [('bare','scores'),('whitespace_union','scores_answer_forms')]:
     p=OLD/folder/split/'qwen'/variant
     if representation=='bare':p=p/'bfloat16'
     p=p/(cp.score_key(task,k)+'.json');s=load(p)
     assert all(s[field]==expected for field,expected in [('model_id',MODEL),('revision',REVISION),('dtype','bfloat16'),('task_id',task),('variant',variant),('k',k),('rendered_prompt_sha256',sha(rendered))])
     assert abs(s['margin']-(s['score_met']-s['score_not_met']))<1e-10
     if representation=='bare':
      assert s['label_tokenization']['met_token_ids']==[32] and s['label_tokenization']['not_met_token_ids']==[33]
      assert abs(s['margin']-s['logprob_margin'])<1e-5
     else:
      nodes={tuple(n['token_ids']):{int(k):v for k,v in n['next_log_probabilities'].items()} for n in s['prefix_nodes']}
      # Adapt only metadata schema; conditional path values are reused unchanged.
      forms={name:dict(token_ids=e['token_ids']) for name,e in s['answer_events'].items()};combined,_=shared.combine(forms,nodes)
      assert abs(combined['whitespace_union']['margin']-s['margin'])<1e-9
     source_hash=dep(p);copy(p,OUT/'preregistration/old_scores'/p.relative_to(OLD))
     baseline.append(dict(experiment='immediate',task_id=task,split=split,variant=variant,k=k,representation=representation,
      margin=s['margin'],p_met=s['p_met'],score_met=s['score_met'],score_not_met=s['score_not_met'],
      source=p.relative_to(ROOT).as_posix(),source_sha256=source_hash,rendered_prompt_sha256=sha(rendered),dtype=s['dtype'],revision=s['revision']))
   assert len(set(normalized_group))==1 and len(set(count_versions))==1
   u_b=count_versions[0];rb=left+u_b+right;condition=key('B',split,variant,task);ids=tok.encode(rb)
   assert len(ids)+CAP+4<=32768
   prepared.append(dict(condition_id=condition,experiment='B',split=split,task_id=task,variant=variant,seed=seed(condition),
    user_message=u_b,user_message_sha256=sha(u_b),rendered_prompt=rb,rendered_prompt_sha256=sha(rb),prompt_token_ids=ids))
   diff=''.join(difflib.unified_diff(old_index[task,variant,0]['user_message'].splitlines(True),u_b.splitlines(True),fromfile='old/'+variant+'/k0',tofile='count/'+variant))
   write(OUT/'preregistration/prompt_diffs'/variant/(slug+'.diff'),diff,True)
 assert len(wrappers)==1
 main=[t for t in truth if t['split']=='main'];assert (len(main),sum(t['z']<8 for t in main),sum(1<=t['z']<=7 for t in main),sum(t['z']==8 for t in main))==(80,54,23,26)
 prepared.sort(key=lambda r:sha(f'{SEED}|{r["condition_id"]}'))
 allowed={'condition_id','experiment','split','task_id','variant','k','seed','user_message','user_message_sha256','rendered_prompt','rendered_prompt_sha256','prompt_token_ids','threshold_span','normalized_sha256','old_rendered_prompt_sha256'}
 assert all(set(r)<=allowed for r in prepared)
 jsonl(OUT/'preregistration/inference_inputs.jsonl',prepared,True)
 jsonl(OUT/'data/truth.jsonl',truth,True);jsonl(OUT/'data/immediate_scores.jsonl',baseline,True)
 left,right=next(iter(wrappers));write(OUT/'preregistration/chat_rendering.json',dict(prefix=left,suffix=right,basis='Exact wrapper recovered and checked against all 1494 historical native rendered prompts; native AutoTokenizer roundtrip remains a required remote preflight.'),True)
 write(OUT/'preregistration/scaffolds.json',dict(A=A_SUFFIX,B=B_SUFFIX),True)
 write(OUT/'preregistration/model.json',dict(model_id=MODEL,revision=REVISION,dtype='bfloat16',quantization=None,old_model_inventory=q),True)
 write(OUT/'preregistration/generation_config.json',dict(do_sample=False,temperature=0,n=1,top_p=1,top_k=-1,min_p=0,repetition_penalty=1,max_tokens=CAP,seed=SEED,include_stop_str_in_output=True,
  engine=dict(dtype='bfloat16',quantization=None,tensor_parallel_size=1,max_model_len=32768,max_num_seqs=8,max_num_batched_tokens=512,enable_chunked_prefill=True,enable_prefix_caching=False,gpu_memory_utilization=.9,enforce_eager=False,logprobs_mode='raw_logprobs',seed=SEED)),True)
 write(OUT/'preregistration/dependencies.json',dict(files=dependencies,git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()),True)
 write(OUT/'preregistration/data_audit.json',dict(created_at=now(),main_tasks=80,smoke_tasks=3,nontrivial_tasks=54,partial_tasks=23,all_pass_tasks=26,
  z_histogram=dict(collections.Counter(t['z'] for t in main)),old_freeze_B_generation_truth_hash_matches=len(freeze_checks),
  old_bare_conditions=1494,old_union_conditions=1494,old_scores_not_regenerated=True,threshold_normalization_groups=166,count_threshold_independence_groups=166,
  shared_path_tokenizer_checks=path_checks,inference_input_fields=sorted(allowed),reference_code_excluded=True,execution_truth_excluded=True,old_judge_outputs_excluded=True,
  no_candidate_code_executed=True,max_prompt_tokens=max(len(r['prompt_token_ids']) for r in prepared)),True)
 print('PREPARED',len(prepared),'inputs; baseline',len(baseline),'score records; no new inference')
if __name__=='__main__':main()
