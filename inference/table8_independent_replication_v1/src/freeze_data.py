"""Freeze executed truth and all judge prompts before any judge inference."""
import collections,csv,io,pathlib,sys
O=pathlib.Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True
sys.path.insert(0,str(O/'preregistration/reused_source'))
from protocol import *
from formalcrrc import code_extension as cp
from formalcrrc.code_extension_answer_tokens import BackendTokenizer
from generate import verify
def main():
 verify(O)
 assert not list((O/'traces').rglob('*.json')) and not list((O/'data/scored').rglob('*.json'))
 assert not (O/'preregistration/freeze.json').exists()
 items=load(O/'instances.json');tok=BackendTokenizer(O/'preregistration/tokenizer/tokenizer.json');wrap=load(O/'preregistration/chat_rendering.json')
 inputs=[];truth=[];manifest=[];groups=collections.defaultdict(set);paths=[]
 for item in items:
  task=item['task_id'];slug=task.replace('/','_');g=load(O/'generation'/(slug+'.json'));t=load(O/'truth'/(slug+'.json'));d=load(O/'truth'/(slug+'__direct_equality.json'))
  assert t['stable'] and d['passed'] and g['code_sha256']==t['code_sha256']==sha(g['code'])
  assert len(item['tests'])==8 and sum(t['pass_bits'])==t['z']
  write(O/'candidates'/(slug+'.py'),g['code'],True);write(O/'tests'/(slug+'.json'),dict(task_id=task,tests=item['tests'],reference_code_sha256=sha(item['reference_code']),source='Frozen original reference-only plus_input construction; freshly revalidated reference and candidate twice, plus independent equality twice'),True)
  truth.append(dict(task_id=task,z=t['z'],pass_bits=t['pass_bits'],truth=[k<=t['z'] for k in range(9)],primary_population=1<=t['z']<=7,candidate_sha256=g['code_sha256'],tests_sha256=sha(canonical(item['tests'])),execution_sha256=sha((O/'truth'/(slug+'.json')).read_bytes()),direct_equality_sha256=sha((O/'truth'/(slug+'__direct_equality.json')).read_bytes())))
  manifest.append(dict(task_id=task,status='generated_and_truth_verified',problem_sha256=sha(item['problem']),candidate_sha256=g['code_sha256'],z=t['z'],primary_population=1<=t['z']<=7))
  for variant in ('original','explicit'):
   for k in range(9):
    p=cp.prompt(item,g['code'],k,variant)
    for ex in ('I','A'):
     user=p['user_message']+(A_SUFFIX if ex=='A' else '');rendered=wrap['prefix']+user+wrap['suffix'];ids=tok.encode(rendered)
     cid=key(ex,'main',variant,task,k);a,b=p['threshold_span'];norm=user.encode()[:a]+b'{threshold}'+user.encode()[b:]
     assert user.encode()[a:b]==str(k).encode();groups[task,variant,ex].add(norm)
     r=dict(condition_id=cid,experiment=ex,split='main',task_id=task,variant=variant,k=k,seed=seed(cid),user_message=user,user_message_sha256=sha(user),rendered_prompt=rendered,rendered_prompt_sha256=sha(rendered),prompt_token_ids=ids,n_prompt_tokens=len(ids),threshold_span=p['threshold_span'],normalized_sha256=sha(norm),candidate_sha256=g['code_sha256'],tests_sha256=sha(canonical(item['tests'])))
     assert len(ids)+8192+4<=32768
     ctx=ids if ex=='I' else ids+tok.encode('Reasoning.\nFINAL:')
     pp,forms=answer_paths(tok,tok.decode(ctx),ctx)
     paths.append(dict(condition_id=cid,scope='actual immediate context' if ex=='I' else 'synthetic boundary engineering probe; real RTS context paths saved during scoring',forms=forms))
     inputs.append(r)
 assert all(len(g)==1 for g in groups.values()) and len(inputs)==len(items)*36
 inputs.sort(key=lambda r:sha('20260911|'+r['condition_id']))
 jsonl(O/'preregistration/inference_inputs.jsonl',inputs,True);jsonl(O/'data/truth.jsonl',truth,True)
 write(O/'preregistration/answer_path_preflight.json',paths,True)
 write(O/'preregistration/population.json',dict(all=[t['task_id'] for t in truth],partial=[t['task_id'] for t in truth if t['primary_population']],nontrivial=[t['task_id'] for t in truth if t['z']<8],z0=[t['task_id'] for t in truth if t['z']==0],z8=[t['task_id'] for t in truth if t['z']==8]),True)
 write(O/'preregistration/prompt_manifest.json',[{k:r[k] for k in ('condition_id','rendered_prompt_sha256','user_message_sha256','seed','n_prompt_tokens','normalized_sha256')} for r in inputs],True)
 write(O/'preregistration/data_audit.json',dict(at=now(),task_n=len(items),partial_n=sum(t['primary_population'] for t in truth),judge_contexts=len(inputs),threshold_only_groups=len(groups),truth_is_not_in_judge_inputs=True,all_candidate_tests_reference_validated=True,raw_observations=len(items)*64),True)
 s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=list(manifest[0]));w.writeheader();w.writerows(manifest);write(O/'task_manifest.csv',s.getvalue())
 files={}
 for folder in ('src','preregistration','generation','truth','candidates','tests','data','runtime','attempts'):
  for p in (O/folder).rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts:files[p.relative_to(O).as_posix()]=sha(p.read_bytes())
 for name in ('instances.json','freeze_A.json','model_inventory.json','task_manifest.csv'):files[name]=sha((O/name).read_bytes())
 record=dict(study='table8_independent_replication_v1',stage=2,frozen_at=now(),identity='Actual candidates and execution truth completed; zero new judge calls before this freeze',stage1_sha256=sha((O/'preregistration/stage1_design_freeze.json').read_bytes()),task_n=len(items),partial_n=sum(t['primary_population'] for t in truth),files=files)
 f=O/'preregistration/stage2_data_freeze.json';write(f,record,True);write(O/'preregistration/freeze.json',f.read_text(),True)
 write(f.with_suffix('.json.sha256'),sha(f.read_bytes())+'  stage2_data_freeze.json\n',True)
 print('STAGE2_FROZEN',record['frozen_at'],'tasks',len(items),'partial',record['partial_n'],'contexts',len(inputs),sha(f.read_bytes()))
if __name__=='__main__':main()
