"""Isolated v2: copy evidence and make exact minimal scaffold replacement."""
import pathlib,sys,json,hashlib,shutil,difflib,subprocess,collections
O=pathlib.Path(__file__).resolve().parents[1];ROOT=O.parents[1];V1=O.with_name('code_protocol_followup_v1')
sys.dont_write_bytecode=True
def sha(b):return hashlib.sha256(b).hexdigest()
def put(p,b):
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_bytes()==b,str(p)
 else:p.write_bytes(b)
def cp(p,q):put(q,p.read_bytes())
def js(p,x):put(p,(json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode())
def rows(p):return [json.loads(l) for l in p.read_text().splitlines()]
def main():
 if (O/'preregistration/freeze.json').exists():raise RuntimeError('Already frozen; verify/resume, do not prepare again')
 for d in ('data','analysis','traces','manuscript','costs','preregistration/old'):(O/d).mkdir(parents=True,exist_ok=True)
 # Copy the exact reused source closure, tokenizer, configuration and original prompts.
 for folder in ('reused_source','tokenizer','verified_model_files'):
  for p in (V1/'preregistration'/folder).rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts:cp(p,O/'preregistration'/p.relative_to(V1/'preregistration'))
 for name in ('model.json','generation_config.json','chat_rendering.json','chat_template.jinja'):
  cp(V1/'preregistration'/name,O/'preregistration'/name)
 for name in ('protocol.py','run_inference.py','test_protocol.py'):
  cp(V1/'src'/name,O/'preregistration/old'/('v1_'+name))
 for name in ('truth.jsonl','immediate_scores.jsonl','A_scores.jsonl','B_counts.jsonl','B_threshold_predictions.jsonl'):
  cp(V1/'data'/name,O/'data'/('old_'+name if name!='truth.jsonl' else name))
 for name in ('freeze.json','scaffolds.json','analysis_plan.md'):
  cp(V1/'preregistration'/name,O/'preregistration/old'/('v1_'+name))
 cp(ROOT/'artifacts/code_extension_v2/prompts/qwen.json',O/'preregistration/old/immediate_prompts.json')
 deps=json.loads((V1/'preregistration/dependencies.json').read_text())['files']
 protected={}
 for rel,h in deps.items():
  p=ROOT/rel;assert sha(p.read_bytes())==h,rel;protected[rel]=h
 for p in V1.rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts:protected[p.relative_to(ROOT).as_posix()]=sha(p.read_bytes())
 for p in ROOT.rglob('ACL.tex'):protected[p.relative_to(ROOT).as_posix()]=sha(p.read_bytes())
 js(O/'preregistration/protected_sources.json',protected)
 put(O/'preregistration/git_status_before.txt',subprocess.check_output(['git','status','--short'],cwd=ROOT))
 put(O/'preregistration/git_head.txt',subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT))
 old=json.loads((V1/'preregistration/scaffolds.json').read_text())['A']
 ending='When you are ready to make the decision, output the exact marker FINAL: and stop immediately after the colon. Do not output A or B before or after that marker.'
 newending='When you are ready to make the decision, end your response with a final line in exactly one of these forms:\nFINAL: A\nFINAL: B\nUse the A/B meanings defined above. Do not write the marker FINAL: anywhere else in your response.'
 assert old.count(ending)==1;new=old.replace(ending,newending)
 put(O/'preregistration/scaffold_v1.txt',old.encode());put(O/'preregistration/scaffold_v2.txt',new.encode())
 put(O/'preregistration/scaffold.diff',''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='v1',tofile='v2')).encode())
 js(O/'preregistration/scaffold_hashes.json',{'v1':sha(old.encode()),'v2':sha(new.encode()),'minimal_replacement_verified':True})
 # Keep the v1 mathematics and boundary rules byte-identical; change only scaffold and study namespace.
 source=(V1/'src/protocol.py').read_text();assert ending in source
 put(O/'src/protocol.py',source.replace(ending,newending).replace('code_protocol_followup_v1|','code_protocol_followup_v2|').encode())
 sys.path.insert(0,str(O/'src'));sys.path.insert(0,str(O/'preregistration/reused_source'))
 from protocol import answer_paths,marker_context
 from formalcrrc.code_extension_answer_tokens import BackendTokenizer
 tok=BackendTokenizer(O/'preregistration/tokenizer/tokenizer.json')
 oldprompts=json.loads((O/'preregistration/old/immediate_prompts.json').read_text())['rows']
 v1={(r['task_id'],r['variant'],r.get('k')):r for r in rows(V1/'preregistration/inference_inputs.jsonl') if r['experiment']=='A'}
 inputs=[];groups=collections.defaultdict(set)
 for r in oldprompts:
  assert 'A' in r['user_message'] and 'B' in r['user_message']
  for ex in ('A','I'):
   original=v1[r['task_id'],r['variant'],r['k']];rr=dict(original)
   u=r['user_message']+(new if ex=='A' else '');base=r['rendered_prompt'];left,right=base.split(r['user_message'])
   rendered=left+u+right;cid=f"{ex}/{r['split']}/{r['variant']}/{r['task_id'].replace('/','_')}__k{r['k']}"
   sd=int.from_bytes(hashlib.sha256(f'20260911|code_protocol_followup_v2|{cid}'.encode()).digest()[:8],'big')%2147483647
   a,b=r['threshold_span'];normalized=u.encode()[:a]+b'{threshold}'+u.encode()[b:]
   rr.update(condition_id=cid,experiment=ex,user_message=u,user_message_sha256=sha(u.encode()),rendered_prompt=rendered,rendered_prompt_sha256=sha(rendered.encode()),prompt_token_ids=tok.encode(rendered),seed=sd,normalized_sha256=sha(normalized))
   assert rr['old_rendered_prompt_sha256']==sha(base.encode())
   if ex=='I':assert rendered==base and len(rr['prompt_token_ids'])==r['n_prompt_tokens']
   else:assert original['user_message']==r['user_message']+old
   assert len(rr['prompt_token_ids'])+8192+4<=32768
   inputs.append(rr);groups[ex,r['task_id'],r['variant']].add(normalized)
 assert len(inputs)==2988 and all(len(g)==1 for g in groups.values())
 inputs.sort(key=lambda r:sha(f"20260911|{r['condition_id']}".encode()))
 put(O/'preregistration/inference_inputs.jsonl',''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in inputs).encode())
 js(O/'preregistration/condition_list.json',[{k:r[k] for k in ('condition_id','experiment','split','task_id','variant','k','seed','rendered_prompt_sha256')} for r in inputs])
 probes=[]
 for text in ['FINAL: A','FINAL: B','\nFINAL: A','\nFINAL: B','FINAL:A','FINAL:B','FINAL:  A','FINAL:\nA','**FINAL:** A','`FINAL: A`','reason\n\nFINAL: A','FINAL:\n','FINAL:']:
  ids=tok.encode(text);ctx,status=marker_context(tok,[],ids,'stop','FINAL:')
  probes.append(dict(text=text,ids=ids,token_decodes=[tok.decode([i]) for i in ids],status=status,prefix_ids=ctx,prefix_text=tok.decode(ctx) if ctx else None,extra_ids=ids[len(ctx):] if ctx else None))
 js(O/'preregistration/tokenizer_probes.json',probes)
 for r in inputs:
  ids=r['prompt_token_ids'];text=r['rendered_prompt']
  if r['experiment']=='A':ids=ids+tok.encode('Reasoning.\nFINAL:');text=tok.decode(ids)
  answer_paths(tok,text,ids)
 js(O/'preregistration/input_audit.json',dict(main_A=1440,main_I=1440,smoke_A=54,smoke_I=54,threshold_invariant_groups=len(groups),old_immediate_prompt_byte_matches=len(oldprompts),A_means='met',B_means='not met',truth_not_in_inference_inputs=True,scorer_context_checks=len(inputs),tokenizer_probes=len(probes)))
 print('PREPARED',len(inputs),'inputs; protected',len(protected),'files; probes',[(p['text'],p['status']) for p in probes])
if __name__=='__main__':main()
