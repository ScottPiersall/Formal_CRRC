"""Explicit read-only SSH/tokenizer action; no model imports, downloads or jobs.

The default validation entry point never invokes this script.
Outputs are exclusive. Run only once in a new review package.
"""
import json
import subprocess
from build_review import ROOT, RETRY, read, save, sha


def main():
    prompts = read(ROOT / 'data/generation_prompts.reviewed_draft.json')
    inventory = read(RETRY / 'configs/model_inventory.json')
    code = 'PROMPTS=' + repr(prompts) + '\nINVENTORY=' + repr(inventory) + '\n' + r'''
import datetime, hashlib, importlib.metadata, json, pathlib, sys
from tokenizers import Tokenizer
from jinja2 import Environment, StrictUndefined
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def fail(s): raise ValueError(s)
models={}
for model in ['qwen','mistral']:
 info=INVENTORY[model];p=pathlib.Path(info['path'])
 assert p.name==info['revision']
 for name in ['tokenizer.json','tokenizer_config.json']:
  assert digest(p/name)==info['files'][name]['sha256']
 cfg=json.loads((p/'tokenizer_config.json').read_text())
 models[model]=(Tokenizer.from_file(str(p/'tokenizer.json')),cfg)
tok,cfg=models['mistral']
env=Environment(trim_blocks=True,lstrip_blocks=True,undefined=StrictUndefined,extensions=['jinja2.ext.loopcontrols'])
env.globals['raise_exception']=fail
rows=[]
for row in PROMPTS:
 rendered=env.from_string(cfg['chat_template']).render(messages=[{'role':'user','content':row['user_message']}],add_generation_prompt=True,tools=None,**{k:cfg.get(k,'') for k in ['bos_token','eos_token','unk_token']})
 ids=tok.encode(rendered,add_special_tokens=False).ids
 assert tok.decode(ids,skip_special_tokens=False)==rendered
 rows.append(dict(task_id=row['task_id'],split='main_pool',rendered_prompt=rendered,prompt_sha256=hashlib.sha256(rendered.encode()).hexdigest(),prompt_token_ids=ids,n_prompt_tokens=len(ids),status='CPU_rendered_not_generated'))
root=pathlib.Path('/REDACTED_LOCAL_PATH')
inputs={r['condition_id']:r for r in json.loads((root/'judge_inputs.json').read_text())}
checks=[]
for p in sorted((root/'requests').glob('*_generation.json')):
 req=json.loads(p.read_text());judge,tail=req['request_id'].split('_',1);cid=tail[:-len('_generation')]
 info=inputs[cid]
 if info['kind']!='corrected_RTS_v2':continue
 result=json.loads((root/'judge_results'/(judge+'_'+cid+'.json')).read_text())
 tok,_=models[judge];decode=lambda ids:tok.decode(ids,skip_special_tokens=False)
 before=decode(req['prompt_token_ids']);full=decode(req['prompt_token_ids']+req['generated_token_ids'])
 assert full.startswith(before) and full[len(before):]==req['returned_text']
 n=result['accepted_generated_prefix_length']
 assert result['context_token_ids']==req['prompt_token_ids']+req['generated_token_ids'][:n]
 accepted=decode(result['context_token_ids'])[len(before):]
 assert accepted.endswith('FINAL:') and accepted.count('FINAL:')==1
 pre=accepted[:-len('FINAL:')]
 checks.append(dict(request_id=req['request_id'],judge=judge,task_id=info['task_id'],template=info['template'],k=info['k'],request_file_sha256=digest(p),result_file_sha256=digest(root/'judge_results'/(judge+'_'+cid+'.json')),full_decode_matches_returned_text=True,actual_accepted_prefix_verified=True,visible_prefix=bool(pre.strip()),pre_marker_characters=len(pre),pre_marker_whitespace_words=len(pre.split())))
assert len(checks)==216
assert not any(m in sys.modules for m in ['torch','transformers','vllm'])
print(json.dumps(dict(checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),native=dict(model_id='mistralai/Mistral-7B-Instruct-v0.3',revision=INVENTORY['mistral']['revision'],native_template_sha256=hashlib.sha256(models['mistral'][1]['chat_template'].encode()).hexdigest(),rows=rows,model_loaded=False,new_generation=False),decode_checks=checks,versions={k:importlib.metadata.version(k) for k in ['tokenizers','jinja2']},model_modules_imported=[],remote_writes=False,new_candidates=0,new_inference_requests=0)))
'''
    delimiter = 'FORMALCRRC_PROTOCOL_CPU_ONLY_20260914'
    assert delimiter not in code
    script = 'export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1\n'
    script += "/REDACTED_LOCAL_PATH - <<'" + delimiter + "'\n" + code + '\n' + delimiter + '\n'
    cmd = ['wsl.exe','-d','Ubuntu-24.04','--','ssh','-S','/REDACTED_LOCAL_PATH','-o','BatchMode=yes','-o','ProxyCommand=false','-o','ConnectTimeout=10','ANONYMOUS@CLUSTER','bash','-l','-s']
    completed = subprocess.run(cmd,input=script.encode(),capture_output=True,timeout=55)
    if completed.returncode:
        save('audit/native_refresh_blocked.json', {'exit_code':completed.returncode,'stderr':completed.stderr.decode(errors='replace'),'formal_native_refresh':'pending','no_fallback_model_or_revision':True})
        raise SystemExit('Read-only CLUSTER CPU refresh blocked; details saved locally')
    result = json.loads(completed.stdout)
    assert len(result['native']['rows']) == 276
    # Compare remote input and result bytes against already-collected local receipts.
    for check in result['decode_checks']:
        assert sha(RETRY/'work/requests'/(check['request_id']+'.json')) == check['request_file_sha256']
        assert sha(RETRY/'work/judge_results'/(check['request_id'][:-len('_generation')]+'.json')) == check['result_file_sha256']
    save('data/generation_native.reviewed_draft.json', result.pop('native'))
    save('audit/CLUSTER_cpu_refresh.json', result)
    print(json.dumps({'native_prompts':276,'existing_RTS_decode_checks':216,'new_model_requests':0,'remote_writes':False,'model_modules_imported':result['model_modules_imported']}))


if __name__ == '__main__':
    main()
