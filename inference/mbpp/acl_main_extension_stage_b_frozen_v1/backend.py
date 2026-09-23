"""Pinned vLLM adapter. Model imports exist only inside the authorized constructor."""
import importlib.metadata
import math
import time
from common import *

def verify_cache(root,judge):
    root=Path(root);info=read(root/'configs/judge_model_inventory.json')[judge];p=Path(info['path'])
    versions={k:importlib.metadata.version(k) for k in ['torch','transformers','vllm','tokenizers']}
    if versions!={'torch':'2.13.0+cu130','transformers':'5.16.1','vllm':'0.28.0','tokenizers':'0.23.2'}:raise RuntimeError('Pinned inference versions differ')
    if p.name!=info['revision']:raise RuntimeError('Model revision path differs')
    for name,meta in info['files'].items():
        if sha(p/name)!=meta['sha256']:raise RuntimeError('Model metadata changed: '+name)
    for shard in info['shards']:
        q=p/shard['name']
        if not q.is_file() or q.stat().st_size!=shard['bytes'] or q.resolve().name!=shard['resolved_basename']:raise RuntimeError('Model shard structure changed')
    return info,versions

class Backend:
    def __init__(self,root,judge,config,rows,authorization_token):
        # A token is issued only by the allocation gate. The constructor also
        # rechecks the gate so importing/instantiating this class cannot bypass it.
        from gates import check_live_claim
        check_live_claim(Path(root),authorization_token)
        self.info,versions=verify_cache(root,judge)
        from transformers import AutoTokenizer
        from vllm import LLM,SamplingParams
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or not torch.cuda.is_bf16_supported():raise RuntimeError('One bf16 GPU required')
        if 'H100' not in torch.cuda.get_device_name():raise RuntimeError('Unexpected GPU device')
        torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
        self.tok=AutoTokenizer.from_pretrained(self.info['path'],local_files_only=True,trust_remote_code=False)
        eos=read(Path(self.info['path'])/'config.json')['eos_token_id'];self.eos={eos} if isinstance(eos,int) else set(eos)
        self.params=SamplingParams
        for row in rows:
            text=self.tok.apply_chat_template([{'role':'user','content':row['user_message']}],tokenize=False,add_generation_prompt=True)
            if self.encode(text)!=row['prompt_token_ids'] or hashlib.sha256(text.encode()).hexdigest()!=row['rendered_prompt_sha256'] or self.decode(row['prompt_token_ids'])!=text:
                raise RuntimeError('Frozen native prompt differs before model load')
            if len(row['prompt_token_ids'])+8192+32>=config['engine']['max_model_len']:raise RuntimeError('Context length overflow')
        engine={k:v for k,v in config['engine'].items() if k not in ['torch','transformers','vllm','tokenizers']};engine['seed']=20260914
        started=time.monotonic();self.llm=LLM(model=self.info['path'],tokenizer=self.info['path'],trust_remote_code=False,**engine)
        self.load_record={'model_id':self.info['model_id'],'revision':self.info['revision'],'versions':versions,'engine':engine,
           'GPU':torch.cuda.get_device_name(),'load_seconds':time.monotonic()-started,'loaded_utc':now()}
    def encode(self,text):return self.tok.encode(text,add_special_tokens=False)
    def decode(self,ids):return self.tok.decode(list(ids),skip_special_tokens=False,clean_up_tokenization_spaces=False)
    def invoke(self,spec):
        ids=spec['context_token_ids'];params=self.params(**spec['params']);started=time.monotonic()
        output=self.llm.generate([{'prompt_token_ids':list(ids)}],params,use_tqdm=False)[0];answer=output.outputs[0]
        generated=list(answer.token_ids);before=self.decode(ids);full=self.decode(ids+generated);clean=generated[:-1] if generated and generated[-1] in self.eos else generated
        result={'prompt_token_ids':list(output.prompt_token_ids),'generated_token_ids':generated,'generated_tokens':len(generated),
            'returned_text':answer.text,'finish_reason':answer.finish_reason,'stop_reason':answer.stop_reason,'elapsed_seconds':time.monotonic()-started,
            'decoded_generated_context':full[len(before):] if full.startswith(before) else None,'prompt_decode_sha256':hashlib.sha256(before.encode()).hexdigest(),
            'decoded_without_terminal_eos':self.decode(clean),'eos_token_ids':sorted(self.eos)}
        def number(value):
            value=float(value);return value if math.isfinite(value) else 'nonfinite:'+str(value)
        if spec['mode']=='score':
            available=answer.logprobs[0] if answer.logprobs else {}
            result['conditional_logprobs']={str(t):number(available[t].logprob) for t in spec['needed_token_ids'] if t in available}
        if spec['mode']=='parity':
            result['teacher_forced_actual_token_logprobs']=[None if i==0 or output.prompt_logprobs is None or output.prompt_logprobs[i] is None or token not in output.prompt_logprobs[i] else number(output.prompt_logprobs[i][token].logprob) for i,token in enumerate(ids)]
        return result
