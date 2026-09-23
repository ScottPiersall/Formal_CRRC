"""Resolve reply tokens without retokenizing the already supplied prompt."""
import pathlib
from formalcrrc import code_extension as cp
from formalcrrc import scoring

def resolve(tok,prompt,answer):
    try:
        ids=scoring.label_continuation_ids(tok,prompt,answer); method='contextual_full_encode'
    except ValueError as e:
        if "changed the prompt's own tokenisation" not in str(e): raise
        # For Qwen, a second newline can merge with the prompt's final newline
        # during full-string re-encoding. The supplied prompt token IDs remain
        # fixed at inference. Resolve the reply behind a stable text boundary,
        # then validate it by decoding behind the actual, unchanged prompt IDs.
        ids=scoring.label_continuation_ids(tok,'x',answer); method='boundary_preserving_sentinel'
    base=list(tok.encode(prompt,add_special_tokens=False))
    before=tok.decode(base,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    after=tok.decode(base+list(ids),skip_special_tokens=False,clean_up_tokenization_spaces=False)
    if not after.startswith(before): raise RuntimeError('Reply decoding changed the prompt')
    suffix=after[len(before):]
    if suffix not in (answer[-1],' '+answer[-1],'\n'+answer[-1]): raise RuntimeError('Reply token path does not decode to an allowed label form')
    return ids,dict(method=method,decoded_suffix=suffix)

class BackendTokenizer:
    def __init__(self,path):
        from tokenizers import Tokenizer
        self.backend=Tokenizer.from_file(str(path))
    def encode(self,text,add_special_tokens=False): return self.backend.encode(text,add_special_tokens=add_special_tokens).ids
    def decode(self,ids,**kwargs): return self.backend.decode(list(ids),skip_special_tokens=False)

def prepare(root,development=False):
    root=pathlib.Path(root); out=cp.outdir(root)
    folder=out/'development_prompts' if development else out/'prompts'
    inventory=cp.load(out/'model_inventory.json')
    for key in cp.MODELS:
        dest=folder/f'{key}__answer_tokens.json'
        if dest.exists(): continue
        path=out/'upstream/tokenizers'/f'{key}.json'
        assert cp.sha(path.read_bytes())==inventory['models'][key]['files']['tokenizer.json']['sha256']
        tok=BackendTokenizer(path); result=[]
        for row in cp.load(folder/f'{key}.json')['rows']:
            prompt=row['rendered_prompt']; assert len(tok.encode(prompt))==row['n_prompt_tokens']
            forms={}
            for label in ['A','B']:
                for prefix in ['',' ','\n']:
                    text=prefix+label; ids,meta=resolve(tok,prompt,text)
                    forms[text]=dict(token_ids=list(ids),**meta)
            assert forms['A']['token_ids']==row['label_tokenization']['met_token_ids']
            assert forms['B']['token_ids']==row['label_tokenization']['not_met_token_ids']
            result.append(dict(task_id=row['task_id'],variant=row['variant'],k=row['k'],rendered_prompt_sha256=row['rendered_prompt_sha256'],answer_forms=forms))
        cp.immutable(dest,dict(created_at=cp.now(),model_key=key,tokenizer_sha256=cp.sha(path.read_bytes()),rows=result))
        print('ANSWER_TOKEN_PREFLIGHT',key,len(result),flush=True)
