"""Additive, post-v2 five-model protocol. No mutation of v2 constants or evidence."""
from __future__ import annotations
import hashlib, json, math, os, pathlib, time
from formalcrrc import code_extension as cp, scoring
from formalcrrc.code_extension_answer_tokens import resolve

MODELS={
 'qwen':dict(model_id='Qwen/Qwen2.5-14B-Instruct',revision='cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8',protocol='immediate_readout',reused=True),
 'llama':dict(model_id='meta-llama/Llama-3.1-8B-Instruct',revision='0e9e39f249a16976918f6564b8830bc894c89659',protocol='immediate_readout',reused=False),
 'mistral':dict(model_id='mistralai/Mistral-7B-Instruct-v0.3',revision='c170c708c41dac9275d15a8fff4eca08d52bab71',protocol='immediate_readout',reused=True),
 'gemma':dict(model_id='google/gemma-2-9b-it',revision='11c9b309abf73637e4b6f9a3fa1e92e615547819',protocol='immediate_readout',reused=False),
 'qwen3':dict(model_id='Qwen/Qwen3-30B-A3B-Thinking-2507',revision='144afc2f379b542fdd4e85a1fcd5e1f79112d95d',protocol='native_reason_then_score',reused=False),
}
NEW=('llama','gemma','qwen3')
FORMS=('A',' A','\nA','B',' B','\nB')
CACHE='/REDACTED_LOCAL_PATH'

def out(root): return pathlib.Path(root)/'artifacts/code_extension_v3_five_models'
def key(row): return f"{row['split']}/{row['variant']}/{cp.score_key(row['task_id'],row['k'])}"
def score_path(root,model,row): return out(root)/'scores'/model/(key(row)+'.json')
def trace_path(root,row): return out(root)/'reasoning/qwen3'/(key(row)+'.json')
def snapshot(key_,cache=CACHE):
    s=MODELS[key_]; return pathlib.Path(cache)/('models--'+s['model_id'].replace('/','--'))/'snapshots'/s['revision']

def paths_for(tok,context,context_ids=None):
    """Resolve against full context, preserving supplied token IDs at a BPE boundary."""
    base=list(tok.encode(context,add_special_tokens=False)) if context_ids is None else list(context_ids)
    before=tok.decode(base,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    paths={}; forms={}
    for name in FORMS:
        ids,meta=resolve(tok,context,name); ids=tuple(ids)
        after=tok.decode(base+list(ids),skip_special_tokens=False,clean_up_tokenization_spaces=False)
        if not after.startswith(before) or after[len(before):] not in (name[-1],' '+name[-1],'\n'+name[-1]):
            raise ValueError('Continuation fails decoding behind actual supplied context IDs')
        if ids in paths and paths[ids]!=name[-1]: raise ValueError('A/B token events confused')
        paths[ids]=name[-1]
        forms[name]=dict(token_ids=list(ids),**meta,actual_context_decoded_suffix=after[len(before):])
    if any(len(a)<len(b) and b[:len(a)]==a for a in paths for b in paths): raise ValueError('Prefix-overlapping events')
    return paths,forms

def lse(values):
    vals=list(values); top=max(vals); return top+math.log(sum(math.exp(x-top) for x in vals))

def finish_scores(forms,likelihood,raw_logits=None):
    paths={tuple(v['token_ids']):name[-1] for name,v in forms.items()}
    events={name:dict(**v,log_probability=likelihood[tuple(v['token_ids'])],probability=math.exp(likelihood[tuple(v['token_ids'])])) for name,v in forms.items()}
    union={label:lse(v for p,v in likelihood.items() if paths[p]==label) for label in ('A','B')}
    bare={label:likelihood[tuple(forms[label]['token_ids'])] for label in ('A','B')}
    result={}
    for representation,values in [('bare',bare),('whitespace_union',union)]:
        if representation=='bare' and raw_logits is not None:
            a,b=raw_logits['A'],raw_logits['B']; method='single_token_next_logit'
        else: a,b=values['A'],values['B']; method='sequence_loglikelihood' if representation=='bare' else 'disjoint_sequence_probability_union'
        mass=(sum(math.exp(v) for v in likelihood.values()) if representation=='whitespace_union' else sum(math.exp(v) for v in bare.values()))
        result[representation]=dict(representation=representation,score_met=a,score_not_met=b,margin=a-b,
            p_met=scoring.normalized_probability(a,b),full_vocab_p_met=math.exp(values['A']),full_vocab_p_not_met=math.exp(values['B']),
            total_probability_mass=mass,scoring_method=method)
    return result,events

def verify_frozen(root,model):
    o=out(root); freeze=cp.load(o/f'freeze_{model}.json')
    cp.assert_manifest(pathlib.Path(root),freeze['files'])
    return cp.sha((o/f'freeze_{model}.json').read_bytes())

def preflight(root,model):
    from transformers import AutoTokenizer,AutoConfig
    o=out(root); s=MODELS[model]
    if (o/f'freeze_{model}.json').exists(): verify_frozen(root,model); print('ALREADY_FROZEN',model); return
    kw=dict(revision=s['revision'],cache_dir=CACHE,local_files_only=True,trust_remote_code=False)
    cfg=AutoConfig.from_pretrained(s['model_id'],**kw)
    assert cfg._commit_hash==s['revision'] and not getattr(cfg,'quantization_config',None)
    tok=AutoTokenizer.from_pretrained(s['model_id'],**kw)
    snap=snapshot(model); index=cp.load(snap/'model.safetensors.index.json')
    shards=[]
    for name in sorted(set(index['weight_map'].values())):
        p=snap/name; assert p.is_file()
        shards.append(dict(name=name,size=p.stat().st_size,blob=p.resolve().name))
    metadata={p.name:dict(sha256=cp.sha(p.read_bytes()),size=p.stat().st_size) for p in snap.iterdir() if p.is_file() and (p.suffix in ('.json','.jinja','.model') or p.name.endswith('.model.v3'))}
    for name in metadata:
        target=o/'tokenizers'/model/name; target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists(): target.write_bytes((snap/name).read_bytes())
    rows=[]; lengths=[]; unique_paths=set()
    for source in cp.load(o/'inputs.json')['rows']:
        row=dict(source); rendered=scoring.render_chat_prompt(tok,row['user_message']); ids=tok.encode(rendered,add_special_tokens=False)
        row.update(rendered_prompt=rendered,prompt_token_ids=ids,rendered_prompt_sha256=cp.sha(rendered),n_prompt_tokens=len(ids))
        raw=row['user_message'].encode(); a,b=row['threshold_span']
        assert raw[a:b]==str(row['k']).encode() and cp.sha(raw[:a]+b'{threshold}'+raw[b:])==row['normalized_sha256']
        if model=='qwen3':
            assert rendered.endswith('<think>\n')
            # Tokenizer-only synthetic completed boundary, never a generated or scored trace.
            context=rendered+'Tokenizer boundary probe.\n</think>\n\n'
        else: context=rendered
        paths,forms=paths_for(tok,context)
        row['answer_forms_preflight']=forms
        unique_paths.add(json.dumps(forms,sort_keys=True)); lengths.append(len(ids))
        assert len(ids)+(8192+8 if model=='qwen3' else 8)<=min(cfg.max_position_embeddings,16384 if model=='qwen3' else 32768)
        rows.append(row)
    inventory=dict(created_at=cp.now(),**s,snapshot=str(snap),resolved_config_commit=cfg._commit_hash,metadata=metadata,shards=shards,
        model_type=cfg.model_type,chat_template=tok.chat_template,chat_template_sha256=cp.sha(tok.chat_template),
        n_rows=len(rows),max_prompt_tokens=max(lengths),distinct_answer_manifests=len(unique_paths),dtype='bfloat16',quantization=None)
    if model=='qwen3':
        end=tok.encode('</think>',add_special_tokens=False); assert len(end)==1
        separator=scoring.label_continuation_ids(tok,rows[0]['rendered_prompt']+'probe\n</think>','\n\n')
        inventory.update(think_end_id=end[0],separator_ids=list(separator),separator='\n\n')
    cp.immutable(o/f'inventory_{model}.json',inventory)
    cp.immutable(o/'prompts'/f'{model}.json',dict(created_at=cp.now(),model_key=model,rows=rows))
    files=[o/'inputs.json',o/'protocol.json',o/'v2_dependency_manifest.json',o/'v2_reuse_audit.json',o/f'inventory_{model}.json',o/'prompts'/f'{model}.json']
    files += [pathlib.Path(root)/p for p in ['src/formalcrrc/code_five.py','src/formalcrrc/code_five_inference.py','scripts/code_five_run.py','src/formalcrrc/code_extension.py','src/formalcrrc/code_extension_answer_tokens.py','src/formalcrrc/scoring.py','src/formalcrrc/reasoning_anchor.py']]
    cp.immutable(o/f'freeze_{model}.json',dict(created_at=cp.now(),model=model,files=cp.manifest(pathlib.Path(root),files),stage='Before smoke and main inference; post-v2 observed-results extension'))
    print('PREFLIGHT_FROZEN',model,len(rows),max(lengths),len(unique_paths),flush=True)
