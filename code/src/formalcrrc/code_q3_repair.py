"""Post-observation two-stage continuation; immutable original evidence."""
import hashlib, math, pathlib
from formalcrrc import code_extension as cp, code_five as f

NAME='code_extension_v4_q3_continuation'
TOTAL_BUDGET=32768
ADDITIONAL_BUDGET=24576
MODEL_LENGTH=40960
MAX_SEQS=8

def out(root): return pathlib.Path(root)/'artifacts'/NAME
def seed(row):
    s=f"42|{NAME}|{row['task_id']}|{row['variant']}|{row['k']}"
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8],'big')%2147483647
def trace_path(root,row): return out(root)/'continuations'/(f.key(row)+'.json')
def score_path(root,row): return out(root)/'scores'/(f.key(row)+'.json')
def receipt_path(root,row): return out(root)/'receipts'/(f.key(row)+'.json')
def node_path(root,row,prefix):
    tag='root' if not prefix else '_'.join(map(str,prefix))
    return out(root)/'nodes'/f.key(row)/(tag+'.json')

def trie(paths):
    paths=set(map(tuple,paths))
    assert paths and all(paths)
    assert not any(len(a)<len(b) and b[:len(a)]==a for a in paths for b in paths)
    return {p:sorted({q[len(p)] for q in paths if len(q)>len(p) and q[:len(p)]==p})
            for p in sorted({q[:i] for q in paths for i in range(len(q))},key=lambda x:(len(x),x))}

def combine(forms,nodes):
    paths={tuple(e['token_ids']) for e in forms.values()}
    expected=trie(paths)
    assert set(nodes)==set(expected)
    for prefix,tokens in expected.items():
        assert set(nodes[prefix])==set(tokens)
        assert all(math.isfinite(x) and x<=0 for x in nodes[prefix].values())
        assert sum(math.exp(x) for x in nodes[prefix].values())<=1.00001
    likelihood={p:sum(nodes[p[:i]][t] for i,t in enumerate(p)) for p in paths}
    scores,events=f.finish_scores(forms,likelihood)
    for event in events.values():
        p=tuple(event['token_ids'])
        event['token_conditional_log_probabilities']=[nodes[p[:i]][t] for i,t in enumerate(p)]
    assert all(s['total_probability_mass']<=1.00001 for s in scores.values())
    return scores,events

def continuation(old,row,receipt,tok,inventory):
    prefix=old['raw_generated_token_ids']; added=receipt['raw_generated_token_ids']
    end=inventory['think_end_id']
    assert not old['natural_end'] and len(prefix)==8192 and end not in prefix
    assert receipt['continuation_seed']==seed(row) and len(added)<=ADDITIONAL_BUDGET
    raw=prefix+added
    natural=bool(added and added[-1]==end and added.count(end)==1 and receipt['finish_reason']=='stop')
    ctx=row['prompt_token_ids']+raw+inventory['separator_ids'] if natural else []
    assert len(raw)<=TOTAL_BUDGET and (not ctx or len(ctx)+3<MODEL_LENGTH)
    return dict(raw_generated_token_ids=raw,additional_generated_token_ids=added,
        reasoning_text=tok.decode(raw,skip_special_tokens=False),additional_reasoning_text=tok.decode(added,skip_special_tokens=False),
        natural_end=natural,reasoning_truncated=not natural,think_end_reached=end in raw,
        reasoning_token_count=len(raw),additional_token_count=len(added),
        finish_reason=receipt['finish_reason'],stop_reason=receipt['stop_reason'],
        status='natural_complete' if natural else 'truncated_or_wrong_boundary',
        final_context_token_ids=ctx,final_context_text=tok.decode(ctx,skip_special_tokens=False) if ctx else None,
        final_context_ids_sha256=cp.sha(cp.canonical(ctx)) if ctx else None,
        original_prefix_exactly_preserved=True,two_stage_seed_not_original_rng_state=True)

def verify(root):
    o=out(root); freeze=cp.load(o/'freeze.json')
    cp.assert_manifest(root,freeze['files'])
    cp.assert_manifest(root,cp.load(o/'baseline_dependency_manifest.json')['files'])
    return cp.sha((o/'freeze.json').read_bytes())
