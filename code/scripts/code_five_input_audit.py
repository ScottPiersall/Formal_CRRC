"""Independent byte-level reuse and one-variable curve audit; no model calls."""
import collections,hashlib,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from formalcrrc import code_extension as cp,code_five as f

def main():
    o=f.out(ROOT);rows=cp.load(o/'inputs.json')['rows']
    prior={(r['task_id'],r['variant'],r['k']):r for r in cp.load(ROOT/'artifacts/code_extension_v2/prompts/qwen.json')['rows']}
    groups=collections.defaultdict(list);seeds=set()
    for row in rows:
        key=(row['task_id'],row['variant'],row['k']);old=prior[key]
        assert row['user_message'].encode()==old['user_message'].encode()
        assert row['split']==old['split'] and row['threshold_span']==old['threshold_span']
        truth=cp.load(ROOT/'artifacts/code_extension_v2/truth'/f'{cp.slug(row["task_id"])}.json')
        assert row['z']==truth['z']
        payload=f"42|code_extension_v3_five_models|{row['task_id']}|{row['variant']}|{row['k']}"
        seed=int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8],'big')%2147483647
        assert seed==row['generation_seed'];seeds.add(seed)
        raw=row['user_message'].encode();left,right=row['threshold_span']
        assert raw[left:right]==str(row['k']).encode()
        groups[row['split'],row['task_id'],row['variant']].append((row['k'],raw[:left]+b'{threshold}'+raw[right:]))
    assert len(rows)==1494 and len(seeds)==1494 and len(groups)==166
    for group in groups.values():
        assert sorted(k for k,_ in group)==list(range(9)) and len({value for _,value in group})==1
    token_paths={};native_curves={}
    for model in f.NEW:
        prepared=cp.load(o/'prompts'/f'{model}.json')['rows']
        assert len(prepared)==1494
        hist=collections.Counter();native_groups=collections.defaultdict(set)
        for row in prepared:
            assert row['user_message']==prior[row['task_id'],row['variant'],row['k']]['user_message']
            raw=row['user_message'].encode();rendered=row['rendered_prompt'].encode()
            trimmed=raw.strip();assert rendered.count(trimmed)==1
            start=rendered.index(trimmed);leading=len(raw)-len(raw.lstrip())
            left,right=(start+n-leading for n in row['threshold_span'])
            assert rendered[left:right]==str(row['k']).encode()
            native_groups[row['split'],row['task_id'],row['variant']].add(rendered[:left]+b'{threshold}'+rendered[right:])
            paths=tuple((name,tuple(form['token_ids']),form['actual_context_decoded_suffix']) for name,form in sorted(row['answer_forms_preflight'].items()))
            hist[paths]+=1
        token_paths[model]=[dict(contexts=n,forms=[dict(form=name,token_ids=list(ids),decoded_suffix=suffix) for name,ids,suffix in paths]) for paths,n in hist.items()]
        assert len(native_groups)==166 and all(len(values)==1 for values in native_groups.values())
        native_curves[model]=len(native_groups)
    result=dict(passed=True,checked_at=cp.now(),identical_v2_input_contexts=1494,curves_with_only_threshold_changed=166,
        main_curves=160,smoke_curves=6,distinct_seeds=1494,exact_truth_reuse=True,tokenizer_preflight_path_histograms=token_paths,
        native_rendered_curves_only_threshold_changed=native_curves,
        qwen3_note='These are tokenizer-only native-boundary probes. Actual post-reasoning paths are separately reconstructed and audited from every saved score.')
    cp.snapshot(o/'analysis/input_and_tokenizer_audit.json',result)
    print('INPUT_AND_TOKENIZER_AUDIT_PASSED',1494,166)
if __name__=='__main__':main()
