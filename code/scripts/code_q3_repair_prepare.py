"""CPU verification and freeze before any supplemental inference."""
import pathlib,sys,subprocess,json,collections,concurrent.futures
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from formalcrrc import code_extension as cp,code_five as f,code_q3_repair as q
from formalcrrc.code_extension_answer_tokens import BackendTokenizer

def main():
    o=q.out(ROOT);o.mkdir(parents=True,exist_ok=True)
    if (o/'freeze.json').exists():q.verify(ROOT);print('ALREADY_FROZEN');return
    old=f.out(ROOT);hand=cp.load(old/'handoff_manifest.json')
    def check(item):
        name,digest=item;assert cp.sha((ROOT/name).read_bytes())==digest,name
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(check,hand['files'].items()))
    print('HISTORICAL_MANIFEST_VERIFIED',len(hand['files']),flush=True)
    rows=cp.load(old/'prompts/qwen3.json')['rows'];inv=cp.load(old/'inventory_qwen3.json')
    tok=BackendTokenizer(old/'tokenizers/qwen3/tokenizer.json')
    traces=[];sources=[old/'handoff_manifest.json',old/'inputs.json',old/'protocol.json',old/'inventory_qwen3.json',old/'prompts/qwen3.json',old/'freeze_qwen3.json']
    sources+=list((old/'tokenizers/qwen3').glob('*'))
    counts=collections.Counter(); norms=collections.defaultdict(set);maxlen=0;distinct=set()
    for row_number,row in enumerate(rows):
        path=f.trace_path(ROOT,row);t=cp.load(path);sources.append(path)
        assert t['generation_seed']==row['generation_seed'] and t['revision']==f.MODELS['qwen3']['revision']
        assert tok.encode(row['rendered_prompt'])==row['prompt_token_ids']
        assert row['rendered_prompt'].endswith('<think>\n') and cp.sha(row['user_message'])==row['user_message_sha256']
        raw=row['user_message'].encode();a,b=row['threshold_span'];assert raw[a:b]==str(row['k']).encode()
        norm=cp.sha(raw[:a]+b'{threshold}'+raw[b:]);assert norm==row['normalized_sha256']
        norms[row['task_id'],row['variant']].add(norm)
        assert len(t['raw_generated_token_ids'])==t['reasoning_token_count']
        if t['natural_end']:
            ctx=row['prompt_token_ids']+t['raw_generated_token_ids']+inv['separator_ids']
            assert ctx==t['final_context_token_ids'] and t['raw_generated_token_ids'][-1]==inv['think_end_id']
            assert cp.sha(cp.canonical(ctx))==t['final_context_ids_sha256']
            context=t['final_context_text']
        else:
            assert t['finish_reason']=='length' and t['reasoning_token_count']==8192 and inv['think_end_id'] not in t['raw_generated_token_ids']
            context=row['rendered_prompt']+'CPU boundary probe.\n</think>\n\n';ctx=tok.encode(context)
        paths,forms=f.paths_for(tok,context,ctx);nodes=q.trie(paths)
        distinct.add(json.dumps({k:v['token_ids'] for k,v in forms.items()},sort_keys=True))
        counts[row['split']+'/'+('natural' if t['natural_end'] else 'truncated')]+=1
        maxlen=max(maxlen,len(row['prompt_token_ids'])+q.TOTAL_BUDGET+len(inv['separator_ids'])+3)
        traces.append(dict(key=f.key(row),task_id=row['task_id'],split=row['split'],variant=row['variant'],k=row['k'],z=row['z'],
            original_trace_path=path.relative_to(ROOT).as_posix(),original_trace_sha256=cp.sha(path.read_bytes()),
            original_natural_end=t['natural_end'],continuation_seed=q.seed(row),answer_forms=forms))
        if row_number%100==99:print('CPU_CONTEXTS_CHECKED',row_number+1,flush=True)
    assert counts=={'main/natural':1380,'main/truncated':60,'smoke/natural':52,'smoke/truncated':2}
    assert len(distinct)==1 and all(len(x)==1 for x in norms.values()) and maxlen<q.MODEL_LENGTH
    sources+=list((old/'scores/qwen3').rglob('*.json'))
    sources+=[ROOT/p for p in hand['files'] if p.startswith('src/') or p.startswith('scripts/code_five')]
    dep=dict(created_at=cp.now(),files=cp.manifest(ROOT,sources),full_baseline_manifest=hand,
        original_handoff_zip='FormalCRRC_Code_Five_Models_Handoff.zip',original_handoff_sha256='c5318a3e97c4de510b9ade2451a65fa5ef5e66153f4f6ff76d0ba314daa8108f')
    cp.immutable(o/'baseline_dependency_manifest.json',dep)
    cp.immutable(o/'inputs.json',dict(created_at=cp.now(),rows=traces))
    protocol=dict(created_at=cp.now(),study='Post-observation authorized supplement; not original v2/v3 preregistration',
        model=f.MODELS['qwen3'],dtype='bfloat16',quantization=None,
        continuation=dict(original_prefix_tokens=8192,additional_budget=24576,total_budget=32768,n=1,temperature=.6,top_p=.95,top_k=20,min_p=0,
            stop_token_ids=[inv['think_end_id']],include_stop_str_in_output=True,
            prefix='Exact original prompt token IDs followed by saved 8192 generated IDs. No inserted text.',
            seeds='SHA256 first 8 bytes big endian of 42|code_extension_v4_q3_continuation|task_id|variant|k modulo 2147483647; full mapping in inputs.json',
            interpretation='Two-stage seeded continuation; original sampler RNG state unavailable, not equivalent to single-shot 32k.',
            selection='Every original truncated row exactly once: 2 smoke, 60 main. No resampling of terminal failures.'),
        scoring=dict(scope='All original natural contexts plus all naturally completed continuations. Same reasoning for both forms.',
            algorithm='Disjoint contextual answer paths; one shared full-vocabulary float32 log_softmax on bf16 model logits per prefix; select needed IDs only after normalization.',
            vllm_api='SamplingParams(max_tokens=1, temperature=0, logprob_token_ids=outgoing_ids); logprobs_mode=raw_logprobs',
            forms=list(f.FORMS),mass_limit=1.00001,generated_scoring_token='One ignored greedy token per prefix request, counted separately; no new reasoning.'),
        engine=dict(max_model_len=q.MODEL_LENGTH,max_num_seqs=q.MAX_SEQS,max_num_batched_tokens=512,enable_chunked_prefill=True,enable_prefix_caching=False,
            gpu_memory_utilization=.92,tensor_parallel_size=1,enforce_eager=False,seed=42,logprobs_mode='raw_logprobs'),
        budget=dict(original_cap_gpu_seconds=28800,prior_conservatively_charged_gpu_seconds=17925,remaining_gpu_seconds=10875,
            rule='Reserve before sbatch; reconcile terminal parent and cleanup maximum elapsed. No new allowance.',
            smoke_gate='Complete all 54 smoke contexts or terminal reasoning failures first. Estimate main from measured continuation token throughput and score input-token throughput with 50% margin. If full estimate exceeds remaining, report and run only batches conservatively fitting allocation; stop before deadline.'),
        scheduling=dict(stages=['smoke','main'],continuation_batch_size=8,score_context_batch_size=32,
            main_order='Uniformly rescore all original natural main contexts first, then continue saved truncated prefixes in frozen row order and score natural completions.',
            retry='Only infrastructure failures may resume missing receipts, at most two infrastructure attempts per logical request, same seed/prefix. Never repeat a saved terminal result.'),
        statistics=f'Unchanged v2 metrics, task bootstrap 5000 with seed {cp.SEED}; all80, z0..7, z1..7, k1..8; all valid and five-model common complete sets separately.',
        versions=['original_8k_old_scorer','original_8k_shared_prefix_scorer','two_stage_32k_shared_prefix_scorer'],
        limitations=['Post-observation design','Two-stage RNG differs from uninterrupted generation','One reasoning draw; task bootstrap does not estimate resampling variability','Four immediate readout models versus one native reasoning model does not identify a causal reasoning effect','No new full-model float32 audit'])
    cp.immutable(o/'protocol.json',protocol)
    cp.immutable(o/'cpu_preflight.json',dict(created_at=cp.now(),historical_manifest_files_verified=len(hand['files']),counts=dict(counts),
        maximum_possible_context_tokens=maxlen,distinct_answer_paths=list(distinct),all_input_and_truth_hashes_unchanged=True,threshold_only_curve_variation=True,
        native_chat_template_sha256=inv['chat_template_sha256'],tokenizer_metadata_verified=True,all_continuation_seeds_frozen=True))
    files=list((ROOT/'src/formalcrrc').glob('code_q3_repair*.py'))+list((ROOT/'scripts').glob('code_q3_repair*.py'))+list((ROOT/'slurm').glob('run_code_q3_repair*.sbatch'))+list((ROOT/'tests').glob('test_code_q3_repair*.py'))
    files+=[o/n for n in ('protocol.json','inputs.json','cpu_preflight.json','baseline_dependency_manifest.json')]
    cp.immutable(o/'freeze.json',dict(created_at=cp.now(),files=cp.manifest(ROOT,files),stage='Before supplemental smoke or main GPU inference'))
    print('FROZEN',dict(counts),'max_context',maxlen,flush=True)
if __name__=='__main__':main()
