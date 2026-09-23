"""Create the extension protocol once, before any new smoke or main inference."""
import pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from formalcrrc import code_five as f,code_extension as cp

o=f.out(ROOT)
data=dict(created_at=cp.now(),study='Code acceptance five-model extension after observed v2 results',
    status='Prospectively fixed new-model expansion; not part of original v2 preregistration',models=f.MODELS,
    data='Exactly v2 80 main and 3 smoke tasks, same generated Mistral candidates, eight tests, expected values and verified truth. No generation, repair, task replacement, new generator or new precision audit.',
    design=dict(main_tasks=80,smoke_tasks=3,templates=['original','explicit'],thresholds=list(range(9)),partial_main_tasks=23,
        new_main_contexts=4320,new_main_score_records=8640,new_smoke_contexts=162,new_smoke_score_records=324,qwen3_main_reasoning_requests=1440),
    dtype='bfloat16',quantization=None,immediate_backend='Transformers sdpa, same environment and weight dtype as v2; no TF32; single token raw logits or exact sequence conditional likelihood',
    answer_forms=list(f.FORMS),answer_tokenization='Resolve full-context continuations using verified v2 resolver. Preserve actual supplied prompt IDs if a trailing whitespace BPE merge makes full re-encoding unstable: stable sentinel continuation must decode to an allowed form behind actual context IDs. Never use isolated label tokenization as a score index. Deduplicate paths and reject A/B collisions or strict prefix overlap.',
    union='Full-vocabulary conditional log-probabilities per path; sum distinct sequence probabilities by label. Save every path and component. No outcome-based form selection. Bare and union share forward work and (Q3) one reasoning trace.',
    qwen3=dict(temperature=.6,top_p=.95,top_k=20,min_p=0,n=1,reasoning_token_budget=8192,native_stop='</think>',post_think_separator='\n\n',
        engine='vllm 0.28.0, unquantized bf16, per-row seeds, stop_token_ids, include_stop_str_in_output=True',
        seed_rule="int.from_bytes(sha256(f'42|code_extension_v3_five_models|{task_id}|{variant}|{k}'.encode()).digest()[:8], 'big') % 2147483647; explicit complete map in inputs.json",
        natural_end_rule='Original completion.token_ids must end in the native think-end ID, within 8192 generated tokens, and finish_reason must be stop. Preserve exact raw output. Never append an absent marker, never force-close or resample a semantic failure.',
        score_rule='Same completed native trace and separator for both score forms. Re-resolve label continuations against actual final context IDs. Teacher-force each distinct path and read full-vocabulary prompt_logprobs. Extra one-token decode required by vLLM likelihood API is counted separately from reasoning and never used for a judgment.',
        history_review='Reuses historical native reason-then-score settings, row_seed SHA256 design, split_reasoning, build_decision_context and sequence likelihood. Historical runner conditionally appended think-end on generic stop; this extension explicitly rejects that fallback.',
        forward_reporting='Report generation requests, output tokens, teacher-forced path requests/positions and actual measured forwards separately. vLLM physical forward calls unavailable without engine instrumentation are null, never substituted by logical scores.'),
    retries='Completed scores and saved traces immutable and skipped. Truncated/wrong-boundary traces are terminal missing scores; never resample. Infrastructure failures may resume the same fixed row/seed at most twice after the initial attempt, retaining attempts. Scoring recovery reuses any saved natural trace. No automatic scientific/validation-error retry.',
    ordering='By split(smoke first), task_id, template, k; Q3 chunks of 54 contexts; different k always has its own reasoning and seed.',
    smoke_gate='All three smoke tasks per model before main. CPU data/tokenizer/metric checks first. Estimate remaining time from actual smoke incl 50% margin and load; defer if beyond authorized available allocation. Save incomplete curves and every denominator.',
    budget='Continue within existing v2 authorization and its remaining budget. No new eight-hour grant. Conservative additional reservation cap 3 GPU h; actual prior GPU use 6824 s separately recorded.',
    analysis=dict(metrics=['accuracy','strict_accuracy','trr','fsrr','tce','oracle_min_errors','k0_correct','strict_trr','strict_fsrr'],subsets=['all z=0..8','nontrivial z=0..7','partial z=1..7'],
        bootstrap_replicates=5000,bootstrap_seed=20260911,bootstrap_unit='task, all nine thresholds together; model/template/form contrasts paired by task',
        missing='Report all expected 80-task denominators, complete/failed/missing curves; per-model all-valid and five-model common-complete analyses separately; no hidden intersection.',
        limitations='Four immediate-readout models and one native reason-then-score model; no isolated causal effect of reasoning. One reasoning sample per row: task bootstrap does not estimate repeated-reasoning variability. Prior two models alone have the ten-task float32 audit.'))
cp.immutable(o/'protocol.json',data)
print('PROTOCOL_SAVED',cp.sha((o/'protocol.json').read_bytes()))
