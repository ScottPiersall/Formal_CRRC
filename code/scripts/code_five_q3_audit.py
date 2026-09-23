"""Audit saved Q3 traces, native completion receipts and likelihood components."""
import pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from formalcrrc import code_five as f,code_extension as cp
from code_five_analyze import audit_new,ProbabilityMassError

o=f.out(ROOT);inv=cp.load(o/'inventory_qwen3.json');freeze=f.verify_frozen(ROOT,'qwen3')
prepared={f.key(r):r for r in cp.load(o/'prompts/qwen3.json')['rows']}
counts={s:dict(traces=0,natural=0,truncated=0,score_contexts=0,raw_saved_score_contexts=0,validation_failed_contexts=0,receipts=0,individual_latencies=0) for s in ('main','smoke')}
validation_failures=[]
for path in (o/'reasoning/qwen3').rglob('*.json'):
    trace=cp.load(path);row=prepared[f.key(trace)];n=counts[trace['split']]
    assert trace['generation_seed']==row['generation_seed'] and trace['freeze_sha256']==freeze
    assert trace['user_message_sha256']==row['user_message_sha256'] and trace['rendered_prompt_sha256']==row['rendered_prompt_sha256']
    raw=trace['raw_generated_token_ids'];assert len(raw)==trace['reasoning_token_count'] and len(raw)<=8192
    natural=bool(raw and raw[-1]==inv['think_end_id'] and trace['finish_reason']=='stop')
    assert natural==trace['natural_end']
    n['traces']+=1;n['natural']+=natural;n['truncated']+=not natural
    if natural:assert trace['final_context_token_ids']==row['prompt_token_ids']+raw+inv['separator_ids']
    else:assert trace['final_context_token_ids']==[]
    receipt_path=o/'completion_receipts'/trace['job_id']/(f.key(trace)+'.json')
    if trace['split']=='main':
        receipt=cp.load(receipt_path);assert receipt['raw_generated_token_ids']==raw and receipt['seed']==row['generation_seed']
        assert receipt['finish_reason']==trace['finish_reason'] and receipt['stop_reason']==trace['stop_reason']
        elapsed=receipt['observed_finished_wall_time']-receipt['batch_submitted_wall_time']
        assert elapsed==trace['request_elapsed_seconds'] and elapsed>=0
        n['receipts']+=1;n['individual_latencies']+=1
    score_path=f.score_path(ROOT,'qwen3',row)
    if score_path.exists():
        assert natural
        score=cp.load(score_path);n['raw_saved_score_contexts']+=1
        try:audit_new(score)
        except ProbabilityMassError as exc:
            n['validation_failed_contexts']+=1
            validation_failures.append(dict(source=str(score_path.relative_to(ROOT)),error=str(exc),raw_preserved=True))
        else:n['score_contexts']+=1
        assert score['reasoning_sha256']==cp.sha(path.read_bytes()) and score['final_context_ids_sha256']==trace['final_context_ids_sha256']
cp.snapshot(o/'analysis/qwen3_trace_and_score_audit.json',dict(passed=not validation_failures,native_trace_checks_passed=True,all_accepted_score_checks_passed=True,
    validation_failures=validation_failures,checked_at=cp.now(),counts=counts,
    checks=['Fixed seeds and frozen context identity','Raw native end token and stop reason','No forced closure for truncated traces','Decision IDs exactly prompt plus original trace plus native separator','Full-vocabulary component probability and form union reconstruction','Main receipt raw-token identity and observed elapsed time']))
print('Q3_TRACE_AUDIT_PASSED_WITH_PRESERVED_SCORE_FAILURES' if validation_failures else 'Q3_TRACE_AND_SCORE_AUDIT_PASSED',counts)
