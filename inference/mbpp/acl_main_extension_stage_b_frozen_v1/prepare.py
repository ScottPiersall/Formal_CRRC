"""Copy previously verified inputs into an exclusive final-freeze workspace."""
import ast
import hashlib
import subprocess
from common import *
PREP=ROOT.parent/'acl_main_extension_stage_b_preparation_v1'
GEN=ROOT.parent/'acl_main_extension_candidate_run_v1'
DEV=ROOT.parent/'acl_main_extension_smoke_preparation_v1/retry_review_v1'
def main():
    manifest=PREP/'completed_preparation_manifest.sha256.json'
    if sha(manifest)!='4771610d242c25712b5ba69a02bf66336eef66d650b1d972e516ae4e00c03492':raise RuntimeError('Preparation snapshot changed')
    n=verify_manifest(PREP,manifest);base=read(PREP/'audit/baseline.json')
    diff=subprocess.run(['git','diff','--binary','HEAD'],cwd=ROOT.parent.parent,capture_output=True,check=True).stdout
    if hashlib.sha256(diff).hexdigest()!=base['tracked_diff_sha256']:raise RuntimeError('User tracked changes differ')
    save(ROOT/'audit/baseline.json',{'created_utc':now(),'preparation_manifest_sha256':sha(manifest),'preparation_files_verified':n,'tracked_diff_sha256':hashlib.sha256(diff).hexdigest(),'prior_preservation':read(PREP/'audit/final_verification.json')['preserved_packages'],'AGENTS_found':[]})
    paths=['data/tasks.json','data/truth_inputs.json','data/candidate_truth_bindings.json','data/judge_inputs.json','data/native/qwen.json','data/native/mistral.json',
      'configs/templates.json','configs/protocol_definition.json','configs/judge_model_inventory.json','configs/judge_chunks.draft.json',
      'results/truth_summary.json','results/truth_edge_review.json','results/judge_budget.json','results/realized_sample_planning.json','runtime/scoring_core.py',
      'audit/current_resources.json','audit/harness_reuse.json','truth_execution_manifest.json']
    bindings={}
    for rel in paths:
        p=ROOT/rel;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as f:f.write((PREP/rel).read_bytes())
        bindings[rel]=sha(PREP/rel)
    # Keep the numerical/statistical functions unchanged; replace the incomplete CLI with a full exporter.
    source=PREP/'analysis.py';text=source.read_text(encoding='utf-8');tree=ast.parse(text)
    functions=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name!='main']
    text_file(ROOT/'statistics_core.py','"""Unchanged validated task-level statistical core from Stage B preparation."""\nimport math\nimport numpy as np\n'+ '\n\n'.join(ast.get_source_segment(text,n) for n in functions)+'\n')
    c=read(PREP/'configs/experiment.stage_B_draft.json')
    c['status']='FINAL_STAGE_B_CONTENT_PENDING_FREEZE';c['scope']='formal_276_fixed_candidates_judge_only_v1'
    c['resources']=c.pop('resource_draft');c['resources']['chunks']='configs/judge_chunks.draft.json'
    c['resources']['slot_rule']='Maximum 24 chronological submission slots, each <=8h. Fixed 24 logical chunks; a chunk retry consumes another slot and may leave other chunks missing at the cap. No additional slots or automatic submission.'
    c['parity']={'task_id':c['task_ids'][0],'threshold':0,'forms':6,'conditions_per_judge':4,'max_abs_sequence_logprob_difference':read(DEV/'configs/smoke.json')['parity']['max_abs_sequence_logprob_difference']}
    c['recovery_contract']['automatic_generation_retries']=0
    c['recovery_contract']['explicit_score_recovery']='Only after prior job terminal, explicit request IDs at submission, at most two additional deterministic score/parity attempts. All original attempts remain. No generation recovery permits.'
    c['freeze']={'Stage_A':True,'Stage_B':False,'pending':['CPU implementation verification and final content freeze']}
    c['authorization']={'judge_inference':False,'GPU_submission':False,'new_candidate_generation':False,'model_download':False}
    c['reporting']={'strata':['all','partial','z0','z8','nonconstant'],'descriptive_CIs':'Task-level percentile bootstrap, PCG64 seed 20260914, 9999 resamples; marginal 95%, no added significance tests.',
       'count_missing':'Report parsed-only performance, successful output/planned yield, parse failures/observed and /planned, unobserved requests, and full-population identification bounds. Do not impute a count.',
       'visible_prefix_denominators':'marker-only/valid scored RTS contexts AND marker-only/planned contexts; report valid boundary separately when scoring missing.',
       'joint_directions':'Report the observed sign pair and whether both original/bare primary tests pass their six-test Holm adjustment; no requirement to obtain accuracy up / FSRR down.',
       'heldout_analysis':'Optional exploratory heldout-cut analysis is not executed by this confirmatory report entry point; any future addition must be labeled exploratory.'}
    save(ROOT/'configs/experiment.json',c)
    save(ROOT/'audit/source_bindings.json',{'copied_unchanged':bindings,'statistics_source_sha256':sha(source),'numerical_functions_copied_unchanged':len(functions),'source_Stage_A_manifest_sha256':sha(GEN/'stage_a_manifest.json'),'new_model_requests':0})
    print(json.dumps({'copied_inputs':len(paths),'task_count':276,'partial_count':33,'GPU_submissions':0}))
if __name__=='__main__':main()
