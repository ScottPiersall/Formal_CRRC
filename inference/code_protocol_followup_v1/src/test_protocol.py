import itertools, math, pathlib, random, sys, pytest
sys.path.insert(0,str(pathlib.Path(__file__).parent))
from protocol import *
sys.path.insert(0,str(OUT/'preregistration/reused_source'))
from formalcrrc import code_q3_repair as shared
from formalcrrc.code_extension_answer_tokens import BackendTokenizer

def test_independent_all_binary_nine_point_and_random_ties():
 vectors=list(itertools.product((-1.,1.),repeat=9))
 rng=random.Random(SEED);vectors += [tuple(rng.choice((-3.,0.,0.,2.,7.)) for _ in range(9)) for _ in range(1000)]
 for m in vectors:
  for z in range(9):
   for start in (0,1):
    a=diagnostic(m[start:],z,tuple(range(start,9)));b=independent(m[start:],z,tuple(range(start,9)))
    assert all(a[k]==v for k,v in b.items())
    if a['fsrr'] is not None: assert a['fsrr']+a['strict_inversion']+a['tie_only_failure']==1

def test_count_identities_all_81_pairs():
 for z,n in itertools.product(range(9),repeat=2):
  m=[n-k+.5 for k in range(9)];a=diagnostic(m,z)
  assert a['tce']==abs(n-z) and a['accuracy']==1-abs(n-z)/9
  assert a['direct_sequence_correct']==(n==z)
  assert a['oracle_fsrr_all'] and a['oracle_trr_all']

def test_count_parser_strict():
 for n in range(9):
  assert parse_count(f'analysis\nFINAL_COUNT: {n}','stop',10)==(n,'ok')
  assert parse_count(f'FINAL_COUNT: {n}\n','stop',10)==(n,'ok')
 for s in ['FINAL_COUNT: 9','FINAL_COUNT: -1','FINAL_COUNT: 2.0','FINAL_COUNT: 02','FINAL_COUNT: 2 text','FINAL_COUNT: 2 ','FINAL_COUNT: 2\n\n','FINAL_COUNT: 2\r\n','FINAL_COUNT: ２','FINAL_COUNT:2','body FINAL_COUNT: 2','FINAL_COUNT: 2\nFINAL_COUNT: 2','I think 2 tests pass','FINAL_COUNT: 2\nmore']:
  assert parse_count(s,'stop',10)[0] is None,s
 assert parse_count('FINAL_COUNT: 2','stop',8192)[0] is None
 assert parse_count('FINAL_COUNT: 2','length',10)[0] is None

def test_marker_and_exact_context_paths():
 tok=BackendTokenizer(OUT/'preregistration/tokenizer/tokenizer.json')
 prompt=tok.encode('User prompt contains FINAL: already.\nAssistant:\n')
 for body in ['Analysis\nFINAL:','FINAL:','Reasoning result FINAL:']:
  ids=tok.encode(body);ctx,status=marker_context(tok,prompt,ids,'stop','FINAL:')
  assert status=='ok' and ctx==prompt+ids
  paths,forms=answer_paths(tok,tok.decode(ctx),ctx)
  assert len(shared.trie(paths))>=1
  for text,meta in forms.items():assert tok.decode(ctx+meta['token_ids'])==tok.decode(ctx)+text
 assert marker_context(tok,prompt,tok.encode('No final output'),'stop',None)[0] is None
 assert marker_context(tok,prompt,tok.encode('FINAL:'),'length',None)[0] is None
 assert marker_context(tok,prompt,[1]*8192,'length',None)[1]=='token_cap'

def test_shared_scorer_multitoken_duplicates_prefix_overlap():
 forms={'A':{'token_ids':[1]},' A':{'token_ids':[1]},'\nA':{'token_ids':[3,1]},'B':{'token_ids':[2]},' B':{'token_ids':[2]},'\nB':{'token_ids':[3,2]}}
 nodes={():{1:math.log(.2),2:math.log(.3),3:math.log(.4)},(3,):{1:math.log(.7),2:math.log(.3)}}
 scores,_=shared.combine(forms,nodes)
 assert abs(scores['bare']['margin']-math.log(.2/.3))<1e-12
 assert abs(scores['whitespace_union']['margin']-math.log(.48/.42))<1e-12
 assert abs(scores['whitespace_union']['total_probability_mass']-.9)<1e-12
 with pytest.raises(AssertionError): shared.trie([(1,),(1,2)])
 with pytest.raises(AssertionError): shared.combine(forms,{():{1:0.,2:0.,3:0.},(3,):nodes[(3,)]})

def test_tie_and_first_crossing_are_different():
 assert diagnostic([0.]*9,4)['tie_only_failure']
 assert diagnostic([0.]*9,8)['direct_sequence_correct']
 d=diagnostic([4,3,2,-1,5,-2,-3,-4,-5],2)
 assert d['trr'] and not d['fsrr'] and d['trr_success_fsrr_failure']

def test_physical_gpu_accounting_no_double_count_or_billing_confusion():
 from cost_gate import physical_jobs
 jobs=physical_jobs('123|COMPLETED|0:0|100|billing=10,cpu=8,gres/gpu=1|gpu1\n123.batch|COMPLETED|0:0|100|gres/gpu=1|gpu1\n123.extern|COMPLETED|0:0|110|gres/gpu=1|gpu1\n124|FAILED|1:0|50|billing=20,gres/gpu=2|gpu2\n125|COMPLETED|0:0|300|billing=4,cpu=4|cpu1')
 assert jobs['123']['physical_gpu_seconds']==110
 assert jobs['124']['physical_gpu_seconds']==100
 assert jobs['125']['physical_gpu_seconds']==0
 assert sum(d['physical_gpu_seconds'] for d in jobs.values())==210
