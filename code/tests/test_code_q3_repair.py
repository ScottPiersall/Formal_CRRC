import math
import pytest
from formalcrrc import code_q3_repair as q

def test_shared_probability_tree():
    forms={name:dict(token_ids=p) for name,p in [('A',[1]),('B',[2]),(' A',[3]),(' B',[4]),('\nA',[5,1]),('\nB',[5,2])]}
    nodes={():{1:math.log(.5),2:math.log(.2),3:math.log(.03),4:math.log(.02),5:math.log(.1)},(5,):{1:math.log(.7),2:math.log(.3)}}
    scores,events=q.combine(forms,nodes)
    assert scores['whitespace_union']['total_probability_mass']==pytest.approx(.85)
    assert scores['whitespace_union']['p_met']==pytest.approx(.6/.85)
    assert events['\nA']['probability']==pytest.approx(.07)
    assert len(q.trie([tuple(e['token_ids']) for e in forms.values()]))==2

def test_invalid_joint_vector_rejected():
    with pytest.raises(AssertionError):q.combine({'A':{'token_ids':[1]},'B':{'token_ids':[2]}},{():{1:math.log(.9),2:math.log(.2)}})

def test_overlapping_paths_rejected():
    with pytest.raises(AssertionError):q.trie([(1,),(1,2)])

def test_dedup_paths_not_double_counted():
    forms={x:dict(token_ids=[1 if x[-1]=='A' else 2]) for x in ['A',' A','\nA','B',' B','\nB']}
    scores,_=q.combine(forms,{():{1:math.log(.6),2:math.log(.4)}})
    assert scores['whitespace_union']['total_probability_mass']==pytest.approx(1.)

class Tok:
    def decode(self,ids,**kw):return ','.join(map(str,ids))

@pytest.mark.parametrize('tail,reason,natural',[([7,9],'stop',True),([7,8],'length',False),([9],'length',False)])
def test_continuation_requires_native_stop(tail,reason,natural):
    row=dict(task_id='HumanEval/1',variant='original',k=2,prompt_token_ids=[100])
    old=dict(natural_end=False,raw_generated_token_ids=[3]*8192)
    receipt=dict(raw_generated_token_ids=tail,continuation_seed=q.seed(row),finish_reason=reason,stop_reason=9 if reason=='stop' else None)
    result=q.continuation(old,row,receipt,Tok(),dict(think_end_id=9,separator_ids=[11]))
    assert result['natural_end']==natural and result['raw_generated_token_ids'][:8192]==old['raw_generated_token_ids']
    assert bool(result['final_context_token_ids'])==natural

def test_seed_mapping_changes_only_defined_keys():
    row=dict(task_id='HumanEval/1',variant='original',k=2)
    assert q.seed(row)==q.seed(dict(row,z=4))
    assert len({q.seed(dict(row,k=k)) for k in range(9)})==9
