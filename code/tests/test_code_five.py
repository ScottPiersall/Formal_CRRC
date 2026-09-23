import math
import pytest
from formalcrrc import code_five as f
from formalcrrc import reasoning_anchor as ra
from formalcrrc.code_extension_analysis import independent,metric_row

def test_merged_probability_deduplicates_token_events():
    forms={name:{'token_ids':list(path)} for name,path in [('A',(1,)),(' A',(1,)),('\nA',(3,1)),('B',(2,)),(' B',(2,)),('\nB',(3,2))]}
    likelihood={(1,):math.log(.2),(2,):math.log(.3),(3,1):math.log(.1),(3,2):math.log(.15)}
    scores,events=f.finish_scores(forms,likelihood)
    assert scores['whitespace_union']['total_probability_mass']==pytest.approx(.75)
    assert scores['whitespace_union']['p_met']==pytest.approx(.4)
    assert scores['bare']['p_met']==pytest.approx(.4)

def test_native_boundary_cannot_be_fabricated():
    trace=ra.split_reasoning([1,2,3],9,3)
    assert trace.truncated and not trace.think_end_reached
    assert trace.token_ids==(1,2,3)
    assert ra.split_reasoning([1,9],9,3).think_end_reached

def test_ties_and_threshold_enumeration():
    # A tie at the acceptance boundary prevents perfect recovery, even though
    # the unshifted sign can look reasonable at most thresholds.
    m=[2.,2.,2.,1.,1.,-1.,-1.,-1.,-1.]
    row=metric_row('fixture','llama','original','bfloat16',m,3)
    assert row['trr'] is False and row['fsrr'] is False
    assert independent(m,3)['oracle_min_errors']==1
