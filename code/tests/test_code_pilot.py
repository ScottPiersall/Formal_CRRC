import json
import random
import subprocess
from pathlib import Path
import pytest
from formalcrrc import code_pilot as cp
from formalcrrc import code_pilot_sandbox as sb
from formalcrrc.sequence_recovery import curve_diagnostics

def enumeration(m,z):
    # Independent ordering partition: all-false, then enable each tied score group.
    groups={v:{i for i,x in enumerate(m) if x==v} for v in m}
    enabled=set(); patterns=[set()]
    for v in sorted(groups,reverse=True):
        enabled=enabled|groups[v]; patterns.append(enabled)
    truth=set(range(z+1))
    return (any(next((i for i in range(9) if i not in p),9)==z+1 for p in patterns),
            any(p==truth for p in patterns),min(len(p^truth) for p in patterns))

def test_recovery_independent_enumeration():
    rng=random.Random(910)
    curves=[[0]*9,list(range(9)),list(range(8,-1,-1)),[3,2,1,0,-1,1,-3,-4,-5]]
    curves += [[rng.choice([-2,-1,0,1,2]) for _ in range(9)] for _ in range(250)]
    for m in curves:
        for z in range(9):
            result=curve_diagnostics(m,z+1); trr,fsrr,error=enumeration(m,z)
            assert result['oracle_min_errors']==error
            if z<8:
                assert result['trr']==trr
                assert result['fsrr']==fsrr
            else: assert result['trr'] is None and result['fsrr'] is None

def test_ties_zero_sign_and_nonmonotone():
    assert curve_diagnostics([0]*9,9)['accuracy']==1
    assert curve_diagnostics([0]*9,1)['trr'] is False
    d=curve_diagnostics([3,2,1,0,-1,1,-3,-4,-5],4)
    assert d['trr'] is True and d['fsrr'] is False
    assert d['oracle_min_errors']==1

def test_prompt_invariance_no_truth_leak():
    item=dict(task_id='fixture',problem='def f(x): ...',entry_point='f',
              tests=[dict(index=i,args=[i],expected_repr=str(i),assertion=f'assert f({i}) == {i}') for i in range(8)],
              reference_code='NEVER_SHOW_REFERENCE',z=7,pass_bits=[1]*7+[0])
    normalized=[]
    for k in range(9):
        row=cp.prompt(item,'def f(x): return 8',k); data=row['user_message'].encode(); a,b=row['threshold_span']
        assert data[a:b]==str(k).encode()
        normalized.append(data[:a]+b'{threshold}'+data[b:])
        assert 'NEVER_SHOW_REFERENCE' not in row['user_message']
        assert 'pass_bits' not in row['user_message']
    assert len(set(normalized))==1

def test_extraction_no_repair():
    code,meta=cp.extract('text\n```python\ndef f(x):\n return x\n```\nsecond')
    assert code=='def f(x):\n return x\n'
    code,meta=cp.extract('not valid python!')
    assert meta['extraction_status']=='syntax_error'
    assert code=='not valid python!\n'
    assert cp.extract('```python\ndef f(')[1]['extraction_status']=='syntax_error'

def test_classification_and_consistency():
    assert sb.compare(dict(category='candidate_timeout'),'0') is False
    assert sb.compare(dict(category='candidate_memory_limit'),'0') is False
    assert sb.compare(dict(category='infrastructure_error'),'0') is None
    assert sb.compare(dict(category='ok',observed_repr='[1, True]'),'[1, True]') is True
    assert sb.compare(dict(category='ok',observed_repr='1.0'),'1') is True  # preserve Python equality
    assert not sb.stable_pair(dict(category='ok',observed_repr='1'),dict(category='ok',observed_repr='2'))

def test_infrastructure_failure_not_candidate(monkeypatch):
    def no_docker(*args,**kwargs): raise FileNotFoundError('docker unavailable fixture')
    monkeypatch.setattr(subprocess,'run',no_docker)
    assert sb.execute('def f(): return 1','f',[[]])['status']=='infrastructure_error'

def test_immutable_resume_and_missing_threshold(tmp_path,monkeypatch):
    out=cp.outdir(tmp_path)
    cp.immutable(out/'instances.json',[dict(task_id='fixture',split='main')])
    cp.immutable(out/'freeze_B.json',dict(files={}))
    monkeypatch.setattr(cp,'assert_freeze_a',lambda r:None)
    monkeypatch.setattr(cp,'assert_freeze_b',lambda r:None)
    for key in cp.MODELS:
        for k in range(9):
            row=dict(task_id='fixture',model_key=key,k=k,revision=cp.MODELS[key]['revision'],margin=1,score_met=1,score_not_met=0)
            path=out/'scores/main'/key/f'fixture__k{k}.json'
            cp.immutable(path,row); cp.immutable(path,row)
    assert len(list((out/'scores/main').rglob('*.json')))==18
    assert cp.integrity(tmp_path)['passed']
    (out/'scores/main/mistral/fixture__k3.json').unlink()
    assert not cp.integrity(tmp_path)['passed']
    assert cp.integrity(tmp_path)['models']['mistral']['missing']==[('fixture',3)]
