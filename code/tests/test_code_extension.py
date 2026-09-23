from formalcrrc import code_extension as cp
from formalcrrc import code_pilot as old
import pytest

def fixture():
    return dict(task_id='fixture',problem='def f(x): ...',entry_point='f',reference_code='SECRET_REFERENCE',
        z=3,pass_bits=[1]*3+[0]*5,tests=[dict(index=i,args=[i],expected_repr=str(i),assertion=f'assert f({i}) == {i}') for i in range(8)])

def test_both_templates_only_vary_threshold_and_hide_truth():
    item=fixture()
    for variant in cp.VARIANTS:
        normalized=[]
        for k in range(9):
            r=cp.prompt(item,'def f(x): return x',k,variant)
            b=r['user_message'].encode(); start,end=r['threshold_span']
            assert b[start:end]==str(k).encode()
            normalized.append(b[:start]+b'{threshold}'+b[end:])
            assert 'SECRET_REFERENCE' not in r['user_message'] and 'pass_bits' not in r['user_message']
        assert len(set(normalized))==1

def test_original_is_exact_replication():
    for k in range(9): assert cp.prompt(fixture(),'def f(x): return x',k)==old.prompt(fixture(),'def f(x): return x',k)

def test_evidence_paths_cannot_collide():
    from pathlib import Path
    paths={cp.score_path(Path('root'),'main',m,v,d,'HumanEval/1',k) for m in cp.MODELS for v in cp.VARIANTS for d in cp.DTYPES for k in range(9)}
    assert len(paths)==72
    assert cp.outdir(Path('root'))!=old.outdir(Path('root'))

def test_development_tampering_rejected(tmp_path):
    out=cp.outdir(tmp_path); p=out/'fixture.json'; cp.immutable(p,{'original':True})
    cp.immutable(out/'development_freeze.json',{'files':cp.manifest(tmp_path,[p])})
    cp.immutable(out/'development_freeze_checksum.json',{'sha256':cp.sha((out/'development_freeze.json').read_bytes())})
    cp.assert_development(tmp_path)
    p.write_text('{}')
    with pytest.raises(RuntimeError): cp.assert_development(tmp_path)

def test_strict_recovery_against_independent_score_cuts():
    import random
    from formalcrrc.code_extension_analysis import metric_row
    rng=random.Random(9211)
    for _ in range(150):
        margins=[rng.choice([-2,-1,0,1,2]) for _ in range(9)]
        groups=[set()]; enabled=set()
        for score in sorted(set(margins[1:]),reverse=True):
            enabled=enabled|{k for k in range(1,9) if margins[k]==score}; groups.append(enabled)
        for z in range(1,8):
            row=metric_row('fixture','mistral','original','bfloat16',margins,z)
            assert row['strict_fsrr']==any(g==set(range(1,z+1)) for g in groups)
            assert row['strict_trr']==any(next((k for k in range(1,9) if k not in g),9)==z+1 for g in groups)

def test_bootstrap_empty_and_paired_constant():
    from formalcrrc.code_extension_analysis import ci
    assert ci([])==dict(n=0,estimate=None,ci95=None)
    assert ci([None])['estimate'] is None
    assert ci([1,1,1])==dict(n=3,estimate=1.0,ci95=[1.0,1.0])

def test_answer_forms_deduplicate_and_reject_overlaps():
    from formalcrrc.code_extension_answer_forms import paths_for
    class Tokenizer:
        mapping={'A':[1],' A':[1],'\nA':[3,1],'B':[2],' B':[2],'\nB':[3,2]}
        def encode(self,text,**kwargs): return [99]+self.mapping[text[1:]] if text!='p' else [99]
        def decode(self,ids,**kwargs): return ''.join({99:'p',1:'A',2:'B',3:'\n',4:''}[i] for i in ids)
    tok=Tokenizer(); paths,forms=paths_for(tok,'p')
    assert len(paths)==4 and len(forms)==6 and paths[(1,)]=='A'
    tok.mapping=dict(tok.mapping,**{' A':[1,4]})
    with pytest.raises(RuntimeError,match='Overlapping'): paths_for(tok,'p')
