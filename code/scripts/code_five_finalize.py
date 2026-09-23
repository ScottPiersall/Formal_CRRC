"""Final non-mutating dependency verification, source diff and provenance."""
import pathlib,sys,difflib,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from formalcrrc import code_five as f,code_extension as cp
o=f.out(ROOT)
manifest=cp.load(o/'v2_dependency_manifest.json')['files']
cp.assert_manifest(ROOT,manifest)
for model in f.NEW:
    if (o/f'freeze_{model}.json').exists():f.verify_frozen(ROOT,model)
for freeze in o.glob('qwen3_engine_meter*freeze.json'):cp.assert_manifest(ROOT,cp.load(freeze)['files'])
for freeze in o.glob('qwen3_runtime*freeze.json'):
    data=cp.load(freeze);cp.assert_manifest(ROOT,data['files']);cp.assert_manifest(ROOT,data.get('preserved_smoke_traces',{}))
cp.snapshot(o/'final_preservation_audit.json',dict(checked_at=cp.now(),passed=True,dependency_files=len(manifest),
    v2_files_and_shared_sources_match_initial_hashes=True,new_model_freezes_match=True))
paths=list((ROOT/'src/formalcrrc').glob('code_five*.py'))+list((ROOT/'scripts').glob('code_five*.py'))+list((ROOT/'slurm').glob('run_code_five*.sbatch'))+list((ROOT/'tests').glob('test_code_five*.py'))+[ROOT/'docs/CODE_FIVE_MODELS.md']
parts=[]
for p in sorted(paths):
    name=p.relative_to(ROOT).as_posix()
    parts.append('diff --git a/'+name+' b/'+name+'\nnew file mode 100644\n')
    parts.extend(difflib.unified_diff([],p.read_text().splitlines(keepends=True),fromfile='/dev/null',tofile='b/'+name))
(o/'new_code_changes.patch').write_text(''.join(parts),encoding='utf-8')
(o/'git_status_after.txt').write_bytes(subprocess.check_output(['git','status','--short'],cwd=ROOT))
cp.snapshot(o/'new_source_manifest.json',dict(files=cp.manifest(ROOT,paths),created_at=cp.now()))
print('FINAL_DEPENDENCIES_PRESERVED',len(manifest))
