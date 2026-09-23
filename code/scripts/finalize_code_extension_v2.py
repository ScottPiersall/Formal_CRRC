# Review adaptation: historical non-English narrative strings omitted; numerical routines retained.
"""Post-inference interpretation and portable packaging; no model calls.

Added after freeze A. Frozen experimental source and raw records stay unchanged.
"""
import argparse, difflib, json, pathlib, subprocess, sys, zipfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
from formalcrrc import code_extension as cp
from formalcrrc import code_extension_analysis as analysis
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = cp.outdir(ROOT)

def enhance_report():
    """Add checked interpretation without recomputing or changing any metrics."""
    path = OUT / 'analysis/INTERPRETATION_ZH.md'
    text = path.read_text(encoding='utf-8')
    if '<!-- actual-findings -->' in text:
        return
    s = cp.load(OUT / 'analysis/summary.json')
    u = s['answer_form_robustness']
    n = s['n_main']
    bare = s['summaries']['mistral/original/bfloat16']['all']['k0_correct']['estimate']
    union = u['summaries']['mistral/original']['all']['k0_correct']['estimate']
    unchanged = all((s['summaries'][f'{m}/{v}/bfloat16']['partial']['fsrr']['estimate'] == u['summaries'][f'{m}/{v}']['partial']['fsrr']['estimate'] for m in cp.MODELS for v in cp.VARIANTS))
    intro = ['Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.' + ('Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.' if unchanged else 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.')]
    rows = cp.load(OUT / 'analysis/per_instance_metrics.json')
    example = {r['variant']: r for r in rows if r['model_key'] == 'qwen' and r['task_id'] == 'HumanEval/101' and (r['dtype'] == 'bfloat16')}
    a, b = (example['original'], example['explicit'])
    if a['errors'] == b['errors'] and a['fsrr'] is False and (b['fsrr'] is True):
        intro.append('Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.')
    intro.append('Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.')
    intro.append('Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.')
    title, rest = text.split('\n', 1)
    path.write_text(title + '\n\n<!-- actual-findings -->\n\n' + '\n\n'.join(intro) + '\n' + rest, encoding='utf-8')

def supplement():
    status = cp.load(OUT / 'execution_status.json')
    assert status['status'] == 'COMPLETE'
    cp.assert_freeze_a(ROOT)
    cp.assert_freeze_b(ROOT)
    review = cp.load(OUT / 'development_review.json')
    assert review['decision'] == 'PROCEED'
    assert review['development_summary_sha256'] == cp.sha((OUT / 'development_summary.json').read_bytes())
    assert review['created_at'] <= cp.load(OUT / 'freeze_A.json')['created_at']
    baseline = cp.load(OUT / 'prior_v1_baseline.json')['files']
    present = {p: h for p, h in baseline.items() if (ROOT / p).exists()}
    cp.assert_manifest(ROOT, present)
    if (ROOT / '.git').exists():
        assert len(present) == len(baseline)
    s = cp.load(OUT / 'analysis/summary.json')
    u = s['answer_form_robustness']
    panels = {'bare_bfloat16': [r for r in cp.load(OUT / 'analysis/per_instance_metrics.json') if r['dtype'] == 'bfloat16'], 'bare_float32': [r for r in cp.load(OUT / 'analysis/per_instance_metrics.json') if r['dtype'] == 'float32'], 'union_bfloat16': cp.load(OUT / 'analysis/answer_form_metrics.json'), 'union_float32': cp.load(OUT / 'analysis/answer_form_float32_metrics.json')}
    index = {name: {(r['model_key'], r['variant'], r['task_id']): r for r in rows} for name, rows in panels.items()}
    tasks = [i['task_id'] for i in cp.load(OUT / 'instances.json') if i['split'] == 'main']
    partial = [t for t in tasks if 1 <= cp.load(OUT / 'truth' / f'{cp.slug(t)}.json')['z'] <= 7]
    numeric = set(cp.load(OUT / 'numeric_task_ids.json'))
    cross = {}
    for key in cp.MODELS:
        forms = ('bare_bfloat16', 'union_bfloat16')
        both = [t for t in partial if all((index[name][key, v, t]['fsrr'] is False for name in forms for v in cp.VARIANTS))]
        strict = [t for t in partial if all((index[name][key, v, t]['strict_fsrr'] is False for name in forms for v in cp.VARIANTS))]
        robust = [t for t in partial if t in numeric and all((index[name][key, v, t]['fsrr_gap'] < -0.01 for name in panels for v in cp.VARIANTS))]
        union_both = [t for t in partial if all((index['union_bfloat16'][key, v, t]['fsrr'] is False for v in cp.VARIANTS))]
        union_numeric = [t for t in partial if t in numeric and all((index[name][key, v, t]['fsrr_gap'] < -0.01 for name in ('union_bfloat16', 'union_float32') for v in cp.VARIANTS))]
        cross[key] = dict(partial_n=len(partial), numerical_partial_n=len(set(partial) & numeric), both_representations_both_templates_failure_ids=both, both_representations_both_templates_without_k0_failure_ids=strict, all_representations_templates_precisions_gap_below_minus_001_ids=robust, union_both_templates_failure_ids=union_both, union_both_templates_both_precisions_gap_below_minus_001_ids=union_numeric)
    checked = 0
    for key in cp.MODELS:
        manifest = cp.load(OUT / 'prompts' / f'{key}__answer_tokens.json')
        tokens = {(r['task_id'], r['variant'], r['k']): r for r in manifest['rows']}
        for category in ('scores_answer_forms', 'scores_answer_forms_float32'):
            for p in (OUT / category).rglob('*.json'):
                r = cp.load(p)
                if r['model_key'] != key:
                    continue
                frozen = tokens[r['task_id'], r['variant'], r['k']]
                assert r['rendered_prompt_sha256'] == frozen['rendered_prompt_sha256']
                assert {label: e['token_ids'] for label, e in r['answer_events'].items()} == {label: e['token_ids'] for label, e in frozen['answer_forms'].items()}
                checked += 1
    assert checked == 3348
    result = dict(created_at=cp.now(), passed=True, verified_frozen_reply_paths=checked, prior_v1_files_present=len(present), prior_v1_files_in_baseline=len(baseline), full_v1_preservation_verified_here=len(present) == len(baseline), cross_factor_descriptive=cross, note='Post-inference descriptive intersections of predeclared factors; not an additional confirmatory hypothesis test. Portable bundle checks included v1 dependency files; full v1 preservation is verified in the original repository.')
    cp.snapshot(OUT / 'post_inference_audit.json', result)
    lines = ['Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.', 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.', 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.', '|---|---|---|---|---|']
    for key in cp.MODELS:
        for v in cp.VARIANTS:
            bare = s['summaries'][f'{key}/{v}/bfloat16']['partial']
            union = u['summaries'][f'{key}/{v}']['partial']
            lines.append('| ' + key + ' | ' + v + ' | ' + ' | '.join((analysis.fmt(x) for x in (bare['fsrr'], union['fsrr'], union['strict_fsrr']))) + ' |')
    lines += ['Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.', 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.', 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.', 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.']
    for key, r in cross.items():
        lines += ['Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.', 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.']
    lines += ['Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.', 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.', 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.']
    (OUT / 'analysis/INTERPRETATION_ZH.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))

def implementation_patch():
    names = list((ROOT / 'src/formalcrrc').glob('code_extension*.py')) + list((ROOT / 'scripts').glob('*code_extension*.py'))
    names += list((ROOT / 'slurm').glob('run_code_extension*.sbatch')) + [ROOT / 'tests/test_code_extension.py', ROOT / 'docs/CODE_EXTENSION_V2.md']
    pieces = []
    for p in sorted(set(names)):
        name = p.relative_to(ROOT).as_posix()
        previous = subprocess.run(['git', 'show', 'HEAD:' + name], cwd=ROOT, capture_output=True)
        old = previous.stdout.decode().splitlines(keepends=True) if previous.returncode == 0 else []
        pieces.append(''.join(difflib.unified_diff(old, p.read_text().splitlines(keepends=True), fromfile='a/' + name if old else '/dev/null', tofile='b/' + name)))
    (OUT / 'implementation.patch').write_text(''.join(pieces), encoding='utf-8')

def package(reuse_verified_archive=False):
    enhance_report()
    implementation_patch()
    target = ROOT.parent / 'FormalCRRC_Code_Extension_v2_Handoff.zip'
    if not reuse_verified_archive:
        analysis.package(ROOT)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        payload = {n: z.read(n) for n in z.namelist()}
    if reuse_verified_archive:
        cp.assert_freeze_a(ROOT)
        cp.assert_freeze_b(ROOT)
        previous = json.loads(payload['archive_manifest.json'])
        assert all((cp.sha(payload[n]) == h for n, h in previous['files'].items()))
        updates = [OUT / 'implementation.patch', OUT / 'portable_validation.json'] + list((OUT / 'analysis').glob('*'))
        for p in updates:
            if p.is_file():
                payload[p.relative_to(ROOT).as_posix()] = p.read_bytes()
    payload.pop('archive_manifest.json', None)
    for name in ['scripts/finalize_code_extension_v2.py', 'scripts/publish_code_extension_reply_tokens.py', 'docs/CODE_PILOT_V1.md', 'FormalCRRC_Code_Pilot_Codex_Task.md']:
        payload[name] = (ROOT / name).read_bytes()
    payload['requirements-analysis.txt'] = b'numpy==2.2.6\npandas==2.2.3\nscipy==1.15.2\nmatplotlib==3.10.1\npytest==8.3.5\ntokenizers==0.23.2\n'
    payload['README.md'] = '# FormalCRRC Code Extension v2 handoff\n\nStart with `artifacts/code_extension_v2/analysis/INTERPRETATION_ZH.md`,\n`REPORT_ZH.md`, `METHODS_RESULTS_EN.md` in the same directory, and\n`artifacts/code_extension_v2/execution_status.json`.\n\nThe archive contains the independent 80-task experiment, separate 30-task\ndevelopment diagnostics, all raw generation/truth/prompts/scores, freezes,\nfailure records, public tokenizer metadata, implementation patch and tests.\nModel weights and credentials are not included. Python 3.12.3 was used locally.\n\n```bash\npython -m pip install -r requirements-analysis.txt\npython -m pytest tests/test_code_extension.py tests/test_code_pilot.py tests/test_scoring.py\npython scripts/code_extension.py analyze\n```\n\nReanalysis uses only archived evidence and the pinned local scientific packages.\nIt does not call models or execute candidate programs. It regenerates derived\nreports/status timestamps; compare numerical fields, not timestamps. The full\noriginal v1 artifact tree is separate: its preservation was checked in the\noriginal repository, while portable analysis verifies included dependencies\nand archived development evidence. The prior-v1 checksum inventory is retained.\n\nFor actual inference/truth reproduction, see `docs/CODE_EXTENSION_V2.md`:\npinned model caches, the recorded GPU software, CLUSTER authorization, and the\npinned Docker sandbox are additionally required. Publish locally prepared reply\nmanifests with `python scripts/publish_code_extension_reply_tokens.py` before\nsyncing freeze B; the frozen bulk sync excludes remote-owned prompt files.\nDo not submit duplicate jobs or regenerate already completed evidence.\n\nThe extension patch contains extension implementation files only; inherited\npilot dependencies are included as source snapshots. Preexisting unrelated\nworkspace edits were preserved. `archive_manifest.json` hashes every other zip\nmember; the adjacent `.zip.sha256` hashes the entire archive.\n'.encode()
    manifest_name = 'artifacts/code_extension_v2/handoff_manifest.json'
    m = cp.load(OUT / 'handoff_manifest.json')
    m['created_at'] = cp.now()
    m['files'] = {n: cp.sha(data) for n, data in sorted(payload.items()) if n != manifest_name}
    payload[manifest_name] = cp.canonical(m).encode()
    (OUT / 'handoff_manifest.json').write_bytes(payload[manifest_name])
    payload['archive_manifest.json'] = cp.canonical(dict(created_at=cp.now(), files={n: cp.sha(data) for n, data in sorted(payload.items())})).encode()
    temp = target.with_suffix('.building.zip')
    with zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name, data in sorted(payload.items()):
            z.writestr(name, data)
    temp.replace(target)
    with zipfile.ZipFile(target) as z:
        assert len(z.namelist()) == len(set(z.namelist())) and z.testzip() is None
        m = json.loads(z.read('archive_manifest.json'))
        assert all((cp.sha(z.read(n)) == h for n, h in m['files'].items()))
        assert not any((n.endswith(('.safetensors', '.pt', '.pem', '.key')) for n in z.namelist()))
    digest = cp.sha(target.read_bytes())
    target.with_suffix('.zip.sha256').write_text(f'{digest}  {target.name}\n')
    print('FINAL_HANDOFF', target, 'SHA256', digest, 'FILES', len(payload))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--package-only', action='store_true')
    args = parser.parse_args()
    if not args.package_only:
        supplement()
    package(reuse_verified_archive=args.package_only and (ROOT.parent / 'FormalCRRC_Code_Extension_v2_Handoff.zip').exists())
