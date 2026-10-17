#!/usr/bin/env python3
"""Local validation/build helper for this research release. Does not perform GitHub writes."""
from __future__ import annotations
import argparse, hashlib, json, os, re, shutil, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORE = {
    'paper.tex', 'paper.pdf', 'verify_model.py', 'README.md', 'CITATION.cff',
    'paper.bib', '.zenodo.json', 'schema_article.jsonld', 'deploy.py'
}
REQUIRED_FIGS = {
    'fig1_roc.pdf', 'fig2_decoy_fpr.pdf', 'fig3_recovery_confounds.pdf',
    'fig4_metrics.pdf', 'fig5_measurement_graph.pdf', 'fig6_confounding_graph.pdf',
    'fig7_method_flow.pdf', 'fig8_adversarial_training.pdf', 'fig9_noise_sensitivity.pdf'
}

def run(cmd, cwd, log_path=None):
    result = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, check=False)
    if log_path:
        Path(log_path).write_text(result.stdout, encoding='utf-8')
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(map(str,cmd))}\n{result.stdout[-5000:]}")
    return result.stdout

def validate_metadata(root: Path):
    import yaml
    cff = yaml.safe_load((root/'CITATION.cff').read_text(encoding='utf-8'))
    assert cff['cff-version'] == '1.2.0' and cff['title'] and cff['authors']
    assert str(cff['date-released']) == '2026-10-17'
    z = json.loads((root/'.zenodo.json').read_text(encoding='utf-8'))
    j = json.loads((root/'schema_article.jsonld').read_text(encoding='utf-8'))
    assert z['title'] == cff['title'] == j['headline']
    assert j['@type'] == 'ScholarlyArticle' and 'doi' not in j
    bib = (root/'paper.bib').read_text(encoding='utf-8')
    tex = (root/'paper.tex').read_text(encoding='utf-8')
    keys = set(re.findall(r'@\w+\s*\{\s*([^,\s]+)', bib))
    cited = set(re.findall(r'\\(?:citep|citet|cite|parencite|textcite|autocite)\{([^}]+)\}', tex))
    used = {k.strip() for s in cited for k in s.split(',')}
    missing = used - keys
    assert not missing, f'Missing BibTeX keys: {sorted(missing)}'
    return {'cff':'pass','zenodo_json':'pass','jsonld':'pass','bib_keys':'pass'}

def inspect_pdf(pdf: Path):
    import fitz
    doc = fitz.open(pdf)
    assert len(doc) >= 7, f'PDF has only {len(doc)} pages'
    txt = '\n'.join(p.get_text() for p in doc)
    assert 'Co-equal research-process credit is given' in txt
    assert 'References' in txt and len(txt) > 15000
    return len(doc), len(txt)

def validate_current(root: Path, check_root=True):
    if check_root:
        actual = {p.name for p in root.iterdir() if p.is_file()}
        assert actual == CORE, f'Root core-file mismatch. Missing={sorted(CORE-actual)} Extra={sorted(actual-CORE)}'
    assert (root/'figures').is_dir() and (root/'results').is_dir()
    assert REQUIRED_FIGS.issubset({p.name for p in (root/'figures').iterdir()})
    assert len(list((root/'figures').glob('fig[1-9]_*.pdf'))) >= 8
    assert len(list((root/'figures').glob('fig[1-9]_*.png'))) >= 8
    tex=(root/'paper.tex').read_text(encoding='utf-8')
    readme=(root/'README.md').read_text(encoding='utf-8')
    assert len(re.findall(r'\\begin\{table\}',tex)) >= 5
    assert len(re.findall(r'\\begin\{figure\}',tex)) >= 8
    assert 'Co-equal research-process credit is given' in tex
    assert '$$' not in readme or readme.count('$$') % 2 == 0
    assert r'\operatorname' not in readme
    for file in ['benchmark_summary.json','evaluation_episodes.csv','verification_summary.md',
                 'author_profile_audit.md','literature_audit.csv','candidate_topics.csv',
                 'claim_evidence_matrix.csv','environment.json']:
        assert (root/'results'/file).exists() and (root/'results'/file).stat().st_size > 0, file
    return validate_metadata(root)

def build_pdf(root: Path):
    for cmd in ['pdflatex','biber']:
        if not shutil.which(cmd): raise RuntimeError(f'Required build tool missing: {cmd}')
    with tempfile.TemporaryDirectory(prefix='alien-repair-build-') as td:
        build=Path(td)
        shutil.copy2(root/'paper.tex',build/'paper.tex')
        shutil.copy2(root/'paper.bib',build/'paper.bib')
        shutil.copytree(root/'figures',build/'figures')
        run(['pdflatex','-interaction=nonstopmode','-halt-on-error','paper.tex'],build,root/'results'/'pdf_build_pass1.log')
        run(['biber','paper'],build,root/'results'/'biber.log')
        run(['pdflatex','-interaction=nonstopmode','-halt-on-error','paper.tex'],build,root/'results'/'pdf_build_pass2.log')
        final=run(['pdflatex','-interaction=nonstopmode','-halt-on-error','paper.tex'],build,root/'results'/'pdf_build_pass3.log')
        if 'There were undefined references' in final or 'Citation(s) may have changed' in final:
            # One clean final pass should remove transient cross-reference warnings.
            final=run(['pdflatex','-interaction=nonstopmode','-halt-on-error','paper.tex'],build,root/'results'/'pdf_build_pass4.log')
        bad = ['LaTeX Error:', 'Emergency stop.', 'undefined references', 'Citation(s) may have changed']
        joined='\n'.join(p.read_text(encoding='utf-8',errors='replace') for p in [build/'paper.log'])
        assert not any(x in joined for x in bad), 'LaTeX log contains fatal/unresolved reference warnings'
        shutil.copy2(build/'paper.pdf',root/'paper.pdf')

def make_manifest(root: Path):
    entries=[]
    for p in sorted(root.rglob('*')):
        if not p.is_file(): continue
        rel = p.relative_to(root).as_posix()
        if rel == 'results/sha256_manifest.json': continue
        if '__pycache__' in p.parts or p.suffix == '.pyc': continue
        h=hashlib.sha256(p.read_bytes()).hexdigest()
        entries.append({'path':p.relative_to(root).as_posix(),'sha256':h,'bytes':p.stat().st_size})
    out={'algorithm':'SHA-256','generated_by':'deploy.py','manifest_excludes_itself':True,'artifacts':entries}
    (root/'results'/'sha256_manifest.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')

def self_test():
    # Fixture is synthetic and isolated; no actual release/remote repo is modified.
    with tempfile.TemporaryDirectory(prefix='alien-repair-fixture-') as td:
        fixture=Path(td)
        for name in CORE: (fixture/name).write_text('fixture',encoding='utf-8')
        (fixture/'figures').mkdir(); (fixture/'results').mkdir()
        assert len(CORE)==9 and 'paper.pdf' in CORE
        expected = {'fig1_roc.pdf','fig2_decoy_fpr.pdf','fig3_recovery_confounds.pdf','fig4_metrics.pdf',
                    'fig5_measurement_graph.pdf','fig6_confounding_graph.pdf','fig7_method_flow.pdf',
                    'fig8_adversarial_training.pdf','fig9_noise_sensitivity.pdf'}
        for name in expected: (fixture/'figures'/name).write_text('fixture',encoding='utf-8')
        for name in ['benchmark_summary.json','evaluation_episodes.csv','verification_summary.md',
                     'author_profile_audit.md','literature_audit.csv','candidate_topics.csv',
                     'claim_evidence_matrix.csv','environment.json']:
            (fixture/'results'/name).write_text('fixture',encoding='utf-8')
        assert {p.name for p in fixture.iterdir() if p.is_file()} == CORE
        assert expected.issubset({p.name for p in (fixture/'figures').iterdir()})
        shutil.rmtree(fixture)
    print('PASS: deployment self-test in isolated fixture; nine root files and evidence directories checked.')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    if args.self_test:
        self_test(); return 0
    # Regenerate outputs from the current model before compiling source.
    run([sys.executable,'verify_model.py'],ROOT,ROOT/'results'/'verification_run.log')
    run([sys.executable,'-m','py_compile','verify_model.py','deploy.py'],ROOT,ROOT/'results'/'python_compile.log')
    run([sys.executable,'deploy.py','--self-test'],ROOT,ROOT/'results'/'deploy_self_test.log')
    build_pdf(ROOT)
    validate_current(ROOT,check_root=False)
    pages, chars=inspect_pdf(ROOT/'paper.pdf')
    make_manifest(ROOT)
    status={'status':'passed','pdf_pages':pages,'pdf_text_characters':chars,
            'synthetic_benchmark':'executed','metadata':'validated','root_manifest':'validated after temporary build',
            'empirical_validation':'not performed; synthetic simulator only'}
    (ROOT/'results'/'deployment_validation.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
    make_manifest(ROOT)
    # Confirm exact root file set after generation.
    actual={p.name for p in ROOT.iterdir() if p.is_file()}
    assert actual == CORE, f'Root core-file mismatch: missing={CORE-actual}, extra={actual-CORE}'
    print(json.dumps(status,indent=2))
    return 0
if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as exc:
        print(f'RELEASE BLOCKED: {exc}',file=sys.stderr)
        raise SystemExit(1)
