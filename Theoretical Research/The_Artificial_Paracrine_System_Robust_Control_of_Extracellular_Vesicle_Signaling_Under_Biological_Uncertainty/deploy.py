#!/usr/bin/env python3
"""Validate the APS release, rebuild its PDF, and optionally commit it to GitHub.

Run from a clean clone with:
  python deploy.py --repo-root /path/to/The-Invention-Archive

The source directory may be outside the target repository. The script never force-pushes,
rewrites history, or deletes unrelated files. Dependencies: Python >=3.10, NumPy >=1.24,
and pdflatex for a fresh PDF build. Git credentials must be configured outside this script.
"""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CORE = {
    'paper.tex', 'paper.pdf', 'verify_model.py', 'README.md', 'medium_story.md',
    'x_thread.md', 'CITATION.cff', 'paper.bib', '.zenodo.json',
    'schema_article.jsonld', 'deploy.py'
}
RELEASE_PARENT = Path('Theoretical Research')
INDEX_NAME = 'Readme.md'

def run(cmd, cwd: Path, capture: bool = True):
    return subprocess.run(cmd, cwd=cwd, check=True, text=True,
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.STDOUT if capture else None)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo-root', type=Path, default=None,
                    help='clean local clone; omit to perform local validation only')
    ap.add_argument('--no-push', action='store_true', help='prepare local files but do not push')
    args = ap.parse_args()
    source = Path(__file__).resolve().parent
    present = {p.name for p in source.iterdir() if p.is_file()}
    if present != CORE:
        raise SystemExit(f'Manifest mismatch: missing={sorted(CORE-present)}; extra={sorted(present-CORE)}')
    json.loads((source/'.zenodo.json').read_text(encoding='utf-8'))
    json.loads((source/'schema_article.jsonld').read_text(encoding='utf-8'))
    run([sys.executable, str(source/'verify_model.py')], source)

    if not shutil.which('pdflatex'):
        raise SystemExit('pdflatex is required to rebuild paper.pdf from paper.tex')
    with tempfile.TemporaryDirectory(prefix='aps-build-') as td:
        tmp = Path(td)
        shutil.copy2(source/'paper.tex', tmp/'paper.tex')
        for _ in range(2):
            result = run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'paper.tex'], tmp)
            log = tmp/'paper.log'
            log_text = log.read_text(errors='replace') if log.exists() else result.stdout
            if 'Undefined control sequence' in log_text or 'LaTeX Error:' in log_text:
                raise SystemExit('LaTeX log contains a fatal error')
        fresh_pdf = tmp/'paper.pdf'
        if not fresh_pdf.is_file() or fresh_pdf.stat().st_size < 1000:
            raise SystemExit('Fresh PDF is missing or implausibly small')
        shutil.copy2(fresh_pdf, source/'paper.pdf')

    if args.repo_root is None:
        print('PASS: local manifest, JSON, numerical checks, and fresh PDF compilation.')
        print('No remote deployment requested; pass --repo-root to deploy from a clean clone.')
        return

    repo = args.repo_root.resolve()
    if not (repo/'.git').exists():
        raise SystemExit(f'Not a Git clone: {repo}')
    if run(['git', 'status', '--porcelain'], repo).stdout.strip():
        raise SystemExit('Repository has local changes. Commit/stash them before deployment; none were altered.')
    branch = run(['git', 'branch', '--show-current'], repo).stdout.strip()
    if not branch:
        raise SystemExit('Detached HEAD is not supported for deployment')
    run(['git', 'fetch', 'origin', branch, '--tags'], repo)
    local_head = run(['git', 'rev-parse', 'HEAD'], repo).stdout.strip()
    remote_head = run(['git', 'rev-parse', f'origin/{branch}'], repo).stdout.strip()
    if local_head != remote_head:
        raise SystemExit(f'Local branch is not up-to-date with origin/{branch}; fetch/reconcile first')

    target = repo/RELEASE_PARENT/source.name
    if target.exists():
        raise SystemExit(f'Release directory already exists; refusing overwrite: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    index = repo/RELEASE_PARENT/INDEX_NAME
    existing = index.read_text(encoding='utf-8') if index.exists() else '# Theoretical Research\n\n| Release | Title | Type | Path |\n|---|---|---|---|\n'
    rel = source.name
    if rel in existing:
        raise SystemExit('Index already contains this release path; inspect for a collision')
    entry = f'| 2026.10.10 | The Artificial Paracrine System | Robust-control preprint | [{rel}]({rel}/) |'
    index.write_text(existing.rstrip()+'\n'+entry+'\n', encoding='utf-8')
    run(['git', 'add', '--', str(target.relative_to(repo)), str(index.relative_to(repo))], repo)
    run(['git', 'commit', '-m', 'Add Artificial Paracrine System research release'], repo)
    if not args.no_push:
        run(['git', 'push', 'origin', branch], repo)
    print(f'Committed release on {branch}: {target.relative_to(repo)}')
    print('Verify the remote branch and every artifact before reporting the release as published.')

if __name__ == '__main__':
    main()
