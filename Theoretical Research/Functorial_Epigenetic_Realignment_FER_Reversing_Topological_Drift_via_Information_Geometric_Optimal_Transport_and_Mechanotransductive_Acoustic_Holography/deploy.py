#!/usr/bin/env python3
"""Deploy a FER release into a local clone of The-Invention-Archive.

Usage from the repository root:
    python3 deploy.py /path/to/FER_release

The script creates an isolated directory under:
    Theoretical Research/<Exact paper title>/

It validates the exact 11-file release invariant, runs verify_model.py,
compiles paper.pdf when a LaTeX toolchain is available, patches or creates
Theoretical Research/Readme.md, and optionally creates a semantic Git tag.

External publication to Zenodo, Medium, X, OpenAIRE, or Google Scholar is not
performed because those steps require authenticated external services and/or
an already minted persistent record.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

FILES = [
    'paper.tex', 'paper.pdf', 'verify_model.py', 'README.md', 'medium_story.md',
    'x_thread.md', 'CITATION.cff', 'paper.bib', '.zenodo.json',
    'schema_article.jsonld', 'deploy.py'
]
TITLE_DIR = 'Functorial Epigenetic Realignment (FER): Reversing Topological Drift via Information-Geometric Optimal Transport and Mechanotransductive Acoustic Holography'
VERSION = 'v2026.10.08'

def run(cmd: list[str], cwd: Path) -> None:
    print('+', ' '.join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True, text=True)

def compile_pdf(source: Path) -> None:
    pdflatex = shutil.which('pdflatex')
    bibtex = shutil.which('bibtex') or shutil.which('bibtex.original')
    if not bibtex and Path('/usr/bin/bibtex.original').exists():
        bibtex = '/usr/bin/bibtex.original'
    if not pdflatex:
        print('pdflatex not found; using the existing paper.pdf')
        return
    with tempfile.TemporaryDirectory(prefix='fer-build-') as td:
        build = Path(td)
        for name in ('paper.tex', 'paper.bib'):
            shutil.copy2(source / name, build / name)
        run([pdflatex, '-interaction=nonstopmode', '-halt-on-error', 'paper.tex'], build)
        if bibtex:
            run([bibtex, 'paper'], build)
        run([pdflatex, '-interaction=nonstopmode', '-halt-on-error', 'paper.tex'], build)
        run([pdflatex, '-interaction=nonstopmode', '-halt-on-error', 'paper.tex'], build)
        shutil.copy2(build / 'paper.pdf', source / 'paper.pdf')

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('source', type=Path)
    ap.add_argument('--repo', type=Path, default=Path('.'))
    ap.add_argument('--no-git', action='store_true', help='Skip git add/commit/tag operations.')
    args = ap.parse_args()
    source = args.source.resolve()
    repo = args.repo.resolve()
    target = repo / 'Theoretical Research' / TITLE_DIR
    target.mkdir(parents=True, exist_ok=True)

    present = sorted(p.name for p in source.iterdir() if p.is_file())
    missing = [f for f in FILES if f not in present]
    extras = [f for f in present if f not in FILES]
    if missing or extras:
        raise SystemExit(f'11-file invariant failed; missing={missing}, extras={extras}')

    run(['python3', str(source / 'verify_model.py')], source)
    compile_pdf(source)

    for name in FILES:
        shutil.copy2(source / name, target / name)

    index = repo / 'Theoretical Research' / 'Readme.md'
    entry = f'| 2026.10.08 | {TITLE_DIR} | Geometric-control preprint | `Theoretical Research/{TITLE_DIR}/` |\n'
    if index.exists():
        text = index.read_text(encoding='utf-8')
        if entry not in text:
            if not text.endswith('\n'):
                text += '\n'
            text += entry
    else:
        text = '# Theoretical Research\n\n| Release | Title | Type | Path |\n|---|---|---|---|\n' + entry
    index.write_text(text, encoding='utf-8')

    if args.no_git:
        print(f'Local research release written to: {target}')
        return

    run(['git', 'add', str(target), str(index)], repo)
    diff = subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=repo)
    if diff.returncode != 0:
        run(['git', 'commit', '-m', 'Add FER theoretical research release'], repo)
    else:
        print('No Git content changes detected.')

    tag_check = subprocess.run(['git', 'rev-parse', '-q', '--verify', f'refs/tags/{VERSION}'], cwd=repo, text=True)
    if tag_check.returncode != 0:
        run(['git', 'tag', '-a', VERSION, '-m', 'FER theoretical research release'], repo)
    else:
        print(f'Tag {VERSION} already exists; not recreated.')
    print(f'Release prepared: {target}')

if __name__ == '__main__':
    main()