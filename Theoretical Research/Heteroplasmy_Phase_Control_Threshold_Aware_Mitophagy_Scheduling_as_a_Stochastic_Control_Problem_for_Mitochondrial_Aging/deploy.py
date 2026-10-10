#!/usr/bin/env python3
"""Validate, rebuild, index, commit, tag, push, and verify this research release.

Run from a clean clone of Devanik21/The-Invention-Archive:
  python "Theoretical Research/Heteroplasmy_Phase_Control_Threshold_Aware_Mitophagy_Scheduling_as_a_Stochastic_Control_Problem_for_Mitochondrial_Aging/deploy.py"

The script embeds no credentials, stages only this release folder plus the parent index,
and refuses an existing target path or duplicate release-index entry. External publication
and DOI minting are deliberately outside the script.
"""
from __future__ import annotations
import argparse, datetime as dt, json, re, shutil, subprocess, sys, tempfile
from pathlib import Path

EXPECTED = {"paper.tex", "paper.pdf", "verify_model.py", "README.md", "medium_story.md", "x_thread.md", "CITATION.cff", "paper.bib", ".zenodo.json", "schema_article.jsonld", "deploy.py"}
RELATIVE_DIR = Path("Theoretical Research/Heteroplasmy_Phase_Control_Threshold_Aware_Mitophagy_Scheduling_as_a_Stochastic_Control_Problem_for_Mitochondrial_Aging")
INDEX = Path("Theoretical Research/Readme.md")

def run(cmd, cwd=None, capture=False, check=True):
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=capture, check=check)

def git(root: Path, *args, capture=True, check=True):
    return run(["git", *args], cwd=root, capture=capture, check=check)

def die(msg: str):
    raise SystemExit(f"ERROR: {msg}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="release tag; defaults to vYYYY.MM.DD and adds a suffix if already present")
    parser.add_argument("--skip-push", action="store_true", help="validate/build locally without committing or pushing")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = Path(git(here, "rev-parse", "--show-toplevel").stdout.strip()).resolve()
    if (root / RELATIVE_DIR).resolve() != here:
        die(f"expected script under {RELATIVE_DIR}")
    present = {p.relative_to(here).as_posix() for p in here.rglob("*") if p.is_file()}
    if present != EXPECTED:
        die(f"file manifest mismatch: missing={sorted(EXPECTED-present)}, extra={sorted(present-EXPECTED)}")
    try:
        import numpy as np
    except ImportError:
        die("NumPy >=1.24 is required")
    run([sys.executable, str(here / "verify_model.py")], cwd=here)
    json.loads((here / ".zenodo.json").read_text(encoding="utf-8"))
    json.loads((here / "schema_article.jsonld").read_text(encoding="utf-8"))
    cff=(here / "CITATION.cff").read_text(encoding="utf-8")
    if "cff-version: 1.2.0" not in cff or "authors:" not in cff: die("CITATION.cff structural check failed")
    words=re.findall(r"\b[\w'-]+\b", (here / "medium_story.md").read_text(encoding="utf-8"))
    if not 1500 <= len(words) <= 2000: die(f"Medium draft has {len(words)} words; expected 1500-2000")
    posts=re.findall(r"(?m)^\*\*\d+/12\*\*", (here / "x_thread.md").read_text(encoding="utf-8"))
    if len(posts) != 12: die(f"X thread has {len(posts)} numbered posts, expected 12")
    readme=(here / "README.md").read_text(encoding="utf-8")
    if len(re.findall(r"(?m)^\$\$$", readme)) != 6: die("README must have three display-equation blocks")
    if r"\n" in readme: die("README contains a literal escaped newline sequence")
    if shutil.which("pdflatex") is None: die("pdflatex is unavailable")
    with tempfile.TemporaryDirectory(prefix="heteroplasmy-build-") as tmp:
        tmpdir=Path(tmp)
        shutil.copy2(here / "paper.tex", tmpdir / "paper.tex")
        for _ in range(2):
            p=run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "paper.tex"], cwd=tmpdir, capture=True, check=False)
            if p.returncode != 0:
                print(p.stdout[-8000:]); print(p.stderr[-2000:]); die("LaTeX compilation failed")
        pdf=tmpdir / "paper.pdf"
        if not pdf.is_file() or not pdf.read_bytes().startswith(b"%PDF-") or pdf.stat().st_size < 1000: die("compiled PDF validation failed")
        shutil.copy2(pdf, here / "paper.pdf")
    print(f"PASS: files=11; medium_words={len(words)}; x_posts={len(posts)}; NumPy={np.__version__}; PDF rebuilt")
    if args.skip_push:
        print("LOCAL VALIDATION ONLY: no commit or push")
        return
    branch=git(root, "branch", "--show-current").stdout.strip()
    if not branch: die("detached HEAD")
    status=git(root, "status", "--porcelain").stdout.splitlines()
    allowed={str(INDEX).replace("\\", "/"), str(here.relative_to(root)).replace("\\", "/")}
    for line in status:
        path=line[3:].strip().strip('"').replace("\\", "/")
        if path not in allowed and not any(path.startswith(a + "/") for a in allowed):
            die(f"unrelated worktree change; refusing to stage: {line}")
    run(["git", "fetch", "origin", branch, "--tags"], cwd=root)
    idx=root/INDEX
    if not idx.exists(): die(f"missing parent index: {INDEX}")
    old=idx.read_text(encoding="utf-8")
    slug=here.name
    if slug in old: die("release slug already appears in parent index")
    title="Heteroplasmy Phase Control"
    row=f"| {dt.date.today().strftime('%Y.%m.%d')} | {title} | Stochastic-control preprint | [{slug}]({slug}/) |"
    idx.write_text(old.rstrip()+"\n"+row+"\n", encoding="utf-8")
    if args.tag:
        tag=args.tag
    else:
        base="v"+dt.date.today().strftime("%Y.%m.%d")
        tag=base
        suffix=1
        while git(root,"ls-remote","--exit-code","--tags","origin",f"refs/tags/{tag}",capture=True,check=False).returncode == 0:
            suffix += 1; tag=f"{base}-{suffix:02d}"
    if git(root,"ls-remote","--exit-code","--tags","origin",f"refs/tags/{tag}",capture=True,check=False).returncode == 0: die(f"tag already exists: {tag}")
    # Stage precisely the release and its parent index.
    git(root,"add","--",str(rel:=here.relative_to(root)),str(INDEX))
    git(root,"diff","--cached","--check")
    git(root,"commit","-m","Add heteroplasmy phase-control research release",capture=False)
    git(root,"tag","-a",tag,"-m",f"Heteroplasmy phase-control theoretical preprint {tag}",capture=False)
    git(root,"push","origin",f"HEAD:{branch}",capture=False)
    git(root,"push","origin",f"refs/tags/{tag}",capture=False)
    run(["git","fetch","origin",branch,"--tags"],cwd=root)
    head=git(root,"rev-parse","HEAD").stdout.strip(); remote=git(root,"rev-parse",f"origin/{branch}").stdout.strip()
    if head != remote: die("remote main head does not match local release commit")
    files=git(root,"ls-tree","-r","--name-only",f"origin/{branch}","--",str(rel)).stdout.splitlines()
    if len(files)!=11: die(f"remote release file count mismatch: {len(files)}")
    for name in sorted(EXPECTED):
        local=git(root,"hash-object",f"{rel}/{name}").stdout.strip()
        remote_sha=git(root,"rev-parse",f"origin/{branch}:{rel}/{name}").stdout.strip()
        if local!=remote_sha: die(f"remote blob mismatch: {name}")
    print(f"PASS: remote commit={head}; tag={tag}; file_count=11; every release blob verified")

if __name__ == "__main__":
    main()
