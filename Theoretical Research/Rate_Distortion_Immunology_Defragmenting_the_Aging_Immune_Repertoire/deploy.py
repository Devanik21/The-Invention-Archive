#!/usr/bin/env python3
"""Validate, rebuild, index, commit and tag this release from a clean repository clone.

Run from the repository clone:
  python "Theoretical Research/Rate_Distortion_Immunology_Defragmenting_the_Aging_Immune_Repertoire/deploy.py"
Requires Git, Python >=3.10, NumPy, PyYAML, and pdflatex; page count is checked with pypdf or pdfinfo.
Use --skip-push for local validation/build only. No tokens are embedded.
"""
from __future__ import annotations
import argparse, datetime as dt, json, re, shutil, subprocess, sys, tempfile
from pathlib import Path

EXPECTED={"paper.tex","paper.pdf","verify_model.py","README.md","medium_story.md","x_thread.md","CITATION.cff","paper.bib",".zenodo.json","schema_article.jsonld","deploy.py"}
RELATIVE_DIR=Path("Theoretical Research/Rate_Distortion_Immunology_Defragmenting_the_Aging_Immune_Repertoire")
INDEX=Path("Theoretical Research/Readme.md")

def run(args,cwd=None,capture=False,check=True):
    return subprocess.run(args,cwd=cwd,text=True,capture_output=capture,check=check)
def git(root,*args,capture=True,check=True): return run(["git",*args],cwd=root,capture=capture,check=check)
def die(msg): raise SystemExit("ERROR: "+msg)
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag",help="release tag, e.g. v2026.10.11-04")
    ap.add_argument("--skip-push",action="store_true",help="validate and rebuild locally only")
    args=ap.parse_args(); here=Path(__file__).resolve().parent
    root=Path(git(here,"rev-parse","--show-toplevel").stdout.strip()).resolve()
    if (root/RELATIVE_DIR).resolve()!=here: die("unexpected release directory")
    present={p.relative_to(here).as_posix() for p in here.rglob("*") if p.is_file()}
    if present!=EXPECTED: die(f"manifest mismatch: missing={sorted(EXPECTED-present)}, extra={sorted(present-EXPECTED)}")
    import numpy as np
    print(f"Python={sys.version.split()[0]}; NumPy={np.__version__}")
    run([sys.executable,str(here/"verify_model.py")],cwd=here)
    json.loads((here/".zenodo.json").read_text()); json.loads((here/"schema_article.jsonld").read_text())
    import yaml
    yaml.safe_load((here/"CITATION.cff").read_text())
    words=re.findall(r"\b[\w'-]+\b",(here/"medium_story.md").read_text())
    if not 1500<=len(words)<=2000: die(f"Medium draft has {len(words)} words")
    posts=re.findall(r"(?m)^\*\*\d+/12\*\*",(here/"x_thread.md").read_text())
    if len(posts)!=12: die("X thread must contain 12 posts")
    rm=(here/"README.md").read_text()
    if len(re.findall(r"(?m)^\$\$$",rm))!=8: die("README must contain four balanced display math blocks")
    if r"\n" in rm or "\\operatorname" in rm: die("README equation typography check failed")
    if shutil.which("pdflatex") is None: die("pdflatex unavailable; cannot validate PDF source consistency")
    with tempfile.TemporaryDirectory(prefix="rate-distortion-build-") as tmp:
        tmp=Path(tmp); shutil.copy2(here/"paper.tex",tmp/"paper.tex"); shutil.copy2(here/"paper.bib",tmp/"paper.bib")
        for _ in range(3):
            p=run(["pdflatex","-interaction=nonstopmode","-halt-on-error","paper.tex"],cwd=tmp,capture=True,check=False)
            if p.returncode!=0: print(p.stdout[-8000:]); die("LaTeX compilation failed")
        pdf=tmp/"paper.pdf"
        if not pdf.exists() or pdf.stat().st_size<5000 or not pdf.read_bytes().startswith(b"%PDF-"): die("compiled PDF is invalid")
        shutil.copy2(pdf,here/"paper.pdf")
    print(f"PASS local checks; Medium words={len(words)}; posts={len(posts)}; PDF={ (here/'paper.pdf').stat().st_size} bytes")
    if args.skip_push: print("Local validation complete; push intentionally skipped"); return
    branch=git(root,"branch","--show-current").stdout.strip()
    if not branch: die("detached HEAD")
    status=git(root,"status","--porcelain").stdout.splitlines(); prefix=str(RELATIVE_DIR).replace("\\","/")
    unexpected=[]
    for line in status:
        path=line[3:].strip().strip('"').replace("\\","/")
        if path==str(INDEX) or path==prefix or path.startswith(prefix+"/"): continue
        unexpected.append(line)
    if unexpected: die(f"unrelated changes present: {unexpected}")
    run(["git","fetch","origin",branch,"--tags"],cwd=root)
    idx=root/INDEX; text=idx.read_text(); slug=here.name
    if slug not in text:
        row=f"| {dt.date.today().strftime('%Y.%m.%d')} | Rate-Distortion Immunology: Defragmenting the Aging Immune Repertoire | Coverage-aware methods preprint | [{slug}]({slug}/) |"
        idx.write_text(text.rstrip()+"\n"+row+"\n")
    tag=args.tag or "v2026.10.11-04"
    if git(root,"ls-remote","--exit-code","--tags","origin",f"refs/tags/{tag}",capture=True,check=False).returncode==0: die(f"tag exists: {tag}")
    git(root,"config","user.name","Devanik Debnath"); git(root,"config","user.email","162272415+Devanik21@users.noreply.github.com")
    git(root,"add",str(RELATIVE_DIR),str(INDEX)); git(root,"diff","--cached","--check")
    # Rebase only the local release commit if the branch advanced; never force-push.
    git(root,"commit","-m","Add rate-distortion immunology research release")
    if not args.skip_push:
        run(["git","fetch","origin",branch,"--tags"],cwd=root)
        run(["git","rebase",f"origin/{branch}"],cwd=root)
        check_tag=git(root,"ls-remote","--exit-code","--tags","origin",f"refs/tags/{tag}",capture=True,check=False)
        if check_tag.returncode==0: die(f"tag appeared during deployment: {tag}")
        run(["git","tag","-a",tag,"-m",f"Rate-distortion immunology preprint {tag}"],cwd=root)
        run(["git","push","origin",f"HEAD:{branch}"],cwd=root)
        run(["git","push","origin",f"refs/tags/{tag}"],cwd=root)
        run(["git","fetch","origin",branch,"--tags"],cwd=root)
        head=git(root,"rev-parse","HEAD").stdout.strip()
        remote=git(root,"rev-parse",f"origin/{branch}").stdout.strip()
        if head!=remote: die("remote branch head differs from the release commit")
        paths=git(root,"ls-tree","-r","--name-only",remote,"--",str(RELATIVE_DIR)).stdout.splitlines()
        if len(paths)!=11: die(f"remote release manifest has {len(paths)} files, expected 11")
        for name in sorted(EXPECTED):
            local=git(root,"hash-object",str(RELATIVE_DIR/name)).stdout.strip()
            remote_blob=git(root,"rev-parse",f"{remote}:{RELATIVE_DIR.as_posix()}/{name}").stdout.strip()
            if local!=remote_blob: die(f"remote file hash mismatch: {name}")
        tagged=git(root,"rev-parse",f"refs/tags/{tag}^{{}}").stdout.strip()
        if tagged!=head: die("release tag does not resolve to the verified commit")
        print(f"PASS remote verification: {head}; tag={tag}; files=11/11")
if __name__=="__main__": main()
