#!/usr/bin/env python3
"""Validate, rebuild, index, commit, and tag the Damage R0 preprint.

Run from a clean repository clone:
  python "Theoretical Research/Damage_R0_A_Next_Generation_Theory_of_Aging_in_Interdependent_Living_Systems/deploy.py"
Local-only mode: append --skip-push. Requires Git, Python >=3.10, NumPy,
PyYAML, and a LaTeX toolchain (pdflatex). No credentials are embedded.
"""
from __future__ import annotations
import argparse, datetime as dt, json, re, shutil, subprocess, sys, tempfile
from pathlib import Path
EXPECTED={"paper.tex","paper.pdf","verify_model.py","README.md","medium_story.md","x_thread.md","CITATION.cff","paper.bib",".zenodo.json","schema_article.jsonld","deploy.py"}
REL=Path("Theoretical Research/Damage_R0_A_Next_Generation_Theory_of_Aging_in_Interdependent_Living_Systems")
INDEX=Path("Theoretical Research/Readme.md")
def run(args,cwd=None,capture=False,check=True): return subprocess.run(args,cwd=cwd,text=True,capture_output=capture,check=check)
def git(root,*args,check=True): return run(["git",*args],cwd=root,capture=True,check=check)
def die(msg): raise SystemExit("ERROR: "+msg)
def main():
 ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--tag",default="v2026.10.13-01"); ap.add_argument("--skip-push",action="store_true"); args=ap.parse_args()
 here=Path(__file__).resolve().parent; root=Path(git(here,"rev-parse","--show-toplevel").stdout.strip()).resolve()
 if (root/REL).resolve()!=here: die("unexpected release directory")
 present={p.name for p in here.iterdir() if p.is_file()}
 if present!=EXPECTED: die(f"11-file manifest mismatch: missing={EXPECTED-present}, extra={present-EXPECTED}")
 import numpy as np, yaml
 run([sys.executable,str(here/"verify_model.py")],cwd=here)
 json.loads((here/".zenodo.json").read_text()); json.loads((here/"schema_article.jsonld").read_text()); cff=yaml.safe_load((here/"CITATION.cff").read_text())
 if cff.get("cff-version")!="1.2.0" or not cff.get("authors"): die("CITATION.cff invalid")
 words=re.findall(r"\b[\w'-]+\b",(here/"medium_story.md").read_text())
 if not 1500<=len(words)<=2000: die(f"Medium word count={len(words)}")
 posts=re.findall(r"(?m)^\*\*\d+/12\*\*",(here/"x_thread.md").read_text())
 if len(posts)!=12: die("X thread must contain 12 posts")
 rm=(here/"README.md").read_text(); n=len(re.findall(r"(?m)^\$\$$",rm))
 if n!=8 or r"\n" in rm or r"\operatorname" in rm: die("README display math integrity failed")
 if shutil.which("pdflatex") is None: die("pdflatex is required for source-consistent build")
 with tempfile.TemporaryDirectory(prefix="damage-r0-build-") as tmp:
  import os
  tmp=Path(tmp); shutil.copy2(here/"paper.tex",tmp/"paper.tex"); shutil.copy2(here/"paper.bib",tmp/"paper.bib")
  for cmd in (["pdflatex","-interaction=nonstopmode","-halt-on-error","paper.tex"],["pdflatex","-interaction=nonstopmode","-halt-on-error","paper.tex"],["pdflatex","-interaction=nonstopmode","-halt-on-error","paper.tex"]):
   p=run(cmd,cwd=tmp,capture=True,check=False)
   if p.returncode: print((p.stdout or "")[-7000:]); die("source build failed")
  pdf=tmp/"paper.pdf"
  if not pdf.exists() or pdf.stat().st_size<5000 or not pdf.read_bytes().startswith(b"%PDF-"): die("invalid rebuilt PDF")
  shutil.copy2(pdf,here/"paper.pdf")
 try:
  from pypdf import PdfReader
  pages=len(PdfReader(str(here/"paper.pdf")).pages)
 except Exception as e: die(f"cannot verify PDF pages: {e}")
 if pages<7: die(f"PDF has only {pages} pages")
 print(f"PASS: source, metadata, {len(words)} Medium words, 12 X posts, PDF pages={pages}")
 if args.skip_push: print("Local-only mode: no GitHub push attempted"); return
 branch=git(root,"branch","--show-current").stdout.strip()
 if not branch: die("detached HEAD")
 status=git(root,"status","--porcelain").stdout.splitlines(); prefix=REL.as_posix()
 extra=[]
 for line in status:
  p=line[3:].strip().strip('"').replace('\\','/')
  if p==INDEX.as_posix() or p==prefix or p.startswith(prefix+'/'): continue
  extra.append(line)
 if extra: die(f"unrelated local changes present: {extra}")
 idx=root/INDEX; old=idx.read_text(encoding="utf-8")
 if REL.name not in old:
  row=f"| 2026.10.13 | Damage R₀: A Next-Generation Theory of Aging in Interdependent Living Systems | Spectral damage-threshold preprint | [{REL.name}]({REL.name}/) |"
  idx.write_text(old.rstrip()+"\n"+row+"\n",encoding="utf-8")
 run(["git","config","user.name","Devanik Debnath"],cwd=root); run(["git","config","user.email","162272415+Devanik21@users.noreply.github.com"],cwd=root)
 run(["git","add",str(REL),str(INDEX)],cwd=root); git(root,"diff","--cached","--check")
 run(["git","fetch","origin",branch,"--tags"],cwd=root)
 if git(root,"ls-remote","--exit-code","--tags","origin",f"refs/tags/{args.tag}",check=False).returncode==0: die(f"tag already exists: {args.tag}")
 run(["git","commit","-m","Add Damage R0 aging-network theory preprint"],cwd=root)
 if args.skip_push: return
 run(["git","fetch","origin",branch,"--tags"],cwd=root); run(["git","rebase",f"origin/{branch}"],cwd=root)
 if git(root,"ls-remote","--exit-code","--tags","origin",f"refs/tags/{args.tag}",check=False).returncode==0: die("tag appeared during deployment")
 run(["git","tag","-a",args.tag,"-m",f"Damage R0 theoretical preprint {args.tag}"],cwd=root); run(["git","push","origin",f"HEAD:{branch}"],cwd=root); run(["git","push","origin",f"refs/tags/{args.tag}"],cwd=root)
 run(["git","fetch","origin",branch,"--tags"],cwd=root); head=git(root,"rev-parse","HEAD").stdout.strip()
 if git(root,"rev-parse",f"origin/{branch}").stdout.strip()!=head: die("remote head differs from release commit")
 paths=git(root,"ls-tree","-r","--name-only",head,"--",prefix).stdout.splitlines()
 if len(paths)!=11: die(f"remote manifest has {len(paths)} files")
 for name in EXPECTED:
  local=git(root,"hash-object",str(REL/name)).stdout.strip(); remote=git(root,"rev-parse",f"{head}:{prefix}/{name}").stdout.strip()
  if local!=remote: die(f"remote hash mismatch: {name}")
 if git(root,"rev-parse",f"refs/tags/{args.tag}^{{}}").stdout.strip()!=head: die("tag target mismatch")
 print(f"PASS remote verification: commit={head}; tag={args.tag}; manifest=11/11")
if __name__=="__main__": main()
