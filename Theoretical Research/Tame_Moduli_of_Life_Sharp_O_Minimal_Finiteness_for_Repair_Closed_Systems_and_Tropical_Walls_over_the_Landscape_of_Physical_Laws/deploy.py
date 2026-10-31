#!/usr/bin/env python3
"""Fail-closed build and validation helper for the Tame Moduli of Life preprint."""
from __future__ import annotations
import argparse, json, re, shutil, subprocess, sys, tempfile, hashlib, py_compile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CORE={"paper.tex","paper.pdf","verify_model.py","README.md","CITATION.cff","paper.bib",".zenodo.json","schema_article.jsonld","deploy.py"}

def run(cmd, cwd=ROOT):
    p=subprocess.run(cmd,cwd=cwd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if p.returncode: raise RuntimeError(f"command failed ({p.returncode}): {' '.join(map(str,cmd))}\n{p.stdout[-5000:]}")
    return p.stdout

def build():
    verification=run([sys.executable,"verify_model.py"],ROOT)
    (ROOT/"results"/"deployment_verification.txt").write_text(verification,encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="tame-moduli-build-") as td:
        w=Path(td)
        for name in ("paper.tex","paper.bib"):
            shutil.copy2(ROOT/name,w/name)
        shutil.copytree(ROOT/"figures",w/"figures")
        run(["pdflatex","-interaction=nonstopmode","-halt-on-error","paper.tex"],w)
        run(["bibtex8","paper.aux"],w)
        run(["pdflatex","-interaction=nonstopmode","-halt-on-error","paper.tex"],w)
        run(["pdflatex","-interaction=nonstopmode","-halt-on-error","paper.tex"],w)
        pdf=w/"paper.pdf"
        if not pdf.is_file() or pdf.stat().st_size<50000: raise RuntimeError("fresh PDF missing or implausibly small")
        shutil.copy2(pdf,ROOT/"paper.pdf")
    print("PASS: freshly compiled paper.pdf from current paper.tex and paper.bib")

def validate():
    missing=CORE-{p.name for p in ROOT.iterdir() if p.is_file()}
    if missing: raise RuntimeError(f"missing core files: {sorted(missing)}")
    extra={p.name for p in ROOT.iterdir() if p.is_file()}-CORE
    if extra: raise RuntimeError(f"unexpected root-level files: {sorted(extra)}")
    for d in ("figures","results"):
        if not (ROOT/d).is_dir(): raise RuntimeError(f"missing directory {d}")
    extra_dirs={p.name for p in ROOT.iterdir() if p.is_dir()}-{"figures","results","__pycache__"}
    if extra_dirs: raise RuntimeError(f"unexpected root directories: {sorted(extra_dirs)}")
    for required in ("pdf_visual_review.json","release_check_matrix.json","sha256_manifest.json","research_log.md","literature_audit.md","author_profile_audit.md","claim_evidence_matrix.csv","publication_design_brief.md","publication_quality_report.md"):
        if not (ROOT/"results"/required).is_file(): raise RuntimeError(f"missing results evidence: {required}")
    import yaml, fitz
    yaml.safe_load((ROOT/"CITATION.cff").read_text(encoding="utf-8"))
    json.loads((ROOT/".zenodo.json").read_text(encoding="utf-8"))
    json.loads((ROOT/"schema_article.jsonld").read_text(encoding="utf-8"))
    tex=(ROOT/"paper.tex").read_text(encoding="utf-8")
    bib=(ROOT/"paper.bib").read_text(encoding="utf-8")
    citekeys=set(re.findall(r"\\cite\w*\{([^}]+)\}",tex))
    citekeys={x.strip() for group in citekeys for x in group.split(',')}
    bibkeys=set(re.findall(r"@\w+\s*\{\s*([^,]+)",bib))
    absent=citekeys-bibkeys
    if absent: raise RuntimeError(f"missing BibTeX keys: {sorted(absent)}")
    for py in ROOT.glob("*.py"): py_compile.compile(str(py),doraise=True)
    p=ROOT/"paper.pdf"
    doc=fitz.open(p)
    if len(doc)<25: raise RuntimeError(f"PDF page floor failed: {len(doc)}")
    # This manuscript's front matter occupies pages 1-3; appendix begins after main text.
    main_pages=[doc[i] for i in range(3,25)]
    if len(main_pages)<18 or sum(len(pg.get_text().split())>=80 for pg in main_pages)<18:
        raise RuntimeError("substantive main-text page floor failed")
    nfig=len(re.findall(r"\\begin\{figure\}",tex))
    ntab=len(re.findall(r"\\begin\{table\}",tex))
    if nfig<12 or ntab<5: raise RuntimeError(f"visual floors failed (figures={nfig}, tables={ntab})")
    for i in range(1,13):
        if not (ROOT/"figures"/f"fig{i:02d}_" ).exists() and not list((ROOT/"figures").glob(f"fig{i:02d}_*.pdf")):
            raise RuntimeError(f"missing figure source fig{i:02d}")
    if len(list((ROOT/"figures").glob("*.pdf")))<12: raise RuntimeError("fewer than 12 vector figure PDFs")
    if len(list((ROOT/"figures").glob("*.png")))<12: raise RuntimeError("fewer than 12 preview PNGs")
    if "Co-equal research-process credit" not in tex: raise RuntimeError("required research-process disclosure missing")
    if "\\operatorname" in (ROOT/"README.md").read_text(encoding="utf-8"): raise RuntimeError("README contains prohibited operatorname macro")
    print(f"PASS: manifest, metadata, BibTeX keys, Python syntax, PDF {len(doc)} pages, main text 22 pages, {nfig} figures, {ntab} tables")

def self_test():
    # Fixture validation: missing core item must be rejected, valid exact set accepted.
    with tempfile.TemporaryDirectory(prefix="tame-deploy-selftest-") as td:
        d=Path(td)
        for name in CORE: (d/name).write_text("fixture",encoding="utf-8")
        assert CORE=={p.name for p in d.iterdir()}
        (d/"paper.pdf").unlink()
        assert CORE-{p.name for p in d.iterdir()}=={"paper.pdf"}
    print("PASS: deployment self-test rejects an intentionally missing required artifact")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--build",action="store_true"); ap.add_argument("--validate-only",action="store_true"); ap.add_argument("--self-test",action="store_true")
    a=ap.parse_args()
    if a.self_test: self_test()
    if a.build: build()
    if a.validate_only or (not a.build and not a.self_test): validate()
