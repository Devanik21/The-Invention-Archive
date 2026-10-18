#!/usr/bin/env python3
"""Fail-closed local release validator for Silica Light Meters."""
from __future__ import annotations
import argparse, hashlib, json, os, py_compile, re, shutil, subprocess, sys, tempfile
from pathlib import Path
CORE={"paper.tex","paper.pdf","verify_model.py","README.md","CITATION.cff","paper.bib",".zenodo.json","schema_article.jsonld","deploy.py"}
MIN_FIGURES={f"fig{i:02d}_{n}.pdf" for i,n in [(1,"surrogate_spectra"),(2,"attenuation"),(3,"incremental_prediction"),(4,"inverse_nonidentifiability"),(5,"causal_graph"),(6,"feature_graph"),(7,"validation_flow"),(8,"fail_closed_flow")]}

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pdf_stats(path):
    try:
        import fitz
        d=fitz.open(path); return len(d),sum(len(p.get_text()) for p in d)
    except Exception as e: raise RuntimeError(f"PDF parse failed: {e}")

def compile_pdf(root):
    if not shutil.which("pdflatex") or not shutil.which("biber"):
        raise RuntimeError("pdflatex and biber are required; refusing a stale PDF fallback")
    commands=[["pdflatex","-interaction=nonstopmode","-halt-on-error","paper.tex"],["biber","paper"],["pdflatex","-interaction=nonstopmode","-halt-on-error","paper.tex"],["pdflatex","-interaction=nonstopmode","-halt-on-error","paper.tex"],["pdflatex","-interaction=nonstopmode","-halt-on-error","paper.tex"]]
    log=[]
    for idx, cmd in enumerate(commands):
        p=subprocess.run(cmd,cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        log.append({"command":" ".join(cmd),"exit_code":p.returncode,"tail":p.stdout[-4000:]})
        if p.returncode: raise RuntimeError("Build failed: "+" ".join(cmd)+"\n"+p.stdout[-3000:])
        if cmd[0] == "pdflatex" and idx == len(commands)-1 and re.search(r"LaTeX Warning: (Citation .* undefined|Reference .* undefined)",p.stdout):
            raise RuntimeError("Build has undefined citations/references after the final pass: "+p.stdout[-3000:])
    return log

def validate(root, full=True):
    root=Path(root).resolve(); problems=[]; evidence=[]
    files={p.name for p in root.iterdir() if p.is_file()}
    if files != CORE: problems.append(f"root core manifest mismatch; missing={sorted(CORE-files)} extra={sorted(files-CORE)}")
    if not (root/"figures").is_dir() or not (root/"results").is_dir(): problems.append("figures/ and results/ directories are required")
    for f in CORE:
        if not (root/f).exists() or (root/f).stat().st_size==0: problems.append(f"missing/empty core file: {f}")
    if problems: raise RuntimeError("; ".join(problems))
    tex=(root/"paper.tex").read_text(encoding="utf-8")
    if tex.count(r"\begin{table}") < 5: problems.append("fewer than 5 table environments")
    if tex.count(r"\begin{figure}") < 8: problems.append("fewer than 8 figure environments")
    for name in MIN_FIGURES:
        if not (root/"figures"/name).exists(): problems.append(f"missing vector figure {name}")
    for name in ["fig01_surrogate_spectra.png","fig02_attenuation.png","fig03_incremental_prediction.png","fig04_inverse_nonidentifiability.png","fig05_causal_graph.png","fig06_feature_graph.png","fig07_validation_flow.png","fig08_fail_closed_flow.png"]:
        if not (root/"figures"/name).exists(): problems.append(f"missing PNG preview {name}")
    try:
        json.loads((root/".zenodo.json").read_text(encoding="utf-8")); json.loads((root/"schema_article.jsonld").read_text(encoding="utf-8"))
    except Exception as e: problems.append(f"metadata JSON validation failed: {e}")
    try:
        import yaml
        cff=yaml.safe_load((root/"CITATION.cff").read_text(encoding="utf-8"))
        if not cff.get("title") or not cff.get("authors"): problems.append("CITATION.cff lacks title/authors")
    except Exception as e: problems.append(f"CITATION.cff validation failed: {e}")
    readme=(root/"README.md").read_text(encoding="utf-8")
    if readme.count("$$") % 2: problems.append("unbalanced README display math delimiters")
    if r"\operatorname" in readme: problems.append("prohibited README operator macro found")
    if "synthetic" not in readme.lower() or "not rcwa" not in readme.lower(): problems.append("README lacks explicit synthetic/non-RCWA boundary")
    if re.search(r"\\cite\{([^}]+)\}",tex):
        keys=set(re.findall(r"\\cite\w*\{([^}]+)\}",tex)); bib=(root/"paper.bib").read_text(encoding="utf-8")
        bibkeys=set(re.findall(r"@\w+\s*\{\s*([^,]+)",bib))
        for keyblob in keys:
            for key in keyblob.split(","):
                if key.strip() not in bibkeys: problems.append(f"missing BibTeX key {key.strip()}")
    for py in [root/"verify_model.py",root/"deploy.py"]:
        try: py_compile.compile(str(py),doraise=True)
        except Exception as e: problems.append(f"Python compile failed for {py.name}: {e}")
    if full and not problems:
        p=subprocess.run([sys.executable,"verify_model.py"],cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        evidence.append({"check":"verify_model.py","exit_code":p.returncode,"output_tail":p.stdout[-2000:]})
        if p.returncode: problems.append("verify_model.py failed on validator run")
    if problems: raise RuntimeError("; ".join(problems))
    pages,chars=pdf_stats(root/"paper.pdf")
    if pages<7: raise RuntimeError(f"PDF is only {pages} pages (<7)")
    if chars<12000: raise RuntimeError(f"PDF text extraction has only {chars} characters (<12000)")
    # Ensure required visual inventory in the embedded manuscript.
    if tex.count(r"\begin{table}")<5 or tex.count(r"\begin{figure}")<8: raise RuntimeError("visual inventory is below the release minimum")
    current_pdf_sha=sha(root/"paper.pdf")
    review_path=root/"results"/"pdf_visual_review.json"
    review_record=None
    visual_quality="NOT_RECORDED"
    if review_path.exists():
        try:
            review_record=json.loads(review_path.read_text(encoding="utf-8"))
            if review_record.get("pdf_sha256") != current_pdf_sha:
                problems.append("pdf_visual_review.json SHA-256 does not match the current paper.pdf")
            if review_record.get("page_count") != pages:
                problems.append("pdf_visual_review.json page count does not match the current PDF")
            expected_pages=list(range(1,pages+1))
            if review_record.get("status") == "PASS" and review_record.get("pages_reviewed") == expected_pages:
                visual_quality="PASS: recorded page-by-page review matches the current PDF hash"
            else:
                visual_quality=f"{review_record.get('status','UNSPECIFIED')}: visual gate not fully cleared"
        except Exception as e:
            problems.append(f"PDF visual-review record is invalid: {e}")
    if problems: raise RuntimeError("; ".join(problems))
    report={"status":"PASS","root":str(root),"pdf_sha256":current_pdf_sha,"pdf_pages":pages,"pdf_extracted_characters":chars,"core_file_count":len(files),"figure_pdf_count":len(list((root/"figures").glob("*.pdf"))),"figure_png_count":len(list((root/"figures").glob("*.png"))),"table_environments":tex.count(r"\begin{table}"),"figure_environments":tex.count(r"\begin{figure}"),"checks":evidence,"empirical_validation":"NOT_PERFORMED","visual_quality":visual_quality,"visual_review_status":(review_record or {}).get("status","NOT_RECORDED")}
    return report

def self_test():
    with tempfile.TemporaryDirectory() as td:
        p=Path(td); (p/"paper.tex").write_text("broken",encoding="utf-8")
        try: validate(p,full=False)
        except RuntimeError as e:
            if "manifest" not in str(e): raise
        else: raise AssertionError("self-test failed: malformed fixture was not blocked")
    print("PASS: malformed fixture is blocked; no real release files touched")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--self-test",action="store_true"); ap.add_argument("--validate-only",action="store_true"); ap.add_argument("--build",action="store_true"); ap.add_argument("--root",default=".")
    a=ap.parse_args(); root=Path(a.root).resolve()
    if a.self_test: self_test(); return
    if a.build:
        with tempfile.TemporaryDirectory(prefix="silica_latex_") as td:
            temp=Path(td)
            for p in ["paper.tex","paper.bib"]: shutil.copy2(root/p,temp/p)
            # Copy only assets that paper.tex includes, preserving relative paths.
            shutil.copytree(root/"figures",temp/"figures")
            log=compile_pdf(temp)
            shutil.copy2(temp/"paper.pdf",root/"paper.pdf")
            (root/"results"/"latex_build_log.json").write_text(json.dumps(log,indent=2)+"\n",encoding="utf-8")
    report=validate(root,full=True)
    results=root/"results"; results.mkdir(exist_ok=True)
    (results/"deployment_validation.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    # SHA-256 manifest excludes itself to avoid recursion.
    manifest={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob("*")) if p.is_file() and p.name!="sha256_manifest.json" and "__pycache__" not in p.parts}
    (results/"sha256_manifest.json").write_text(json.dumps({"algorithm":"sha256","files":manifest},indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__=="__main__": main()
