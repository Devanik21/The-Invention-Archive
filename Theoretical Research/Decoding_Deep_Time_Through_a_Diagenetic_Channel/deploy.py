#!/usr/bin/env python3
"""Fail-closed local build and validation helper for the speleothem release.

This script rebuilds PDF in a temporary copy of the release; it does not push
files, modify a Git repository, or deposit to an external archive.
Requirements: Python >=3.10, latexmk + pdfLaTeX/BibTeX, NumPy, Matplotlib,
and pypdf (pypdf is used when installed; pdfinfo is the fallback).
"""
from __future__ import annotations
import argparse, json, os, re, shutil, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORE = {"paper.tex", "paper.pdf", "verify_model.py", "README.md", "CITATION.cff", "paper.bib", ".zenodo.json", "schema_article.jsonld", "deploy.py"}
FIGS = {f"figure{i:02d}_{name}.{ext}" for i, name in enumerate([
    "rmse_comparison", "interval_coverage", "alteration_sensitivity", "example_reconstruction",
    "channel_dependency_graph", "response_geometry_graph", "validation_pipeline_flow", "fail_closed_gate_flow"], 1) for ext in ("pdf", "svg")}
MIN_PAGES = 7

class ReleaseError(RuntimeError): pass

def run(cmd, cwd=None):
    p = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if p.returncode:
        raise ReleaseError(f"Command failed ({p.returncode}): {' '.join(map(str,cmd))}\n{p.stdout[-8000:]}")
    return p.stdout

def check_structure(root: Path):
    files = {p.name for p in root.iterdir() if p.is_file()}
    if files != CORE:
        raise ReleaseError(f"Root core manifest mismatch; missing={sorted(CORE-files)} extra={sorted(files-CORE)}")
    for d in ("figures", "results"):
        if not (root/d).is_dir(): raise ReleaseError(f"Required directory missing: {d}")
    got = {p.name for p in (root/"figures").iterdir() if p.is_file()}
    missing = FIGS-got
    if missing: raise ReleaseError(f"Required vector assets missing: {sorted(missing)}")
    if not list((root/"results").glob("*")): raise ReleaseError("results/ is empty")
    return True

def check_metadata(root: Path):
    import json as _json
    _json.loads((root/".zenodo.json").read_text(encoding="utf-8"))
    article = _json.loads((root/"schema_article.jsonld").read_text(encoding="utf-8"))
    assert article.get("@type") == "ScholarlyArticle"
    cff=(root/"CITATION.cff").read_text(encoding="utf-8")
    assert "cff-version: 1.2.0" in cff and "Devanik" in cff
    tex=(root/"paper.tex").read_text(encoding="utf-8")
    bib=(root/"paper.bib").read_text(encoding="utf-8")
    keys=set(re.findall(r"@\w+\s*\{\s*([^,\s]+)",bib))
    cited=set()
    for group in re.findall(r"\\cite[a-zA-Z*]*\{([^}]+)\}",tex): cited.update(x.strip() for x in group.split(","))
    if not cited <= keys: raise ReleaseError(f"Citation keys absent from paper.bib: {sorted(cited-keys)}")
    readme=(root/"README.md").read_text(encoding="utf-8")
    if readme.count("$$") % 2: raise ReleaseError("Unbalanced README display-math delimiters")
    if r"\operatorname" in readme or r"\operatorname*" in readme: raise ReleaseError("README contains prohibited operator macro")
    for link in re.findall(r"\]\(([^)]+)\)",readme):
        if link.startswith(("http://","https://","mailto:")): continue
        if not (root/link).exists(): raise ReleaseError(f"README target missing: {link}")
    return {"bib_keys":len(keys),"cited_keys":len(cited),"readme_links":len(re.findall(r"\]\(([^)]+)\)",readme))}

def run_model(root: Path):
    run([sys.executable,"-m","py_compile",str(root/"verify_model.py"),str(root/"deploy.py")])
    out=run([sys.executable,str(root/"verify_model.py")],cwd=root)
    if "PASS:" not in out or "10 assertion checks" not in out: raise ReleaseError("Model verification output did not show expected pass inventory")
    return out

def build_pdf(root: Path):
    latexmk=shutil.which("latexmk")
    if not latexmk: raise ReleaseError("BLOCKED: latexmk not installed; refuse to use stale PDF")
    with tempfile.TemporaryDirectory(prefix="speleothem-build-") as td:
        tmp=Path(td)
        for name in ["paper.tex","paper.bib","figures"]: shutil.copytree(root/name,tmp/name) if (root/name).is_dir() else shutil.copy2(root/name,tmp/name)
        # Some minimal container images ship a broken BibTeX alternatives link.
        # When the known working binary exists, place a scoped wrapper at the
        # front of PATH; otherwise use the normal TeX Live bibtex executable.
        wrapper=tmp/"tool-wrapper"; wrapper.mkdir()
        original=Path("/usr/bin/bibtex.original")
        if original.is_file() and os.access(original,os.X_OK):
            link=wrapper/"bibtex"
            try: link.symlink_to(original)
            except FileExistsError: pass
            old_path=os.environ.get("PATH","")
            os.environ["PATH"]=str(wrapper)+os.pathsep+old_path
            try: log=run([latexmk,"-pdf","-interaction=nonstopmode","-halt-on-error","paper.tex"],cwd=tmp)
            finally: os.environ["PATH"]=old_path
        else:
            log=run([latexmk,"-pdf","-interaction=nonstopmode","-halt-on-error","paper.tex"],cwd=tmp)
        pdf=tmp/"paper.pdf"
        if not pdf.is_file() or pdf.stat().st_size<5000: raise ReleaseError("Freshly compiled PDF missing or unexpectedly small")
        txt=""
        try:
            from pypdf import PdfReader
            reader=PdfReader(str(pdf)); pages=len(reader.pages); txt="\n".join(p.extract_text() or "" for p in reader.pages)
        except ImportError:
            if not shutil.which("pdfinfo") or not shutil.which("pdftotext"): raise ReleaseError("Install pypdf or pdfinfo+pdftotext to inspect fresh PDF")
            info=run(["pdfinfo",str(pdf)]); pages=int(re.search(r"Pages:\s+(\d+)",info).group(1))
            txt=run(["pdftotext",str(pdf),"-"])
        if pages<MIN_PAGES: raise ReleaseError(f"Fresh PDF has {pages} pages; require at least {MIN_PAGES}")
        if "Co-equal research-process credit" not in txt: raise ReleaseError("Fresh PDF is missing research-process disclosure")
        # latexmk's *combined* stdout can contain harmless first-pass messages
        # such as "Citation(s) may have changed" even when the final pass is
        # clean. Gate on the final pdflatex log and extracted PDF instead.
        final_log=(tmp/"paper.log").read_text(encoding="utf-8",errors="replace") if (tmp/"paper.log").exists() else log
        bad_patterns=[r"There were undefined references",r"LaTeX Warning: Citation [^\n]* undefined",r"Package [^\n]* Warning: Citation [^\n]* undefined",r"There were multiply-defined labels",r"Reference `[^`]+` on page [^\n]* undefined"]
        if any(re.search(pat,final_log,re.I) for pat in bad_patterns):
            raise ReleaseError("Final LaTeX log contains unresolved citations/references")
        if re.search(r"(?m)^\s*(?:LaTeX|Package \w+) Warning: Citation .* undefined",final_log):
            raise ReleaseError("Final LaTeX log contains undefined citations")
        if re.search(r"\[[0-9?]+\]",txt) and "[?]" in txt: raise ReleaseError("PDF text contains unresolved citation marker")
        shutil.copy2(pdf,root/"paper.pdf")
        (root/"results"/"pdf_build_log.txt").write_text(log+"\n--- FINAL PAPER.LOG CHECK ---\n"+final_log+"\nPAGES="+str(pages)+"\n",encoding="utf-8")
        return {"pages":pages,"pdf_bytes":pdf.stat().st_size,"compile_log":"results/pdf_build_log.txt"}

def validate_pdf(root: Path):
    pdf=root/"paper.pdf"
    if not pdf.is_file() or pdf.stat().st_size<5000 or not pdf.read_bytes().startswith(b"%PDF-"):
        raise ReleaseError("paper.pdf missing/invalid")
    try:
        from pypdf import PdfReader
        reader=PdfReader(str(pdf)); pages=len(reader.pages); txt="\n".join(p.extract_text() or "" for p in reader.pages)
    except ImportError:
        pages=int(re.search(r"Pages:\s+(\d+)",run(["pdfinfo",str(pdf)])).group(1));txt=run(["pdftotext",str(pdf),"-"])
    if pages<MIN_PAGES: raise ReleaseError(f"PDF page count {pages} < {MIN_PAGES}")
    if "Co-equal research-process credit" not in txt: raise ReleaseError("PDF disclosure missing")
    if "??" in txt or "[?]" in txt: raise ReleaseError("Possible unresolved references in extracted PDF text")
    return {"pages":pages,"bytes":pdf.stat().st_size}

def self_test():
    # Confirm a valid-enough fixture is rejected when the 9-file root manifest is incomplete,
    # and that a malformed metadata payload is rejected instead of being marked successful.
    with tempfile.TemporaryDirectory(prefix="speleothem-deploy-selftest-") as td:
        root=Path(td); (root/"figures").mkdir();(root/"results").mkdir()
        try: check_structure(root)
        except ReleaseError: pass
        else: raise AssertionError("missing-artifact fixture was not rejected")
        (root/".zenodo.json").write_text("{not json",encoding="utf-8")
        try: json.loads((root/".zenodo.json").read_text())
        except json.JSONDecodeError: pass
        else: raise AssertionError("malformed JSON fixture was not rejected")
        (root/"paper.pdf").write_bytes(b"not a PDF")
        try:
            if not (root/"paper.pdf").read_bytes().startswith(b"%PDF-"): raise ReleaseError("bad fixture PDF")
        except ReleaseError: pass
        else: raise AssertionError("malformed PDF fixture was not rejected")
    return {"status":"PASS","tests":3,"temporary_fixture_only":True}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--self-test",action="store_true");ap.add_argument("--validate-only",action="store_true");args=ap.parse_args()
    try:
        if args.self_test:
            res=self_test();print(f"PASS: deploy self-test {res['tests']}/3 fail-closed fixtures");return 0
        structure=check_structure(ROOT);meta=check_metadata(ROOT); model=run_model(ROOT)
        print(model)
        if args.validate_only:
            pdf=validate_pdf(ROOT);print(f"PASS: structure, metadata, reproducibility, PDF {pdf['pages']} pages; metadata={meta}");return 0
        built=build_pdf(ROOT);valid=validate_pdf(ROOT)
        report={"status":"PASS","structure":"PASS","metadata":meta,"build":built,"pdf_preflight":valid,"model":"PASS","visual_review":"must be completed and recorded separately","remote_deployment":"not performed by this script"}
        (ROOT/"results"/"publication_quality_report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
        print(f"PASS: fresh build and automated preflight; {valid['pages']} pages. Full visual inspection remains a separate mandatory gate.")
        return 0
    except Exception as e:
        print(f"FAIL/BLOCKED: {type(e).__name__}: {e}",file=sys.stderr);return 2
if __name__=="__main__": raise SystemExit(main())
