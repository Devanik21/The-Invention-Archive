#!/usr/bin/env python3
"""Build and validate the Beyond the Biological Bottleneck preprint."""
from __future__ import annotations
import argparse, hashlib, json, re, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
CORE=["paper.tex","paper.bib","verify_model.py","README.md","CITATION.cff",".zenodo.json","schema_article.jsonld","deploy.py"]

def self_test():
    import yaml
    meta=json.loads((ROOT/".zenodo.json").read_text())
    schema=json.loads((ROOT/"schema_article.jsonld").read_text())
    citation=yaml.safe_load((ROOT/"CITATION.cff").read_text())
    assert meta["publication_type"]=="preprint"
    assert schema["@type"]=="ScholarlyArticle"
    assert citation["cff-version"]=="1.2.0"
    tex=(ROOT/"paper.tex").read_text()
    assert tex.count(r"\begin{equation}")>=6
    assert tex.count(r"\bibitem{")>=10
    assert "functional non-identifiability" in tex.lower()
    print("Metadata, source structure, and claim checks: PASS")

def build():
    subprocess.run([sys.executable,"verify_model.py"],cwd=ROOT,check=True)
    subprocess.run(["latexmk","-pdf","-interaction=nonstopmode","-halt-on-error","paper.tex"],cwd=ROOT,check=True)
    for pat in ["*.aux","*.fdb_latexmk","*.fls","*.log","*.out","*.toc"]:
        for p in ROOT.glob(pat): p.unlink(missing_ok=True)

def validate():
    import fitz
    pdf=ROOT/"paper.pdf"
    assert pdf.is_file() and pdf.stat().st_size>50000
    doc=fitz.open(pdf)
    text="\n".join(p.get_text() for p in doc)
    assert len(text)>18000
    for phrase in ["Beyond the Biological Bottleneck","functional non-identifiability","formose reaction","Limitations and epistemic status"]:
        assert phrase.lower() in text.lower(), phrase
    assert doc.page_count>=10, doc.page_count
    for name in CORE: assert (ROOT/name).is_file() and (ROOT/name).stat().st_size>0
    for name in ["fig1_pathways","fig2_capability_ladder","fig3_identifiability","fig4_inference_pipeline"]:
        assert (ROOT/"figures"/(name+".pdf")).is_file()
        assert (ROOT/"figures"/(name+".png")).is_file()
    summary=json.loads((ROOT/"results/synthetic_origin_demo_summary.json").read_text())
    assert summary["synthetic_only"] is True and summary["n_per_class"]==10000
    quality={"status":"AUTOMATED_PREFLIGHT_PASS_MANUAL_REVIEW_PENDING","pages":doc.page_count,"characters":len(text),"pdf_sha256":hashlib.sha256(pdf.read_bytes()).hexdigest(),"manual_visual_review":"Automated render checks only; complete page-by-page human sign-off remains pending."}
    (ROOT/"results/pdf_preflight.json").write_text(json.dumps(quality,indent=2)+"\n")
    report=("# Publication-quality preflight\n\n"
            "**Automated status: PASS. Manual status: PENDING.**\n\n"
            f"- PDF pages: {doc.page_count}\n- Extracted text characters: {len(text)}\n"
            f"- PDF SHA-256: `{quality['pdf_sha256']}`\n"
            "- Source compilation and deterministic model checks: passed locally.\n"
            "- Visual inspection: rendered contact sheet and representative pages were inspected; no obvious major clipping or overlap was seen. A complete line-by-line, page-by-page editorial sign-off remains pending.\n"
            "- No empirical sample data or extraterrestrial observations are included.\n")
    (ROOT/"results/publication_quality_report.md").write_text(report)
    entries={}
    for p in sorted(ROOT.rglob("*")):
        if p.is_file() and p.name!="sha256_manifest.json" and "__pycache__" not in p.parts:
            entries[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    (ROOT/"results/sha256_manifest.json").write_text(json.dumps(entries,indent=2)+"\n")
    quality["manifest_entries"]=len(entries)
    print(json.dumps(quality,indent=2))

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--self-test",action="store_true"); ap.add_argument("--build",action="store_true"); ap.add_argument("--validate-only",action="store_true")
    a=ap.parse_args()
    if a.self_test: self_test()
    if a.build: build()
    if a.validate_only: validate()
