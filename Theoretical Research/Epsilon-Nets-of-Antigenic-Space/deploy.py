#!/usr/bin/env python3
"""Fail-closed local build, validator, and self-test for this release.

Usage:
  python deploy.py --self-test
  python deploy.py --build
  python deploy.py --validate-only

This script never pushes, deletes, force-pushes, or modifies a Git repository.
"""
from __future__ import annotations
import argparse, hashlib, json, re, shutil, subprocess, sys, tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
CORE={"paper.tex","paper.pdf","verify_model.py","README.md","CITATION.cff","paper.bib",".zenodo.json","schema_article.jsonld","deploy.py"}
STEMS=["fig01_hill_response","fig02_incidence_heatmap","fig03_epsilon_scaling","fig04_sampling_success","fig05_escape_neighbourhood","fig06_multicover_cost","fig07_bipartite_graph","fig08_overlap_graph","fig09_epsilon_pipeline","fig10_active_assay_loop","fig11_range_mass_distribution","fig12_threshold_sensitivity"]
REQ_RESULTS={"research_log.md","profile_audit.md","topic_comparison.csv","literature_audit.csv","claim_evidence_matrix.csv","numerical_outputs.json","verification_summary.txt","verification_stdout.txt","environment_report.txt","publication_design_brief.md","publication_quality_report.md","pdf_visual_review.json","release_check_matrix.json","release_check_summary.md","figure_inventory.csv","sha256_manifest.json"}
class Blocked(RuntimeError): pass
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def need(ok,msg):
    if not ok: raise Blocked(msg)
def run(cmd,cwd=ROOT):
    p=subprocess.run(cmd,cwd=cwd,text=True,capture_output=True)
    if p.returncode: raise Blocked(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{p.stdout}\n{p.stderr}")
    return p.stdout+p.stderr

def check_core(root:Path):
    exp=CORE|{"figures","results"}; got={p.name for p in root.iterdir()}
    need(exp==got,f"exact root manifest mismatch missing={sorted(exp-got)} extra={sorted(got-exp)}")
    for n in CORE: need((root/n).is_file() and (root/n).stat().st_size>0,f"missing/empty core file {n}")
    for n in STEMS:
        for ext in ("pdf","svg"):
            p=root/"figures"/f"{n}.{ext}"; need(p.is_file() and p.stat().st_size>100,f"missing figure asset {p}")
    for n in REQ_RESULTS:
        p=root/"results"/n; need(p.is_file() and p.stat().st_size>0,f"missing result file results/{n}")

def check_meta(root:Path):
    import yaml
    c=yaml.safe_load((root/"CITATION.cff").read_text()); z=json.loads((root/".zenodo.json").read_text()); j=json.loads((root/"schema_article.jsonld").read_text())
    need(c.get("cff-version")=="1.2.0" and c.get("authors") and c.get("date-released"),"CFF validation failed")
    need(z.get("upload_type")=="publication" and z.get("publication_type")=="preprint","Zenodo metadata type mismatch")
    need(not any(x in z for x in ("doi","prereserve_doi")),"Zenodo metadata must not claim minted DOI")
    need(j.get("@type")=="ScholarlyArticle" and j.get("author",{}).get("name")=="Devanik Debnath","JSON-LD validation failed")
    need(c.get("title")==z.get("title")==j.get("headline"),"metadata titles disagree")

def check_text(root:Path):
    tex=(root/"paper.tex").read_text(); bib=(root/"paper.bib").read_text(); readme=(root/"README.md").read_text()
    cited=set()
    for g in re.findall(r"\\cite\w*\{([^}]+)\}",tex): cited.update(x.strip() for x in g.split(","))
    keys=set(re.findall(r"@\w+\s*\{\s*([^,]+)",bib)); items=set(re.findall(r"\\bibitem(?:\[[^]]*\])?\{([^}]+)\}",tex))
    need(cited<=keys,f"missing paper.bib keys {sorted(cited-keys)}")
    need(cited<=items,f"missing rendered bibliography keys {sorted(cited-items)}")
    need(tex.count("\\begin{figure}")>=12,"fewer than 12 figure environments")
    need(tex.count("\\begin{table}")>=5,"fewer than five tables")
    need("Research-process credit and disclosure" in tex,"Xylia process credit missing")
    need(readme.count("$$")%2==0,"unbalanced README display math")
    need("\\operatorname" not in readme and "\\operatorname*" not in readme,"prohibited README math macro")
    for n in ("paper.pdf","paper.tex","verify_model.py","paper.bib","CITATION.cff",".zenodo.json","schema_article.jsonld","deploy.py","figures/","results/"):
        need(n in readme,f"README lacks reference to {n}")

def check_pdf(root:Path):
    import fitz
    d=fitz.open(root/"paper.pdf"); need(len(d)>=25,f"PDF total page gate failed: {len(d)}")
    txt=[p.get_text() for p in d]
    st=next((i for i,t in enumerate(txt) if "A broad-spectrum antibody cocktail is a finite set" in t),None)
    en=next((i for i,t in enumerate(txt) if "Research-process credit and disclosure" in t),None)
    need(st is not None and en is not None and en>=st,"main text range not detected")
    main=en-st+1; need(main>=18,f"main text page gate failed: {main}")
    alltext="\n".join(txt)
    nf=len(re.findall(r"Figure\s+\d+:",alltext)); nt=len(re.findall(r"Table\s+\d+:",alltext))
    need(nf>=12,f"PDF has only {nf} figure captions"); need(nt>=5,f"PDF has only {nt} table captions")
    return {"pages":len(d),"main_text_pages":main,"main_start_pdf_page":st+1,"main_end_pdf_page":en+1,"figures":nf,"tables":nt,"pdf_sha256":sha(root/"paper.pdf")}

def check_manifest(root:Path):
    m=json.loads((root/"results"/"sha256_manifest.json").read_text())
    for rel,digest in m["files"].items():
        p=root/rel; need(p.is_file(),f"manifest path absent: {rel}"); need(sha(p)==digest,f"manifest hash mismatch: {rel}")

def validate(root:Path):
    check_core(root); check_meta(root); check_text(root); stats=check_pdf(root); check_manifest(root); return stats

def build(root:Path):
    check_core(root)
    v=run([sys.executable,"verify_model.py"])
    (root/"results"/"verification_stdout.txt").write_text(v)
    run([sys.executable,"-m","py_compile","verify_model.py","deploy.py"])
    if shutil.which("pdflatex") is None: raise Blocked("pdflatex unavailable; stale PDF will not be used")
    with tempfile.TemporaryDirectory(prefix="epsilon_nets_build_") as td:
        t=Path(td); shutil.copy2(root/"paper.tex",t/"paper.tex"); shutil.copytree(root/"figures",t/"figures")
        logs=[]
        for _ in range(2): logs.append(run(["pdflatex","-interaction=nonstopmode","-halt-on-error","paper.tex"],t))
        final=logs[-1]
        for bad in ("LaTeX Warning: Citation","There were undefined references","undefined citations","LaTeX Error","! Emergency stop"):
            need(bad not in final,f"final LaTeX log contains {bad}")
        need((t/"paper.pdf").is_file() and (t/"paper.pdf").stat().st_size>10000,"fresh PDF build missing/too small")
        shutil.copy2(t/"paper.pdf",root/"paper.pdf")
    (root/"results"/"build_log.txt").write_text("\n\n".join(logs))
    stats=check_pdf(root)
    provenance={"build":"fresh temporary-directory pdfTeX build","pdf_sha256":stats["pdf_sha256"],"paper_tex_sha256":sha(root/"paper.tex"),"paper_bib_sha256":sha(root/"paper.bib"),"verification_exit_code":0,"pdf":stats,"bibliography":"manually typeset; BibTeX unavailable; keys cross-checked"}
    (root/"results"/"build_provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    return provenance

def selftest():
    # Fail-closed logic is tested on isolated fixture directories and never touches the release.
    checks=[]
    with tempfile.TemporaryDirectory(prefix="epsilon-net-fixture-") as td:
        p=Path(td); (p/"figures").mkdir(); (p/"results").mkdir()
        for n in CORE: (p/n).write_text("fixture")
        for st in STEMS:
            for ext in ("pdf","svg"): (p/"figures"/f"{st}.{ext}").write_text("fixture")
        for n in REQ_RESULTS: (p/"results"/n).write_text("fixture")
        (p/"paper.pdf").unlink()
        try: check_core(p); raise AssertionError("missing core file accepted")
        except Blocked: checks.append({"case":"missing root paper.pdf","status":"PASS"})
        (p/"paper.pdf").write_text("fixture"); (p/"unexpected.txt").write_text("x")
        try: check_core(p); raise AssertionError("extra root accepted")
        except Blocked: checks.append({"case":"unexpected root file","status":"PASS"})
        (p/"unexpected.txt").unlink(); (p/"figures"/f"{STEMS[-1]}.svg").unlink()
        try: check_core(p); raise AssertionError("missing figure accepted")
        except Blocked: checks.append({"case":"missing SVG figure","status":"PASS"})
    return {"checks":checks,"summary":"3/3 malformed fixture releases blocked; real release unchanged."}

def main():
    ap=argparse.ArgumentParser(); g=ap.add_mutually_exclusive_group(required=True); g.add_argument("--build",action="store_true"); g.add_argument("--validate-only",action="store_true"); g.add_argument("--self-test",action="store_true"); a=ap.parse_args()
    try:
        if a.self_test: print(json.dumps(selftest(),indent=2)); return 0
        if a.build: print(json.dumps(build(ROOT),indent=2)); return 0
        print(json.dumps(validate(ROOT),indent=2)); return 0
    except (Blocked,OSError,ValueError,json.JSONDecodeError) as e:
        print(f"BLOCKED: {e}",file=sys.stderr); return 2
if __name__=="__main__": raise SystemExit(main())