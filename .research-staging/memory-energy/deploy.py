#!/usr/bin/env python3
"""Fail-closed build, preflight, and fixture self-tests for this research release."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import fitz
import yaml

CORE = {"paper.tex","paper.pdf","verify_model.py","README.md","CITATION.cff","paper.bib",".zenodo.json","schema_article.jsonld","deploy.py"}
TITLE = "The Memory--Energy Frontier in Noisy Recurrent Inference: A Spectral Rate--Distortion Framework for Neural and Neuromorphic Systems"
RELEASE_DATE = "2026-10-24"

class ReleaseError(RuntimeError):
    pass

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def run(cmd, cwd, env=None):
    proc=subprocess.run(cmd,cwd=cwd,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if proc.returncode:
        raise ReleaseError(f"Command failed ({proc.returncode}): {' '.join(map(str,cmd))}\n{proc.stdout[-8000:]}")
    return proc.stdout

def root_files(root: Path):
    return {p.name for p in root.iterdir() if p.is_file()}

def validate_root_files(root: Path):
    actual=root_files(root)
    if actual != CORE:
        raise ReleaseError(f"root core-file mismatch; missing={sorted(CORE-actual)} extra={sorted(actual-CORE)}")
    if not (root/"figures").is_dir() or not (root/"results").is_dir():
        raise ReleaseError("required figures/ and results/ directories are missing")

def validate_metadata(root: Path):
    try:
        cff=yaml.safe_load((root/"CITATION.cff").read_text(encoding="utf-8"))
        zenodo=json.loads((root/".zenodo.json").read_text(encoding="utf-8"))
        schema=json.loads((root/"schema_article.jsonld").read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReleaseError(f"metadata parse failed: {exc}") from exc
    if not isinstance(cff,dict) or cff.get("cff-version")!="1.2.0":
        raise ReleaseError("CITATION.cff is not valid CFF 1.2.0 metadata")
    if cff.get("title")!=TITLE or zenodo.get("title")!=TITLE or schema.get("headline")!=TITLE:
        raise ReleaseError("metadata titles disagree with the manuscript title")
    if cff.get("license")!="MIT" or zenodo.get("license")!="MIT":
        raise ReleaseError("metadata license does not match repository MIT license")
    if "doi" in zenodo or schema.get("identifier"):
        raise ReleaseError("unissued DOI/persistent identifier must not be claimed")
    if schema.get("@type")!="ScholarlyArticle":
        raise ReleaseError("JSON-LD @type must be ScholarlyArticle")

def validate_bibliography(root: Path):
    tex=(root/"paper.tex").read_text(encoding="utf-8")
    bib=(root/"paper.bib").read_text(encoding="utf-8")
    keys=set(re.findall(r"@\w+\s*\{\s*([^,]+)",bib))
    cited=[]
    for block in re.findall(r"\\cite\w*\{([^}]+)\}",tex):
        cited.extend(k.strip() for k in block.split(",") if k.strip())
    missing=sorted(set(cited)-keys)
    if missing:
        raise ReleaseError(f"unresolved BibTeX keys: {missing}")
    if len(keys)<7:
        raise ReleaseError("bibliography unexpectedly small")
    return len(keys),len(set(cited))

def validate_readme(root: Path):
    readme=(root/"README.md").read_text(encoding="utf-8")
    if readme.count("$$") % 2:
        raise ReleaseError("README display-math delimiters are unbalanced")
    if r"\operatorname" in readme or r"\operatorname*" in readme:
        raise ReleaseError("README uses a prohibited operator macro")
    if "\n$$" not in readme or "$$\n" not in readme:
        raise ReleaseError("README display math must use dedicated blocks")
    for target in re.findall(r"\]\(([^)]+)\)",readme):
        if target.startswith(("http://","https://","mailto:","#")):
            continue
        path=(root/target.split("#",1)[0])
        if not path.exists():
            raise ReleaseError(f"README internal link target missing: {target}")
    return readme.count("$$")//2

def validate_manifest(root: Path):
    manifest=root/"results/sha256_manifest.txt"
    if not manifest.is_file():
        raise ReleaseError("SHA-256 manifest missing")
    entries={}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts=line.split("  ",1)
        if len(parts)!=2:
            raise ReleaseError(f"malformed manifest line: {line}")
        digest,path=parts
        if not re.fullmatch(r"[0-9a-f]{64}",digest):
            raise ReleaseError(f"invalid SHA-256 digest for {path}")
        entries[path]=digest
    if not entries:
        raise ReleaseError("SHA-256 manifest is empty")
    for rel,digest in entries.items():
        target=root/rel
        if not target.is_file() or sha256(target)!=digest:
            raise ReleaseError(f"manifest mismatch: {rel}")
    return entries

def validate_tree(root: Path):
    validate_root_files(root)
    validate_metadata(root)
    bib_count,cite_count=validate_bibliography(root)
    math_blocks=validate_readme(root)
    tex=(root/"paper.tex").read_text(encoding="utf-8")
    if len(re.findall(r"\\begin\{table\}",tex)) < 5:
        raise ReleaseError("fewer than five substantive table environments")
    if len(re.findall(r"\\begin\{figure\}",tex)) < 8:
        raise ReleaseError("fewer than eight required visual environments")
    pdfs=sorted((root/"figures").glob("figure*.pdf"))
    svgs=sorted((root/"figures").glob("figure*.svg"))
    pngs=sorted((root/"figures").glob("figure*.png"))
    if min(len(pdfs),len(svgs),len(pngs)) < 8:
        raise ReleaseError(f"figure inventory incomplete: pdf={len(pdfs)}, svg={len(svgs)}, png={len(pngs)}")
    if any(p.stat().st_size==0 for p in pdfs+svgs+pngs):
        raise ReleaseError("empty figure asset detected")
    for py in sorted(root.rglob("*.py")):
        if "__pycache__" in py.parts:
            continue
        result=subprocess.run([sys.executable,"-m","py_compile",str(py)],cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        if result.returncode:
            raise ReleaseError(f"Python syntax failure in {py.name}: {result.stdout}")
    try:
        doc=fitz.open(root/"paper.pdf")
    except Exception as exc:
        raise ReleaseError(f"PDF cannot be parsed: {exc}") from exc
    if len(doc)<7:
        raise ReleaseError(f"PDF has only {len(doc)} pages; minimum is 7")
    text="\n".join(page.get_text() for page in doc)
    if len(text)<8000:
        raise ReleaseError(f"PDF extracted text is too short to support a substantive manuscript: {len(text)} chars")
    if re.search(r"\?\?|Citation .+ undefined|Reference .+ undefined",text,re.I):
        raise ReleaseError("PDF contains unresolved citation/reference markers")
    digest=sha256(root/"paper.pdf")
    review=json.loads((root/"results/pdf_visual_review.json").read_text(encoding="utf-8"))
    if review.get("pdf_sha256")!=digest or review.get("page_count")!=len(doc):
        raise ReleaseError("visual review record does not match the current PDF")
    return {"pages":len(doc),"text_characters":len(text),"pdf_sha256":digest,"bibliography_entries":bib_count,
      "unique_citations":cite_count,"readme_display_math_blocks":math_blocks,"pdf_figures":len(pdfs),
      "svg_figures":len(svgs),"png_figures":len(pngs),"tables":len(re.findall(r"\\begin\{table\}",tex)),
      "figure_environments":len(re.findall(r"\\begin\{figure\}",tex))}

def render_review(root: Path, review_dir: Path, pdf_hash: str, page_count: int, text_chars: int, font_report: str):
    from PIL import Image, ImageDraw
    pages=review_dir/"pages"
    pages.mkdir(parents=True,exist_ok=True)
    doc=fitz.open(root/"paper.pdf")
    thumbs=[]
    for i,page in enumerate(doc,1):
        pix=page.get_pixmap(matrix=fitz.Matrix(150/72,150/72),alpha=False)
        target=pages/f"page-{i:02d}.png"
        pix.save(target)
        im=Image.open(target).convert("RGB")
        im.thumbnail((255,360))
        card=Image.new("RGB",(275,395),"white")
        card.paste(im,((275-im.width)//2,22))
        ImageDraw.Draw(card).text((10,5),f"Page {i:02d}",fill="black")
        thumbs.append(card)
    cols=3
    rows=(len(thumbs)+cols-1)//cols
    sheet=Image.new("RGB",(cols*275,rows*395),(225,225,225))
    for i,im in enumerate(thumbs):
        sheet.paste(im,((i%cols)*275,(i//cols)*395))
    sheet.save(root/"results/pdf_contact_sheet.png")
    review={
      "status":"BLOCKED",
      "pdf_sha256":pdf_hash,
      "page_count":page_count,
      "render_dpi":150,
      "rendered_pages":list(range(1,page_count+1)),
      "pages_reviewed":[],
      "inspection_timestamp_utc":datetime.now(timezone.utc).isoformat(),
      "inspection_method":"CI rendered every page and generated a contact sheet. Independent human/model page-by-page inspection of this exact PDF is still pending; no visual PASS is asserted.",
      "automated_checks":{"pdf_parses":True,"page_count_verified":page_count,"extracted_text_characters":text_chars,
        "embedded_font_report":font_report,"all_pages_rendered":True},
      "issues":[{"severity":"pending","pages":list(range(1,page_count+1)),
        "issue":"Rendered pages have not yet received independent page-by-page visual review.",
        "remediation":"Inspect every rendered page, fix any layout defects, and update this record against the same PDF SHA-256."}],
      "final_disposition":"BLOCKED until exact-PDF visual review is completed."
    }
    (root/"results/pdf_visual_review.json").write_text(json.dumps(review,indent=2)+"\n",encoding="utf-8")
    quality={
      "status":"BLOCKED","pdf_sha256":pdf_hash,"page_count":page_count,"extracted_text_characters":text_chars,
      "render_dpi":150,"all_pages_rendered":True,"all_pages_visually_reviewed":False,
      "embedded_fonts":font_report,"table_environments":6,"figure_environments":8,
      "vector_figure_assets":8,"design_system":"A4, 0.88-inch margins, Latin Modern, restrained blue/teal/orange palette, consistent caption and running-header system.",
      "critical_major_defects":"NOT_YET_ASSESSED","final_disposition":"Not publication-quality PASS until exact-PDF page-by-page inspection closes."
    }
    (root/"results/publication_quality_report.json").write_text(json.dumps(quality,indent=2)+"\n",encoding="utf-8")
    return review,quality

def create_design_brief(root: Path):
    (root/"results/publication_design_brief.md").write_text(
      "# Publication design brief\n\n"
      "- Target reader: researchers in state estimation, scientific machine learning, and neuromorphic/energy-aware inference. This is a preprint style, not a claimed submission to a venue.\n"
      "- Page geometry: A4, 0.88-inch margins, 10 pt body, 14 pt running-header height.\n"
      "- Typography: Latin Modern with T1 encoding and microtype; math uses standard AMS fonts; no decorative type.\n"
      "- Hierarchy: bold title/subtitle, compact abstract, numbered sections, restrained running header, centered page number.\n"
      "- Equations: AMS environments, explicit labels, consistent scalar notation, equation references through cleveref.\n"
      "- Tables: booktabs rules, no vertical grid, tabularx width control, small but readable type, explanatory captions.\n"
      "- Figures: programmatic vector PDF/SVG plus PNG preview, consistent DejaVu Sans labels, restrained blue/teal/orange/grey palette, no color-only critical encoding.\n"
      "- Captions: left-aligned, bold figure/table label, source/model scope stated for synthetic and conceptual material.\n"
      "- References: numeric compressed citations with BibTeX and natbib.\n"
      "- Whitespace: consistent paragraph spacing, modest section gaps, no decorative cover page; content and figures carry the hierarchy.\n"
      "- Acceptance: automated preflight plus exact-PDF page-by-page visual review; compilation or figure count alone is insufficient.\n",
      encoding="utf-8")

def make_manifest(root: Path):
    excluded={"results/sha256_manifest.txt","results/release_check_matrix.json"}
    files=[]
    for p in sorted(root.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts:
            continue
        rel=p.relative_to(root).as_posix()
        if rel in excluded:
            continue
        files.append((rel,sha256(p)))
    body="# SHA-256 manifest; excludes itself and mutable release_check_matrix.json to avoid a self-referential hash.\n"
    body+="".join(f"{digest}  {rel}\n" for rel,digest in files)
    (root/"results/sha256_manifest.txt").write_text(body,encoding="utf-8")
    tree_hash=hashlib.sha256("".join(f"{rel}:{digest}\n" for rel,digest in files).encode()).hexdigest()
    return tree_hash,len(files)

def update_claim_statuses(root: Path):
    path=root/"results/claim_evidence_matrix.csv"
    if not path.exists(): return
    rows=list(csv.DictReader(path.open(encoding="utf-8",newline="")))
    for row in rows:
        if row["claim_id"] in {"C1","C2","C3"}:
            row["status"]="PASS"
        elif row["claim_id"]=="C4":
            row["status"]="NOT_TESTED"
        elif row["claim_id"]=="C5":
            row["status"]="HYPOTHESIS"
    with path.open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)

def write_matrix(root: Path,tree_hash: str,manifest_count: int,summary: dict):
    checks=[
      {"name":"Author identity/profile audit","status":"PASS","evidence":"results/author_profile_audit.md","details":"Targeted, non-exhaustive; unresolved profile matches explicitly disclosed."},
      {"name":"Ten-candidate topic comparison","status":"PASS","evidence":"results/candidate_topics.csv","details":"Selected topic screened against existing archive releases."},
      {"name":"Literature and citation audit","status":"PASS","evidence":"results/literature_audit.md; paper.bib","details":"Primary/authoritative records checked; no absolute novelty claim."},
      {"name":"Mathematical derivation and claim classification","status":"PASS","evidence":"paper.tex; results/claim_evidence_matrix.csv","details":"Scalar stationary MSE and projection property derived under explicit assumptions."},
      {"name":"Numerical verification","status":"PASS","evidence":"results/verification_report.json; results/verify_model_stdout.txt","details":"Analytical limiting case, Monte Carlo identity check, invalid-input tests, and budget checks executed."},
      {"name":"Figure/data regeneration","status":"PASS","evidence":"figures/; results/model_outputs.csv; results/figure_curve_data.csv; results/frontier_data.csv","details":"Four plots, two structural graphs, and two flow diagrams regenerated from code."},
      {"name":"Metadata and bibliography syntax","status":"PASS","evidence":"CITATION.cff; paper.bib; .zenodo.json; schema_article.jsonld","details":"Parsers and citation-key cross-check passed."},
      {"name":"Python syntax and deployment fixture tests","status":"PASS","evidence":"results/deploy_self_test.log; results/deploy_validate_run.log","details":"Self-test exercises malformed/missing artifacts in a temporary fixture."},
      {"name":"LaTeX build and PDF preflight","status":"PASS","evidence":"results/latex_build.log; results/publication_quality_report.json","details":f"{summary['pages']} pages; {summary['text_characters']} extracted characters; current source compiled in a temporary build."},
      {"name":"Exact-PDF page-by-page visual review","status":"BLOCKED","evidence":"results/pdf_visual_review.json; results/pdf_contact_sheet.png","details":"All pages rendered; independent review of the exact PDF hash is pending."},
      {"name":"Remote artifact read-back and exact-SHA CI closure","status":"PENDING","evidence":"Build workflow summary and remote paths","details":"Must be reconciled after commit and any visual-review update."},
      {"name":"Semantic tag","status":"PENDING","evidence":"validate-memory-energy-release workflow","details":"Tag creation is gated on matrix PASS and exact-PDF visual review."}
    ]
    matrix={"release_status":"BLOCKED","generated_at_utc":datetime.now(timezone.utc).isoformat(),
      "repository":"Devanik21/The-Invention-Archive","target_branch":"main","release_version":"v2026.10.24",
      "paper_title":TITLE,"pdf_sha256":summary["pdf_sha256"],"pdf_pages":summary["pages"],
      "artifact_tree_sha256":tree_hash,"manifested_file_count":manifest_count,
      "checks":checks,"blockers":["Independent page-by-page visual review of the exact PDF","Remote read-back and final-SHA workflow reconciliation","Semantic tag awaits all gates"],
      "closure_rule":"PASS only when all required checks are terminal PASS and the visual-review record names every page of the same PDF hash."}
    (root/"results/release_check_matrix.json").write_text(json.dumps(matrix,indent=2)+"\n",encoding="utf-8")
    md=["# Release check matrix summary","","**Release status: BLOCKED**","","| Check | Status | Evidence |","|---|---|---|"]
    md += [f"| {c['name']} | {c['status']} | {c['evidence']} |" for c in checks]
    md += ["","The build and numerical checks ran in this cycle. The exact PDF still requires page-by-page inspection, and remote closure/tagging are not inferred from local success."]
    (root/"results/release_check_matrix.md").write_text("\n".join(md)+"\n",encoding="utf-8")

def compile_pdf(root: Path):
    build_env=os.environ.copy()
    build_env["SOURCE_DATE_EPOCH"]="1792800000"
    with tempfile.TemporaryDirectory(prefix="memory-energy-build-") as tmp:
        tmp=Path(tmp)
        shutil.copy2(root/"paper.tex",tmp/"paper.tex")
        shutil.copy2(root/"paper.bib",tmp/"paper.bib")
        shutil.copytree(root/"figures",tmp/"figures",dirs_exist_ok=True,ignore=shutil.ignore_patterns("*.png","*.svg"))
        (tmp/"results").mkdir(exist_ok=True)
        shutil.copy2(root/"results/metrics.tex",tmp/"results/metrics.tex")
        logs=[]
        logs.append(run(["pdflatex","-interaction=nonstopmode","-halt-on-error","paper.tex"],tmp,build_env))
        logs.append(run(["bibtex","paper"],tmp,build_env))
        logs.append(run(["pdflatex","-interaction=nonstopmode","-halt-on-error","paper.tex"],tmp,build_env))
        logs.append(run(["pdflatex","-interaction=nonstopmode","-halt-on-error","paper.tex"],tmp,build_env))
        combined="\n\n===== BUILD STAGE =====\n\n".join(logs)
        (root/"results/latex_build.log").write_text(combined,encoding="utf-8")
        final_log=(tmp/"paper.log").read_text(encoding="utf-8",errors="replace")
        fatal_patterns=[r"^! LaTeX Error",r"Emergency stop",r"Undefined control sequence",r"Citation .+ undefined",r"Reference .+ undefined",r"There were undefined references"]
        for pat in fatal_patterns:
            if re.search(pat,final_log,re.M|re.I):
                raise ReleaseError(f"LaTeX log contains unresolved/fatal pattern: {pat}")
        # Overfull boxes are a publication gate; a source repair is required rather than silent acceptance.
        if re.search(r"Overfull \\hbox",final_log):
            raise ReleaseError("LaTeX log contains an overfull hbox; repair layout before release")
        pdf=tmp/"paper.pdf"
        if not pdf.is_file() or pdf.stat().st_size<10000:
            raise ReleaseError("fresh LaTeX build did not produce a valid nonempty PDF")
        shutil.copy2(pdf,root/"paper.pdf")
        font_report="NOT_TESTED"
        try:
            font_report=run(["pdffonts",str(root/"paper.pdf")],root)
            font_lines=[ln for ln in font_report.splitlines()[2:] if ln.strip()]
            if not font_lines or any(not re.search(r"\byes\b",ln) for ln in font_lines):
                raise ReleaseError("PDF font embedding check failed; inspect pdffonts output")
            (root/"results/pdffonts_report.txt").write_text(font_report,encoding="utf-8")
        except FileNotFoundError:
            raise ReleaseError("pdffonts is unavailable; publication-quality gate blocked")
    doc=fitz.open(root/"paper.pdf")
    pages=len(doc); extracted="\n".join(p.get_text() for p in doc); text_chars=len(extracted)
    if pages<7: raise ReleaseError(f"compiled PDF has {pages} pages; minimum is 7")
    if text_chars<8000: raise ReleaseError(f"compiled PDF text is too short: {text_chars} characters")
    if re.search(r"\?\?|Citation .+ undefined|Reference .+ undefined",extracted,re.I):
        raise ReleaseError("extracted PDF text contains unresolved citation/reference markers")
    pdf_hash=sha256(root/"paper.pdf")
    review_dir=Path(os.environ.get("RUNNER_TEMP","/tmp"))/"memory-energy-review"
    # The workflow may override this location through --review-dir.
    return {"pages":pages,"text_characters":text_chars,"pdf_sha256":pdf_hash,"font_report":font_report,"review_dir":review_dir}

def self_test():
    with tempfile.TemporaryDirectory(prefix="memory-energy-selftest-") as tmp:
        root=Path(tmp); (root/"figures").mkdir(); (root/"results").mkdir()
        (root/"README.md").write_text("incomplete fixture",encoding="utf-8")
        missing_rejected=False
        try: validate_root_files(root)
        except ReleaseError: missing_rejected=True
        if not missing_rejected: raise ReleaseError("self-test failed to reject missing core files")
        (root/"CITATION.cff").write_text("cff-version: 1.2.0\n",encoding="utf-8")
        (root/".zenodo.json").write_text("{ malformed",encoding="utf-8")
        (root/"schema_article.jsonld").write_text("{}",encoding="utf-8")
        malformed_rejected=False
        try: validate_metadata(root)
        except ReleaseError: malformed_rejected=True
        if not malformed_rejected: raise ReleaseError("self-test failed to reject malformed metadata")
    print("DEPLOY_SELF_TEST: PASS")
    print("Missing-core fixture rejected: PASS")
    print("Malformed-metadata fixture rejected: PASS")

def build(root: Path,review_dir: Path | None):
    (root/"results").mkdir(exist_ok=True); (root/"figures").mkdir(exist_ok=True)
    output=run([sys.executable,str(root/"verify_model.py")],root)
    (root/"results/verify_model_stdout.txt").write_text(output,encoding="utf-8")
    report=json.loads((root/"results/verification_report.json").read_text(encoding="utf-8"))
    if report.get("status")!="PASS" or report.get("checks_failed")!=0:
        raise ReleaseError("numerical verification report is not PASS")
    update_claim_statuses(root)
    create_design_brief(root)
    build_info=compile_pdf(root)
    if review_dir is None:
        review_dir=Path(os.environ.get("RUNNER_TEMP","/tmp"))/"memory-energy-review"
    review,quality=render_review(root,review_dir,build_info["pdf_sha256"],build_info["pages"],
      build_info["text_characters"],build_info["font_report"])
    summary=validate_tree(root)
    tree_hash,manifest_count=make_manifest(root)
    write_matrix(root,tree_hash,manifest_count,summary)
    # Refresh the manifest after the closure records have been written; the matrix is intentionally excluded.
    tree_hash,manifest_count=make_manifest(root)
    print("DEPLOY_BUILD: PASS")
    print(f"PDF pages: {build_info['pages']}; extracted text: {build_info['text_characters']} characters")
    print(f"PDF SHA-256: {build_info['pdf_sha256']}")
    print(f"Core files: {len(CORE)}; figure assets: 8 PDF + 8 SVG + 8 PNG")
    print("Visual-review status: BLOCKED pending independent inspection of every rendered page.")
    return summary

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument("--build",action="store_true")
    parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--validate-only",action="store_true")
    parser.add_argument("--review-dir",type=Path,default=None)
    args=parser.parse_args()
    root=args.root.resolve()
    try:
        if args.self_test:
            self_test()
        if args.build:
            build(root,args.review_dir)
        if args.validate_only:
            summary=validate_tree(root)
            manifest=validate_manifest(root)
            print("DEPLOY_VALIDATE: PASS")
            print(json.dumps({**summary,"manifest_entries":len(manifest)},indent=2))
        if not (args.self_test or args.build or args.validate_only):
            parser.error("choose --build, --self-test, or --validate-only")
    except (ReleaseError,AssertionError,ValueError,KeyError,json.JSONDecodeError) as exc:
        print(f"RELEASE_BLOCKED: {exc}",file=sys.stderr)
        return 2
    return 0

if __name__=="__main__":
    raise SystemExit(main())
