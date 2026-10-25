from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import fitz
import yaml
from PIL import Image, ImageOps, ImageDraw

ROOT=Path(__file__).resolve().parent
CORE=["paper.tex","paper.pdf","verify_model.py","README.md","CITATION.cff","paper.bib",".zenodo.json","schema_article.jsonld","deploy.py"]
MUTABLE_AUDIT={"results/sha256_manifest.json","results/pdf_visual_review.json","results/publication_quality_report.md","results/release_check_matrix.json"}

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def run(cmd,cwd,log_path=None):
    p=subprocess.run(cmd,cwd=cwd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if log_path:
        Path(log_path).write_text(p.stdout,encoding="utf-8")
    if p.returncode:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(map(str,cmd))}\n{p.stdout[-6000:]}")
    return p.stdout

def metadata_errors(root,check_links=True):
    errors=[]
    try:
        cff=yaml.safe_load((root/"CITATION.cff").read_text(encoding="utf-8"))
        if not isinstance(cff,dict) or cff.get("cff-version")!="1.2.0" or not cff.get("title") or not cff.get("authors"):
            errors.append("CITATION.cff schema fields missing")
        expected_title="The Rejuvenation Mirage: Identifiability Limits of Epigenetic-Clock Reset and an Active-Design Rule for Functional Repair"
        if isinstance(cff,dict) and cff.get("title")!=expected_title: errors.append("CITATION.cff title does not match manuscript")
    except Exception as e: errors.append(f"CITATION.cff parse error: {e}")
    for name in [".zenodo.json","schema_article.jsonld"]:
        try:
            obj=json.loads((root/name).read_text(encoding="utf-8"))
            if not isinstance(obj,dict): errors.append(f"{name} is not a JSON object")
            if isinstance(obj,dict) and obj.get("title",obj.get("name"))!="The Rejuvenation Mirage: Identifiability Limits of Epigenetic-Clock Reset and an Active-Design Rule for Functional Repair":
                errors.append(f"{name} title does not match manuscript")
        except Exception as e: errors.append(f"{name} parse error: {e}")
    tex=(root/"paper.tex").read_text(encoding="utf-8")
    bib=(root/"paper.bib").read_text(encoding="utf-8")
    bibkeys=set(re.findall(r"@\w+\s*\{\s*([^,\s]+)",bib))
    cites=set()
    for group in re.findall(r"\\cite[a-zA-Z*]*\s*\{([^}]+)\}",tex):
        cites.update(x.strip() for x in group.split(",") if x.strip())
    missing=sorted(cites-bibkeys)
    if missing: errors.append("Missing BibTeX keys: "+", ".join(missing))
    if not re.search(r"\\bibliography\{paper\}",tex): errors.append("paper.tex does not consume paper.bib")
    readme=(root/"README.md").read_text(encoding="utf-8")
    if check_links:
        for target in re.findall(r"\]\(([^)]+)\)",readme):
            if target.startswith(("http://","https://","#","mailto:")): continue
            local=(root/target.split("#",1)[0])
            if not local.exists(): errors.append(f"Broken local README link: {target}")
    if readme.count("$$")%2: errors.append("Unbalanced README display-math delimiters")
    if r"\operatorname" in readme: errors.append("Prohibited operatorname macro in README")
    if re.search(r"@@[A-Z_]+@@",readme): errors.append("Unreplaced generated-result marker in README")
    return errors

def figure_counts(root):
    inv=root/"results/figure_inventory.csv"
    if not inv.exists(): return {"total":0,"plot":0,"graph":0,"flow":0}
    with inv.open(newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
    return {"total":len(rows),"plot":sum(r.get("kind")=="plot" for r in rows),
            "graph":sum(r.get("kind")=="graph" for r in rows),"flow":sum(r.get("kind")=="flow" for r in rows)}

def pdf_metrics(pdf_path,tex_path):
    doc=fitz.open(pdf_path)
    page_text=[p.get_text("text") for p in doc]
    main_start=None
    for i,text in enumerate(page_text):
        if i>=2 and re.search(r"1\s*\.?\s+Introduction",text):
            main_start=i; break
    ref_start=None
    if main_start is not None:
        for i in range(main_start+1,len(page_text)):
            if re.search(r"^\s*References\s*$",page_text[i],re.M):
                ref_start=i; break
    main_pages=(ref_start-main_start) if main_start is not None and ref_start is not None else 0
    full_text="\n".join(page_text)
    figs=len(set(re.findall(r"Figure\s+(\d+)",full_text)))
    tabs=len(set(re.findall(r"Table\s+(\d+)",full_text)))
    return {"pages":len(doc),"main_start_page":(main_start+1 if main_start is not None else None),
            "references_start_page":(ref_start+1 if ref_start is not None else None),
            "main_text_pages":main_pages,"distinct_figures_in_pdf":figs,"distinct_tables_in_pdf":tabs,
            "text":full_text}

def write_visual_artifacts(root,pdf_path,metrics):
    page_dir=root/"results/pdf_pages"
    if page_dir.exists(): shutil.rmtree(page_dir)
    page_dir.mkdir(parents=True)
    doc=fitz.open(pdf_path); thumbs=[]
    dpi=120
    matrix=fitz.Matrix(dpi/72,dpi/72)
    for i,page in enumerate(doc):
        pix=page.get_pixmap(matrix=matrix,alpha=False)
        path=page_dir/f"page-{i+1:02d}.png"; pix.save(path)
        im=Image.open(path).convert("RGB")
        im.thumbnail((240,340))
        canvas=Image.new("RGB",(250,365),"white")
        canvas.paste(im,((250-im.width)//2,8))
        ImageDraw.Draw(canvas).text((8,345),f"Page {i+1}",fill=(23,50,77))
        thumbs.append(canvas)
    cols=5; rows=(len(thumbs)+cols-1)//cols
    sheet=Image.new("RGB",(cols*250,rows*365),(238,244,246))
    for i,im in enumerate(thumbs): sheet.paste(im,((i%cols)*250,(i//cols)*365))
    sheet.save(root/"results/pdf_contact_sheet.jpg",quality=90)
    return {"status":"RENDERED_PENDING_MANUAL_REVIEW","inspection_timestamp_utc":datetime.now(timezone.utc).isoformat(),
      "paper_pdf_sha256":sha256(pdf_path),"page_count":metrics["pages"],"render_dpi":dpi,
      "all_pages_rendered":True,"pages_rendered":list(range(1,metrics["pages"]+1)),
      "pages_individually_reviewed":[],"contact_sheet_reviewed":False,
      "reviewer":"pending manual review","review_method":"All pages rendered to PNG and a contact sheet was created; rendering is not a manual visual pass.",
      "issues":[],"critical_or_major_defects":[],"final_disposition":"BLOCKED — all pages rendered, but manual page-by-page visual inspection is pending."}

def make_manifest(root):
    files={}
    for p in sorted(root.rglob("*")):
        if not p.is_file(): continue
        rel=p.relative_to(root).as_posix()
        if rel in MUTABLE_AUDIT: continue
        if rel.startswith(".build/") or "__pycache__" in rel: continue
        files[rel]=sha256(p)
    data={"schema_version":"1.0","generated_utc":datetime.now(timezone.utc).isoformat(),
          "algorithm":"SHA-256","self_reference_excluded":True,
          "mutable_closure_records_excluded":sorted(MUTABLE_AUDIT),
          "files":files}
    (root/"results/sha256_manifest.json").write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")

def write_quality_report(root,metrics,counts,pdf_hash):
    text=(
      "# Publication-quality report\n\n"
      "**Status: BLOCKED pending manual visual review.** Automated compilation and PDF preflight are recorded below; they do not certify page-level visual quality.\n\n"
      f"- PDF SHA-256: {pdf_hash}\n- Total pages: {metrics['pages']}\n- Substantive main-text pages by PDF text-boundary check: {metrics['main_text_pages']}\n"
      f"- Distinct numbered figures detected in PDF text: {metrics['distinct_figures_in_pdf']}\n"
      f"- Distinct numbered tables detected in PDF text: {metrics['distinct_tables_in_pdf']}\n"
      f"- Figure inventory: {counts['total']} total; {counts['plot']} plots; {counts['graph']} graphs; {counts['flow']} flow diagrams.\n\n"
      "## Evidence-based scorecard\n\n"
      "| Dimension | Score (1–5) | Evidence / disposition |\n|---|---:|---|\n"
      "| First-page hierarchy | NOT SCORED | Requires manual inspection of the final rendered PDF. |\n"
      "| Typography and mathematical legibility | NOT SCORED | Requires page-by-page inspection and font review. |\n"
      "| Design consistency | NOT SCORED | Requires contact-sheet and individual-page inspection. |\n"
      "| Table quality | NOT SCORED | Requires inspection of all table pages. |\n"
      "| Figure/diagram clarity | NOT SCORED | Requires inspection of all plot, graph, and flow pages. |\n"
      "| Page rhythm and whitespace | NOT SCORED | Requires inspection of every page. |\n"
      "| Section/navigation structure | NOT SCORED | Automated contents/page count checks are not sufficient. |\n"
      "| Overall editorial polish | NOT SCORED | Pending manual review. |\n\n"
      "No visual score is inferred from page count or successful compilation. The release remains blocked until all pages are inspected and every critical/major defect is resolved.\n"
    )
    (root/"results/publication_quality_report.md").write_text(text,encoding="utf-8")

def write_matrix(root,checks,metrics,counts,pdf_hash):
    data={"schema_version":"1.0","execution_timestamp_utc":datetime.now(timezone.utc).isoformat(),
      "repository":"Devanik21/The-Invention-Archive","target_branch":"main",
      "release_slug":"The_Rejuvenation_Mirage_Identifiability_Limits_of_Epigenetic_Clock_Reset_and_Active_Design_for_Functional_Repair",
      "release_status":"BLOCKED","pdf_sha256":pdf_hash,"pdf_pages":metrics["pages"],
      "main_text_pages":metrics["main_text_pages"],"figure_inventory":counts,
      "checks":checks,"blockers":["Manual page-by-page visual inspection and final visual scorecard pending."],
      "matrix_self_reference_note":"This matrix and the visual-review/quality reports are mutable closure records excluded from the SHA-256 manifest to avoid a self-referential hash loop. The final report must name the exact final head and verifier run."}
    (root/"results/release_check_matrix.json").write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")

def validate_release(root,allow_pending_visual=False):
    errors=[]
    for name in CORE:
        if not (root/name).is_file() or (root/name).stat().st_size==0: errors.append(f"Missing or empty core file: {name}")
    top_files=[p.name for p in root.iterdir() if p.is_file()]
    if sorted(top_files)!=sorted(CORE): errors.append(f"Root-level core manifest mismatch: {sorted(top_files)}")
    for d in ["figures","results"]:
        if not (root/d).is_dir(): errors.append(f"Missing directory: {d}")
    errors.extend(metadata_errors(root))
    inv=figure_counts(root)
    if inv["total"]<12 or inv["plot"]<6 or inv["graph"]<2 or inv["flow"]<2:
        errors.append(f"Figure inventory floor failed: {inv}")
    if len(list((root/"figures").glob("fig*.pdf")))<12: errors.append("Fewer than 12 vector PDF figures")
    if (root/"paper.pdf").is_file() and (root/"paper.tex").is_file():
        try:
            m=pdf_metrics(root/"paper.pdf",root/"paper.tex")
            if m["pages"]<25: errors.append(f"PDF has only {m['pages']} pages; floor is 25")
            if m["main_text_pages"]<18: errors.append(f"Substantive main text has {m['main_text_pages']} pages; floor is 18")
            if m["distinct_figures_in_pdf"]<12: errors.append(f"PDF contains only {m['distinct_figures_in_pdf']} distinct numbered figures")
            if m["distinct_tables_in_pdf"]<5: errors.append(f"PDF contains only {m['distinct_tables_in_pdf']} distinct numbered tables")
            if not (root/"results/pdf_visual_review.json").exists(): errors.append("Missing PDF visual-review record")
            else:
                vr=json.loads((root/"results/pdf_visual_review.json").read_text(encoding="utf-8"))
                if vr.get("paper_pdf_sha256")!=sha256(root/"paper.pdf"): errors.append("Visual-review record PDF hash does not match paper.pdf")
                if vr.get("status")!="PASS" and not allow_pending_visual: errors.append("Manual visual-review gate is not PASS")
                if not allow_pending_visual:
                    if len(vr.get("pages_individually_reviewed",[])) != vr.get("page_count"):
                        errors.append("Visual-review record does not list every page as individually reviewed")
                    if vr.get("critical_or_major_defects"):
                        errors.append("Visual-review record contains unresolved critical/major defects")
                    quality=root/"results/publication_quality_report.md"
                    if not quality.exists(): errors.append("Missing publication-quality scorecard")
                    else:
                        qtext=quality.read_text(encoding="utf-8")
                        scores=[int(x) for x in re.findall(r"\|[^|]+\|\s*([1-5])\s*\|",qtext)]
                        if len(scores)<8 or min(scores)<4: errors.append("Publication-quality scorecard does not contain eight scores of at least 4/5")
                    matrix=root/"results/release_check_matrix.json"
                    if not matrix.exists(): errors.append("Missing release check matrix")
                    else:
                        mdata=json.loads(matrix.read_text(encoding="utf-8"))
                        if mdata.get("release_status")!="PASS": errors.append("Release check matrix is not PASS")
                        unresolved=[c for c in mdata.get("checks",[]) if c.get("status") not in ("PASS","NOT_APPLICABLE")]
                        if unresolved: errors.append("Release matrix contains unresolved checks: "+", ".join(c.get("name","unknown") for c in unresolved))
        except Exception as e: errors.append(f"PDF preflight error: {e}")
    else: errors.append("PDF/source missing; build is blocked")
    manifest=root/"results/sha256_manifest.json"
    if manifest.exists():
        try:
            data=json.loads(manifest.read_text(encoding="utf-8"))
            for rel,expected in data.get("files",{}).items():
                p=root/rel
                if not p.is_file() or sha256(p)!=expected: errors.append(f"Manifest mismatch: {rel}")
        except Exception as e: errors.append(f"Manifest parse error: {e}")
    return errors

def self_test(root):
    with tempfile.TemporaryDirectory(prefix="deploy-self-test-") as tmp:
        fixture=Path(tmp)
        for name in ["paper.tex","verify_model.py","README.md","CITATION.cff","paper.bib",".zenodo.json","schema_article.jsonld","deploy.py"]:
            (fixture/name).write_text("{}",encoding="utf-8")
        (fixture/"figures").mkdir(); (fixture/"results").mkdir()
        errors=validate_release(fixture,allow_pending_visual=True)
        if not any("paper.pdf" in e for e in errors): raise RuntimeError("Fail-closed self-test did not reject a missing PDF")
        (fixture/"paper.pdf").write_bytes(b"not a PDF")
        (fixture/".zenodo.json").write_text("{bad json",encoding="utf-8")
        errors=validate_release(fixture,allow_pending_visual=True)
        if not any("parse error" in e or "PDF preflight error" in e for e in errors):
            raise RuntimeError("Fail-closed self-test did not reject malformed artifacts")
    print("deploy.py self-test: PASS; missing and malformed release fixtures were blocked.")

def build(root):
    (root/"results").mkdir(exist_ok=True); (root/"figures").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="python-syntax-check-") as syntax_tmp:
        shutil.copy2(root/"verify_model.py",Path(syntax_tmp)/"verify_model.py")
        shutil.copy2(root/"deploy.py",Path(syntax_tmp)/"deploy.py")
        run([sys.executable,"-m","py_compile","verify_model.py","deploy.py"],syntax_tmp,root/"results/python_compile_stdout.txt")
    out=run([sys.executable,"verify_model.py"],root)
    (root/"results/verify_model_stdout.txt").write_text(out,encoding="utf-8")
    md=metadata_errors(root,check_links=False)
    if md: raise RuntimeError("Metadata/source preflight failed: "+"; ".join(md))
    with tempfile.TemporaryDirectory(prefix="rejuvenation-mirage-build-") as tmp:
        tmp=Path(tmp)
        shutil.copy2(root/"paper.tex",tmp/"paper.tex"); shutil.copy2(root/"paper.bib",tmp/"paper.bib")
        shutil.copytree(root/"figures",tmp/"figures",dirs_exist_ok=True)
        shutil.copy2(root/"results/generated_macros.tex",tmp/"results_generated_macros.tex")
        (tmp/"results").mkdir(exist_ok=True)
        # Keep the manuscript's relative generated-macro path stable.
        shutil.copy2(root/"results/generated_macros.tex",tmp/"results/generated_macros.tex")
        stdout=run(["latexmk","-pdf","-interaction=nonstopmode","-halt-on-error","-file-line-error","-outdir=build","paper.tex"],tmp)
        (root/"results/latex_build_stdout.txt").write_text(stdout,encoding="utf-8")
        built=tmp/"build/paper.pdf"
        if not built.is_file() or built.stat().st_size<10000: raise RuntimeError("Current LaTeX build did not produce a nonempty PDF")
        log=(tmp/"build/paper.log").read_text(encoding="utf-8",errors="replace") if (tmp/"build/paper.log").exists() else ""
        if re.search(r"Undefined control sequence|Citation .+ undefined|Reference .+ undefined|Emergency stop|Fatal error",log):
            raise RuntimeError("LaTeX log contains a fatal or unresolved reference/citation marker")
        metrics=pdf_metrics(built,root/"paper.tex")
        if metrics["pages"]<25: raise RuntimeError(f"PDF page floor failed: {metrics['pages']} < 25")
        if metrics["main_text_pages"]<18: raise RuntimeError(f"Substantive main-text floor failed: {metrics['main_text_pages']} < 18")
        if metrics["distinct_figures_in_pdf"]<12: raise RuntimeError(f"Figure floor failed in PDF text: {metrics['distinct_figures_in_pdf']} < 12")
        if metrics["distinct_tables_in_pdf"]<5: raise RuntimeError(f"Table floor failed in PDF text: {metrics['distinct_tables_in_pdf']} < 5")
        shutil.copy2(built,root/"paper.pdf")
    pdf=root/"paper.pdf"; pdfhash=sha256(pdf); metrics=pdf_metrics(pdf,root/"paper.tex"); counts=figure_counts(root)
    (root/"results/pdfinfo.txt").write_text(run(["pdfinfo","paper.pdf"],root),encoding="utf-8")
    fonts=run(["pdffonts","paper.pdf"],root)
    (root/"results/pdffonts_report.txt").write_text(fonts,encoding="utf-8")
    font_lines=[line for line in fonts.splitlines()[2:] if line.strip()]
    if not font_lines or any(len(line.split())<6 or line.split()[3].lower()!="yes" for line in font_lines):
        raise RuntimeError("PDF font embedding preflight failed or returned no fonts")
    extracted=metrics.pop("text")
    (root/"results/pdf_text_extract.txt").write_text(extracted,encoding="utf-8")
    visual=write_visual_artifacts(root,pdf,metrics)
    (root/"results/pdf_visual_review.json").write_text(json.dumps(visual,indent=2)+"\n",encoding="utf-8")
    write_quality_report(root,metrics,counts,pdfhash)
    checks=[
      {"name":"Scientific/model checks","status":"PASS","evidence":"results/verification_report.json"},
      {"name":"Python syntax","status":"PASS","evidence":"results/python_compile_stdout.txt"},
      {"name":"LaTeX build from current source","status":"PASS","evidence":"results/latex_build_stdout.txt"},
      {"name":"PDF total/main-text page floors","status":"PASS","evidence":"results/pdfinfo.txt and results/pdf_text_extract.txt"},
      {"name":"Figure/table inventory","status":"PASS","evidence":"results/figure_inventory.csv"},
      {"name":"Metadata and bibliography preflight","status":"PASS","evidence":"CITATION.cff, paper.bib, .zenodo.json, schema_article.jsonld"},
      {"name":"PDF visual inspection/publication-quality gate","status":"BLOCKED","evidence":"results/pdf_visual_review.json; manual inspection pending"},
      {"name":"Remote read-back and required CI on exact final SHA","status":"PENDING","evidence":"to be recorded after GitHub publication"},
    ]
    write_matrix(root,checks,metrics,counts,pdfhash)
    make_manifest(root)
    final_md=metadata_errors(root,check_links=True)
    if final_md: raise RuntimeError('Final metadata/README validation failed: '+'; '.join(final_md))
    print(json.dumps({"build":"PASS","pdf_sha256":pdfhash,"pdf_pages":metrics["pages"],
      "main_text_pages":metrics["main_text_pages"],"figures":counts,"visual_status":visual["status"],
      "release_status":"BLOCKED pending manual visual review"},indent=2))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--build",action="store_true")
    parser.add_argument("--validate-only",action="store_true")
    parser.add_argument("--allow-pending-visual",action="store_true")
    args=parser.parse_args()
    if args.self_test: self_test(ROOT); return
    if args.build: build(ROOT); return
    if args.validate_only:
        errors=validate_release(ROOT,allow_pending_visual=args.allow_pending_visual)
        if errors:
            print("BLOCKED:")
            for e in errors: print(" - "+e)
            raise SystemExit(1)
        print("Release validation: PASS")
        return
    parser.error("Choose --self-test, --build, or --validate-only")

if __name__=="__main__": main()
