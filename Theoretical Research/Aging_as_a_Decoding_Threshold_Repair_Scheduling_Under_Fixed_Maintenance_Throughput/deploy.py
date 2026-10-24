#!/usr/bin/env python3
"""Non-destructive validation helper for this research release."""
from pathlib import Path
import argparse, json, subprocess, sys, tempfile, shutil, re
EXPECTED={"paper.tex","paper.pdf","verify_model.py","README.md","CITATION.cff","paper.bib",".zenodo.json","schema_article.jsonld","deploy.py"}
class ReleaseError(RuntimeError): pass

def validate(root: Path):
    root=root.resolve()
    files={p.name for p in root.iterdir() if p.is_file()}
    if files != EXPECTED: raise ReleaseError(f"root manifest mismatch: missing={sorted(EXPECTED-files)} extra={sorted(files-EXPECTED)}")
    for d in (root/"figures",root/"results"):
        if not d.is_dir(): raise ReleaseError(f"missing required directory {d.name}")
    pdfs=list((root/"figures").glob("figure*.pdf")); pngs=list((root/"figures").glob("figure*.png"))
    if len(pdfs)<8 or len(pngs)<8: raise ReleaseError(f"expected at least eight PDFs and PNGs; got {len(pdfs)} PDFs, {len(pngs)} PNGs")
    if not (root/"paper.pdf").is_file() or (root/"paper.pdf").stat().st_size<5000: raise ReleaseError("paper.pdf is missing or too small")
    if (root/"paper.pdf").read_bytes()[:5] != b"%PDF-": raise ReleaseError("invalid PDF signature")
    if not (root/"results"/"verification_report.json").exists(): raise ReleaseError("missing verification report")
    report=json.loads((root/"results"/"verification_report.json").read_text(encoding="utf-8"))
    if report.get("status") != "PASS" or report.get("checks_failed") != 0: raise ReleaseError("model verification did not pass")
    for name in (".zenodo.json","schema_article.jsonld"):
        json.loads((root/name).read_text(encoding="utf-8"))
    import yaml
    cff=yaml.safe_load((root/"CITATION.cff").read_text(encoding="utf-8"))
    if cff.get("cff-version")!="1.2.0" or not cff.get("authors"): raise ReleaseError("CITATION.cff invalid")
    tex=(root/"paper.tex").read_text(encoding="utf-8")
    for token in ("\\begin{table}","\\begin{figure}","Co-equal research-process credit is given to Devanik Debnath and Xylia"):
        if token not in tex: raise ReleaseError(f"paper.tex missing required item: {token}")
    bib=(root/"paper.bib").read_text(encoding="utf-8")
    keys=set(re.findall(r"@\w+\s*\{\s*([^,\s]+)",bib))
    cited=set()
    for block in re.findall(r"\\cite\w*\{([^}]+)\}",tex): cited.update(x.strip() for x in block.split(","))
    if cited and not cited.issubset(keys): raise ReleaseError(f"missing bibliography keys: {sorted(cited-keys)}")
    for py in (root/"verify_model.py",root/"deploy.py"):
        r=subprocess.run([sys.executable,"-m","py_compile",str(py)],capture_output=True,text=True)
        if r.returncode: raise ReleaseError(r.stderr)
    return {"root_files":sorted(files),"figure_pdf_count":len(pdfs),"figure_png_count":len(pngs),"verification_status":report["status"]}

def self_test():
    with tempfile.TemporaryDirectory() as td:
        p=Path(td); (p/"README.md").write_text("test")
        try: validate(p)
        except Exception: pass
        else: raise ReleaseError("negative fixture was incorrectly accepted")
    return True

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=Path(__file__).resolve().parent);ap.add_argument("--self-test",action="store_true");ap.add_argument("--validate-only",action="store_true")
    a=ap.parse_args()
    try:
        if a.self_test:
            self_test(); print("PASS: malformed fixture correctly rejected")
        if a.validate_only or not a.self_test: print(json.dumps(validate(a.root),indent=2))
    except Exception as e: print(f"BLOCKED: {e}",file=sys.stderr);sys.exit(2)
