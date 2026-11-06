#!/usr/bin/env python3
"""Fail-closed build and release validation for the Burst-Robust Codes preprint."""
from __future__ import annotations
import argparse, csv, hashlib, json, re, subprocess, sys, importlib.util
from pathlib import Path
import fitz

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
REQUIRED_ROOT_FILES = {
    "paper.tex", "paper.pdf", "paper.bib", "verify_model.py", "deploy.py",
    "README.md", "CITATION.cff", ".zenodo.json", "schema_article.jsonld"
}
REQUIRED_DIRS = {"figures", "results"}


def run(cmd: list[str], *, cwd: Path = ROOT, stdout=None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True, stdout=stdout)


def source_metrics() -> dict:
    tex = (ROOT / "paper.tex").read_text(encoding="utf-8")
    bib = (ROOT / "paper.bib").read_text(encoding="utf-8")
    table_count = len(re.findall(r"\\begin\{table\*?\}", tex))
    figure_count = len(re.findall(r"\\begin\{figure\*?\}", tex))
    bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,]+),", bib))
    cite_keys = set()
    for group in re.findall(r"\\cite[a-zA-Z]*\{([^}]+)\}", tex):
        cite_keys.update(x.strip() for x in group.split(",") if x.strip())
    missing_cites = sorted(cite_keys - bib_keys)
    inventory_path = RESULTS / "figure_inventory.csv"
    inventory = list(csv.DictReader(inventory_path.open(encoding="utf-8"))) if inventory_path.exists() else []
    return {
        "table_count": table_count,
        "figure_count": figure_count,
        "bibliography_entries": len(bib_keys),
        "citation_keys": len(cite_keys),
        "missing_citations": missing_cites,
        "figure_inventory": inventory,
    }


def pdf_metrics() -> dict:
    pdf = ROOT / "paper.pdf"
    if not pdf.is_file() or pdf.stat().st_size == 0:
        raise RuntimeError("paper.pdf is missing or empty")
    doc = fitz.open(pdf)
    pages = [" ".join(p.get_text().split()) for p in doc]
    intro = None
    appendix = None
    for i, text in enumerate(pages):
        if intro is None and "1 Introduction: reliability under spatially extended events" in text and "The reliability of a complex system" in text:
            intro = i
        if intro is not None and appendix is None and "A Exact expected-occupancy calibration" in text:
            appendix = i
    if intro is None or appendix is None or appendix < intro:
        raise RuntimeError("Unable to locate main-text and appendix boundaries in extracted PDF text")
    main_pages = appendix - intro + 1
    main_words = sum(len(pages[i].split()) for i in range(intro, appendix + 1))
    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
    return {
        "page_count": len(doc),
        "main_text_pages": main_pages,
        "main_text_words_in_boundary": main_words,
        "pdf_sha256": digest,
        "page_text": pages,
        "page_word_counts": [len(x.split()) for x in pages],
    }


def validate(*, allow_pending_visual: bool = False, check_root_layout: bool = True) -> dict:
    sm = source_metrics()
    pm = pdf_metrics()
    errors = []
    if not 28 <= pm["page_count"] <= 40:
        errors.append(f"PDF page count {pm['page_count']} is outside 28–40")
    if pm["main_text_pages"] < 18:
        errors.append(f"Main text has only {pm['main_text_pages']} pages; minimum is 18")
    if sm["table_count"] < 5:
        errors.append(f"Only {sm['table_count']} substantive table environments")
    if sm["figure_count"] != 12:
        errors.append(f"Expected 12 numbered figure environments; found {sm['figure_count']}")
    if len(sm["figure_inventory"]) != 12:
        errors.append(f"Figure inventory contains {len(sm['figure_inventory'])} rows, not 12")
    kinds = [x.get("kind", "").lower() for x in sm["figure_inventory"]]
    if kinds.count("plot") < 6 or kinds.count("graph") < 2 or kinds.count("flow") < 2:
        errors.append(f"Figure type floors not met: {kinds.count('plot')} plots, {kinds.count('graph')} graphs, {kinds.count('flow')} flows")
    pdf_figures = list(FIGURES.glob("fig*.pdf"))
    png_figures = list(FIGURES.glob("fig*.png"))
    if len(pdf_figures) != 12 or len(png_figures) != 12:
        errors.append(f"Expected 12 vector and 12 PNG figures; found {len(pdf_figures)} PDF and {len(png_figures)} PNG")
    if sm["missing_citations"]:
        errors.append("Missing bibliography keys: " + ", ".join(sm["missing_citations"]))
    if sm["bibliography_entries"] < 15:
        errors.append(f"Only {sm['bibliography_entries']} bibliography entries")
    for name in ("verification_report.json", "figure_inventory.csv", "spanning_sweep.csv"):
        if not (RESULTS / name).is_file():
            errors.append(f"Required reproducibility artifact missing: results/{name}")
    try:
        vr = json.loads((RESULTS / "verification_report.json").read_text(encoding="utf-8"))
        if vr.get("status") != "PASS" or vr.get("checks_failed") != 0:
            errors.append("Model verification report is not PASS")
        if vr.get("checks_passed", 0) < 10:
            errors.append("Fewer than 10 model validation checks are recorded")
    except Exception as exc:
        errors.append(f"Verification report is unreadable: {exc}")
    for name, reader in ((".zenodo.json", json.loads), ("schema_article.jsonld", json.loads)):
        try:
            reader((ROOT / name).read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"Metadata file {name} is invalid: {exc}")
    cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    if "cff-version: 1.2.0" not in cff or "date-released: 2026-11-06" not in cff:
        errors.append("CITATION.cff is missing required release metadata")
    review_path = RESULTS / "pdf_visual_review.json"
    if review_path.exists():
        review = json.loads(review_path.read_text(encoding="utf-8"))
        if review.get("paper_pdf_sha256") != pm["pdf_sha256"]:
            errors.append("Visual-review record does not match the current PDF SHA-256")
        if review.get("page_count") != pm["page_count"] or not review.get("all_pages_rendered"):
            errors.append("Visual-review record does not cover all current PDF pages")
    elif not allow_pending_visual:
        errors.append("Visual-review record is missing")
    if check_root_layout:
        root_files = {p.name for p in ROOT.iterdir() if p.is_file()}
        root_dirs = {p.name for p in ROOT.iterdir() if p.is_dir()}
        if root_files != REQUIRED_ROOT_FILES:
            errors.append(f"Root release files mismatch; missing={sorted(REQUIRED_ROOT_FILES-root_files)}, extra={sorted(root_files-REQUIRED_ROOT_FILES)}")
        if root_dirs != REQUIRED_DIRS:
            errors.append(f"Root directories mismatch; expected {sorted(REQUIRED_DIRS)}, found {sorted(root_dirs)}")
    result = {"status": "PASS" if not errors else "FAIL", "errors": errors,
              "page_count": pm["page_count"], "main_text_pages": pm["main_text_pages"],
              "main_text_words_in_boundary": pm["main_text_words_in_boundary"],
              "pdf_sha256": pm["pdf_sha256"], "tables": sm["table_count"],
              "figure_environments": sm["figure_count"], "figure_classes": {
                  "plot": kinds.count("plot"), "graph": kinds.count("graph"), "flow": kinds.count("flow")},
              "bibliography_entries": sm["bibliography_entries"]}
    if errors:
        raise RuntimeError(json.dumps(result, indent=2))
    return result


def self_test() -> dict:
    spec = importlib.util.spec_from_file_location("burst_model", ROOT / "verify_model.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    q = module.hit_probabilities(24, 5)
    occ = [module.expected_occupancy(module.calibrated_intensity(p, q), q) for p in (0.2, 0.4, 0.6)]
    assert all(abs(a-b) < 1e-10 for a,b in zip(occ, (0.2,0.4,0.6)))
    assert module.wilson(0, 4000)[0] == 0.0
    assert 0.0 < module.wilson(0, 4000)[1] < 0.01
    assert module.wilson(4000,4000)[1] == 1.0
    assert abs((1.0 - __import__('math').exp(-0.3*0.2*10)) - 0.4511883639059736) < 1e-12
    return {"status":"PASS","checks":5,"details":["occupancy calibration","zero-count Wilson lower bound","all-success Wilson upper bound","Poisson thinning expression","model module import"]}


def build() -> dict:
    run(["latexmk", "-C", "paper.tex"])
    with (RESULTS / "latex_build_stdout.txt").open("w", encoding="utf-8") as log:
        run(["latexmk", "-xelatex", "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", "paper.tex"], stdout=log)
    # Fail closed on unresolved cross-references/citations and fatal glyph warnings.
    latex_log = (ROOT / "paper.log").read_text(encoding="utf-8", errors="replace")
    if "There were undefined references" in latex_log or re.search(r"Reference `[^']+' .*undefined", latex_log):
        raise RuntimeError("LaTeX build contains unresolved references")
    if re.search(r"Citation `[^']+' .*undefined", latex_log):
        raise RuntimeError("LaTeX build contains unresolved citations")
    run(["pdfinfo", "paper.pdf"], stdout=(RESULTS / "pdfinfo.txt").open("w", encoding="utf-8"))
    run(["pdffonts", "paper.pdf"], stdout=(RESULTS / "pdffonts_report.txt").open("w", encoding="utf-8"))
    run(["pdftotext", "-layout", "paper.pdf", "results/pdf_text_extract.txt"])
    pm = pdf_metrics()
    (RESULTS / "pdf_page_text_index.json").write_text(json.dumps({"pdf_sha256":pm["pdf_sha256"],"page_count":pm["page_count"],"pages":[{"page":i+1,"word_count":pm["page_word_counts"][i],"opening_text":pm["page_text"][i][:220]} for i in range(pm["page_count"])]},indent=2)+"\n", encoding="utf-8")
    return validate(check_root_layout=False)


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--self-test",action="store_true")
    ap.add_argument("--build",action="store_true")
    ap.add_argument("--validate-only",action="store_true")
    ap.add_argument("--allow-pending-visual",action="store_true")
    args=ap.parse_args()
    try:
        if args.self_test:
            print(json.dumps(self_test(),indent=2))
        elif args.build:
            print(json.dumps(build(),indent=2))
        else:
            print(json.dumps(validate(allow_pending_visual=args.allow_pending_visual),indent=2))
    except Exception as exc:
        print(f"ERROR: {exc}",file=sys.stderr)
        raise SystemExit(1)

if __name__=="__main__": main()
