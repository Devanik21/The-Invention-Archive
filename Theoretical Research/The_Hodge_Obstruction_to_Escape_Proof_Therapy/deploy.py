#!/usr/bin/env python3
"""Fail-closed build and release validator for the Hodge safety-game preprint.

Run from a clean Python environment with latexmk/TeX Live and the documented
NumPy, Matplotlib, and NetworkX dependencies installed:
  python deploy.py --build
  python deploy.py --validate-only
  python deploy.py --self-test
The self-test uses isolated temporary fixtures and never writes into the archive.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, re, shutil, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORE = {
    "paper.tex", "paper.pdf", "verify_model.py", "README.md", "CITATION.cff",
    "paper.bib", ".zenodo.json", "schema_article.jsonld", "deploy.py"
}

class GateError(RuntimeError):
    pass

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def read_inventory(root: Path):
    p = root / "results" / "figure_inventory.csv"
    if not p.is_file():
        raise GateError("Missing results/figure_inventory.csv")
    with p.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 12:
        raise GateError(f"Expected 12 distinct numbered scientific figures, found {len(rows)}")
    kinds = [r.get("kind", r.get("type", "")).strip().lower() for r in rows]
    # Current generator emits a 'kind' field; accept 'type' only for compatibility.
    counts = {k: kinds.count(k) for k in set(kinds)}
    if counts.get("plot", 0) < 6 or counts.get("graph", 0) < 2 or counts.get("flow", 0) < 2:
        raise GateError(f"Figure category floor failed: {counts}")
    pdfs = list((root / "figures").glob("fig*.pdf"))
    pngs = list((root / "figures").glob("fig*.png"))
    if len(pdfs) < 12 or len(pngs) < 12:
        raise GateError(f"Expected at least 12 vector PDFs and 12 PNG previews; found {len(pdfs)} and {len(pngs)}")
    return rows, counts

def pdf_facts(root: Path):
    try:
        import fitz
    except ImportError as e:
        raise GateError("PyMuPDF is required for PDF validation") from e
    pdf = root / "paper.pdf"
    if not pdf.is_file() or pdf.stat().st_size < 1000:
        raise GateError("paper.pdf missing or implausibly small")
    try:
        doc = fitz.open(pdf)
    except Exception as e:
        raise GateError(f"PDF parser rejected paper.pdf: {e}") from e
    if not doc.is_pdf or doc.page_count < 25:
        raise GateError(f"PDF validity/page floor failed: pages={doc.page_count}")
    texts = [page.get_text("text") for page in doc]
    start = next((i for i, t in enumerate(texts) if ("Introduction: from drug sequences to safety" in t or "Introduction: from drug sequences to safety games" in t) and "Antimicrobial resistance is an evolutionary control problem" in t), None)
    # The author's section title is included by the actual source; exact phrase match is intentional.
    end = next((i for i, t in enumerate(texts) if "Research-process credit and disclosure" in t), None)
    if start is None or end is None or end < start:
        raise GateError("Cannot establish main-text boundaries from rendered PDF text")
    main_pages = end - start + 1
    if main_pages < 18:
        raise GateError(f"Substantive main-text floor failed: {main_pages} pages (need at least 18)")
    alltext = "\n".join(texts)
    # Structural source checks plus rendered text captions; do not treat image count as quality evidence.
    tex = (root / "paper.tex").read_text(encoding="utf-8")
    table_source_count = len(re.findall(r"\\begin\{table\*?\}", tex))
    if table_source_count < 5:
        raise GateError(f"Fewer than five distinct source tables: {table_source_count}")
    fig_caps = len(re.findall(r"\\caption\{", tex))
    if fig_caps < 17: # 12 numbered figures plus at least five tables; source citations checked separately.
        raise GateError(f"Unexpectedly small caption inventory: {fig_caps}")
    # Flag common unresolved markers in extracted PDF text.
    for bad in ("??", "[?]", "undefined citation", "undefined reference"):
        if bad.lower() in alltext.lower():
            raise GateError(f"Unresolved marker found in extracted PDF: {bad}")
    return {"pages": doc.page_count, "main_text_pages": main_pages, "pdf_sha256": sha256(pdf), "table_source_count": table_source_count, "caption_count": fig_caps}

def validate(root: Path, allow_pending_visual=False):
    if not root.is_dir():
        raise GateError("Release directory is absent")
    actual = {p.name for p in root.iterdir() if p.is_file()}
    if actual != CORE:
        raise GateError(f"Exact 9-file root manifest failed; missing={sorted(CORE-actual)} extra={sorted(actual-CORE)}")
    for dirname in ("figures", "results"):
        if not (root / dirname).is_dir():
            raise GateError(f"Missing required {dirname}/ directory")
    for name in CORE:
        if (root / name).stat().st_size == 0:
            raise GateError(f"Empty core file: {name}")
    try:
        json.loads((root / ".zenodo.json").read_text(encoding="utf-8"))
        schema = json.loads((root / "schema_article.jsonld").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        raise GateError(f"Metadata JSON failed to parse: {e}") from e
    if schema.get("@type") != "ScholarlyArticle":
        raise GateError("JSON-LD @type must be ScholarlyArticle")
    cff = (root / "CITATION.cff").read_text(encoding="utf-8")
    for req in ("cff-version:", "title:", "authors:", "version:", "date-released:"):
        if req not in cff:
            raise GateError(f"CITATION.cff missing {req}")
    bib = (root / "paper.bib").read_text(encoding="utf-8")
    tex = (root / "paper.tex").read_text(encoding="utf-8")
    keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bib))
    cites = set(k for grp in re.findall(r"\\cite[tp]?\{([^}]+)\}", tex) for k in grp.split(","))
    missing = sorted(cites-keys)
    if missing:
        raise GateError(f"BibTeX citation keys missing: {missing}")
    readme = (root / "README.md").read_text(encoding="utf-8")
    if readme.count("$$") % 2:
        raise GateError("Unbalanced GitHub display-math delimiters in README")
    if r"\operatorname" in readme or r"\operatorname*" in readme:
        raise GateError("README contains a prohibited operatorname macro")
    if not re.search(r"Co-equal research-process credit is given to Devanik Debnath and Xylia", tex):
        raise GateError("Missing Xylia research-process credit/disclosure")
    rows, category_counts = read_inventory(root)
    facts = pdf_facts(root)
    verification = json.loads((root / "results" / "verification_report.json").read_text(encoding="utf-8"))
    if verification.get("all_passed") is not True:
        raise GateError("Numerical verification report is not all_passed=true")
    review_path = root / "results" / "pdf_visual_review.json"
    if not review_path.is_file():
        if not allow_pending_visual:
            raise GateError("Visual review record missing")
        review = {"status": "BLOCKED"}
    else:
        review = json.loads(review_path.read_text(encoding="utf-8"))
    review_ok = (review.get("status") == "PASS" and
                 review.get("paper_pdf_sha256") == facts["pdf_sha256"] and
                 len(set(review.get("pages_individually_reviewed", []))) == facts["pages"] and
                 not any(i.get("severity") in ("critical", "major") and i.get("final_status") != "resolved" for i in review.get("issues", [])))
    if not review_ok and not allow_pending_visual:
        raise GateError("Publication-quality gate blocked: full page review/hash/issue criteria do not pass")
    return {**facts, "figure_count": len(rows), "figure_category_counts": category_counts,
            "visual_review": "PASS" if review_ok else "BLOCKED", "verification_assertions": len(verification.get("checks", []))}

def build(root: Path):
    # Verify current code before building; nonzero exits block release.
    run = subprocess.run([sys.executable, str(root / "verify_model.py")], cwd=root, text=True, capture_output=True)
    (root / "results" / "verify_model_stdout.txt").write_text(run.stdout + run.stderr, encoding="utf-8")
    if run.returncode != 0:
        raise GateError(f"verify_model.py failed (exit {run.returncode}); see results/verify_model_stdout.txt")
    latexmk = shutil.which("latexmk")
    if not latexmk:
        raise GateError("latexmk not found; refusing to use stale paper.pdf")
    with tempfile.TemporaryDirectory(prefix="hodge-build-") as td:
        tmp = Path(td)
        shutil.copy2(root / "paper.tex", tmp / "paper.tex")
        shutil.copy2(root / "paper.bib", tmp / "paper.bib")
        shutil.copytree(root / "figures", tmp / "figures")
        # Some minimal TeX environments ship bibtex.original but leave the
        # /usr/bin/bibtex alternatives symlink dangling. Repair this only in the
        # temporary build PATH; never modify system alternatives.
        binfix = tmp / "build-bin"
        binfix.mkdir()
        bibexe = shutil.which("bibtex") or shutil.which("bibtex.original")
        build_env = os.environ.copy()
        if bibexe:
            link = binfix / "bibtex"
            link.symlink_to(bibexe)
            build_env["PATH"] = str(binfix) + os.pathsep + build_env.get("PATH", "")
        proc = subprocess.run([latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", "paper.tex"], cwd=tmp, env=build_env, text=True, capture_output=True)
        (root / "results" / "latex_build_stdout.txt").write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr, encoding="utf-8")
        logpath = tmp / "paper.log"
        log = logpath.read_text(encoding="utf-8", errors="replace") if logpath.exists() else ""
        unresolved = re.findall(r"(?:Citation|Reference) `[^']+' on page .* undefined|There were undefined references|Citation .* undefined", log, re.I)
        if proc.returncode != 0 or not (tmp / "paper.pdf").is_file() or unresolved:
            raise GateError(f"Clean LaTeX build failed or has unresolved refs (exit={proc.returncode}, unresolved={unresolved[:5]}); refusing stale PDF")
        shutil.copy2(tmp / "paper.pdf", root / "paper.pdf")
        (root / "results" / "latex_build_log.txt").write_text(log, encoding="utf-8")
    return validate(root, allow_pending_visual=True)

def self_test():
    # A deliberately incomplete fixture must be rejected and must not affect the real root.
    with tempfile.TemporaryDirectory(prefix="hodge-deploy-selftest-") as td:
        fixture = Path(td) / "release"
        fixture.mkdir()
        for dirname in ("figures", "results"):
            (fixture / dirname).mkdir()
        for name in CORE - {"paper.pdf"}:
            (fixture / name).write_text("fixture", encoding="utf-8")
        try:
            validate(fixture, allow_pending_visual=True)
        except GateError as e:
            print(f"[PASS] malformed/missing-artifact fixture was blocked: {e}")
        else:
            raise GateError("Self-test failed: malformed fixture unexpectedly passed")
        (fixture / "paper.pdf").write_bytes(b"%PDF-bogus")
        try:
            pdf_facts(fixture)
        except GateError as e:
            print(f"[PASS] invalid-PDF fixture was blocked: {e}")
        else:
            raise GateError("Self-test failed: invalid PDF fixture unexpectedly passed")
    print("Deployment self-test: PASS (two malformed release fixtures rejected)")

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--build", action="store_true", help="run verification, clean LaTeX build, and non-visual preflight")
    g.add_argument("--validate-only", action="store_true", help="validate release artifacts without building")
    g.add_argument("--self-test", action="store_true", help="test fail-closed behaviour using isolated malformed fixtures")
    ap.add_argument("--allow-pending-visual", action="store_true", help="allow diagnostics before manual visual review; never represents publication-quality PASS")
    args = ap.parse_args()
    try:
        if args.self_test:
            self_test()
        elif args.build:
            result = build(ROOT)
            print(json.dumps(result, indent=2))
            if result["visual_review"] != "PASS":
                print("Publication-quality status remains BLOCKED until visual review record matches this PDF hash.")
        else:
            result = validate(ROOT, allow_pending_visual=args.allow_pending_visual)
            print(json.dumps(result, indent=2))
        return 0
    except GateError as e:
        print(f"BLOCKED: {e}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
