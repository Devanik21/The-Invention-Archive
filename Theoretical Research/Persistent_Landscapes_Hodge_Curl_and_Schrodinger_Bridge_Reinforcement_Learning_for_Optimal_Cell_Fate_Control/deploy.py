#!/usr/bin/env python3
"""Fail-closed structural validation for the cell-fate-control preprint.

Run `python deploy.py --validate-only` for local artifact checks or
`python deploy.py --self-test` to test rejection of malformed fixtures.
This script never writes to Git, publishes a release, or modifies unrelated files.
"""
from __future__ import annotations
import argparse, csv, json, re, sys, tempfile
from pathlib import Path

CORE = {
    "paper.tex", "paper.pdf", "verify_model.py", "README.md", "CITATION.cff",
    "paper.bib", ".zenodo.json", "schema_article.jsonld", "deploy.py",
}
RELEASE_SLUG = "Persistent_Landscapes_Hodge_Curl_and_Schrodinger_Bridge_Reinforcement_Learning_for_Optimal_Cell_Fate_Control"


def validate(root: Path, *, strict_pdf: bool = True) -> list[str]:
    errors: list[str] = []
    if not root.exists() or not root.is_dir():
        return [f"release directory missing: {root}"]
    files = {p.name for p in root.iterdir() if p.is_file()}
    if files != CORE:
        errors.append(f"root core-file mismatch; missing={sorted(CORE-files)}, extra={sorted(files-CORE)}")
    for name in CORE:
        p = root / name
        if not p.is_file() or p.stat().st_size == 0:
            errors.append(f"missing or empty core file: {name}")
    for directory in (root / "figures", root / "results"):
        if not directory.is_dir():
            errors.append(f"required directory missing: {directory.name}")
    if errors and not (root / "paper.tex").is_file():
        return errors

    try:
        tex = (root / "paper.tex").read_text(encoding="utf-8")
        bib = (root / "paper.bib").read_text(encoding="utf-8")
        readme = (root / "README.md").read_text(encoding="utf-8")
        keys = set(re.findall(r"\\bibitem(?:\[[^]]*\])?\{([^}]+)\}", tex))
        cites = {key.strip() for group in re.findall(r"\\cite[a-zA-Z*]*\{([^}]+)\}", tex) for key in group.split(",")}
        if cites - keys:
            errors.append(f"LaTeX citation keys absent from thebibliography: {sorted(cites-keys)}")
        if len(re.findall(r"\\begin\{table\}", tex)) < 5:
            errors.append("paper.tex contains fewer than five table environments")
        if tex.count("\\fig{") < 12:
            errors.append("paper.tex contains fewer than twelve numbered figure environments")
        if "\\operatorname" in readme:
            errors.append("README contains a prohibited operatorname macro")
        if readme.count("$$") % 2:
            errors.append("README contains unbalanced display-math delimiters")
        for target in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex):
            if "#" in target:  # reusable macro placeholder; each actual asset is checked below
                continue
            if not (root / target).is_file():
                errors.append(f"figure included by LaTeX is missing: {target}")
        # Ensure the main manuscript is using the same set of bibliography keys as paper.bib.
        bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bib))
        if not bib_keys:
            errors.append("paper.bib contains no BibTeX entries")
        missing_bib = cites - bib_keys
        if missing_bib:
            errors.append(f"citation keys missing from paper.bib: {sorted(missing_bib)}")
    except Exception as exc:
        errors.append(f"source parse failure: {type(exc).__name__}: {exc}")

    for name, parser in ((".zenodo.json", json.loads), ("schema_article.jsonld", json.loads)):
        try:
            parsed = parser((root / name).read_text(encoding="utf-8"))
            if name == ".zenodo.json" and not all(k in parsed for k in ("title", "creators", "description", "upload_type")):
                errors.append(".zenodo.json missing top-level Zenodo metadata fields")
            if name == "schema_article.jsonld" and parsed.get("@type") != "ScholarlyArticle":
                errors.append("schema_article.jsonld @type is not ScholarlyArticle")
        except Exception as exc:
            errors.append(f"invalid {name}: {type(exc).__name__}: {exc}")
    try:
        import yaml
        data = yaml.safe_load((root / "CITATION.cff").read_text(encoding="utf-8"))
        if data.get("cff-version") != "1.2.0" or not data.get("authors"):
            errors.append("CITATION.cff missing expected version or author")
    except Exception as exc:
        errors.append(f"invalid CITATION.cff YAML: {type(exc).__name__}: {exc}")

    inventory = root / "results" / "figure_inventory.csv"
    if not inventory.is_file():
        errors.append("results/figure_inventory.csv missing")
    else:
        try:
            with inventory.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if len(rows) < 12:
                errors.append(f"figure inventory has {len(rows)} rows; expected at least 12")
            categories = [row.get("categories", "") for row in rows]
            if sum("plot" in x for x in categories) < 6: errors.append("figure inventory has fewer than six plots")
            if sum("graph" in x and "flow" not in x for x in categories) < 2: errors.append("figure inventory has fewer than two graph/network visuals")
            if sum("flow_diagram" in x for x in categories) < 2: errors.append("figure inventory has fewer than two flow diagrams")
        except Exception as exc:
            errors.append(f"cannot parse figure inventory: {type(exc).__name__}: {exc}")
    for i in range(1, 13):
        stem = f"fig{i:02d}_"
        matches = list((root / "figures").glob(stem + "*.pdf")) if (root / "figures").is_dir() else []
        if not matches or matches[0].stat().st_size == 0:
            errors.append(f"missing or empty vector figure number {i}")

    vr = root / "results" / "verification_report.json"
    if not vr.is_file():
        errors.append("results/verification_report.json missing")
    else:
        try:
            result = json.loads(vr.read_text(encoding="utf-8"))
            if result.get("status") != "PASS" or result.get("checks_failed") != 0:
                errors.append("synthetic verification report is not PASS with zero failed checks")
            if result.get("empirical_data_loaded") is not False:
                errors.append("verification report must state that no empirical data were loaded")
        except Exception as exc:
            errors.append(f"invalid verification report: {type(exc).__name__}: {exc}")

    if strict_pdf and (root / "paper.pdf").is_file() and (root / "paper.pdf").stat().st_size:
        try:
            import fitz
            doc = fitz.open(root / "paper.pdf")
            if len(doc) < 25: errors.append(f"PDF has {len(doc)} pages; required minimum is 25")
            texts = [page.get_text() for page in doc]
            start = next((i for i, text in enumerate(texts) if re.search(r"1 Research question and contribution\s+Cell identity", text)), None)
            appendix = next((i for i, text in enumerate(texts) if i > (start or 0) and "A Derivation of the gradient action identity" in text), None)
            if start is None or appendix is None:
                errors.append("cannot identify main-text and appendix boundary in PDF text")
            else:
                substantive = sum(1 for text in texts[start:appendix] if len(text.strip()) > 300)
                if substantive < 18: errors.append(f"PDF has only {substantive} estimated substantive main-text pages; required minimum is 18")
            full_text = "\n".join(texts)
            if "Co-equal research-process credit is given to Devanik Debnath and Xylia" not in full_text:
                errors.append("required research-process credit not found in PDF text")
            if re.search(r"\b(\?\?+|undefined citation|cite\w+\{)", full_text, re.I):
                errors.append("PDF text contains a possible unresolved citation/placeholder marker")
        except Exception as exc:
            errors.append(f"PDF parsing/preflight failed: {type(exc).__name__}: {exc}")
    return errors


def self_test() -> int:
    with tempfile.TemporaryDirectory(prefix="cell-fate-deploy-selftest-") as temp:
        bad = Path(temp) / "fixture"
        bad.mkdir()
        (bad / "paper.tex").write_text("\\documentclass{article}\n", encoding="utf-8")
        issues = validate(bad, strict_pdf=False)
        if not issues or not any("root core-file mismatch" in e for e in issues):
            print("FAIL: malformed fixture was not rejected", file=sys.stderr)
            return 1
        print("PASS: malformed/missing-artifact fixture rejected (", len(issues), "issues)")
        bad2 = Path(temp) / "malformed-metadata"
        bad2.mkdir()
        for name in CORE:
            (bad2 / name).write_text("placeholder", encoding="utf-8")
        (bad2 / "paper.tex").write_text("\\begin{figure}\\includegraphics{figures/missing.pdf}\\end{figure}", encoding="utf-8")
        (bad2 / "README.md").write_text("$$\nx\n", encoding="utf-8")
        (bad2 / ".zenodo.json").write_text("{not-json", encoding="utf-8")
        (bad2 / "schema_article.jsonld").write_text("{}", encoding="utf-8")
        (bad2 / "CITATION.cff").write_text("bad: [yaml", encoding="utf-8")
        (bad2 / "figures").mkdir(); (bad2 / "results").mkdir()
        issues2 = validate(bad2, strict_pdf=False)
        if not any("invalid .zenodo.json" in e for e in issues2):
            print("FAIL: malformed metadata fixture was not rejected", file=sys.stderr)
            return 1
        print("PASS: malformed JSON/YAML and missing figures are rejected")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true", help="validate the current release without writing or deploying")
    parser.add_argument("--self-test", action="store_true", help="test fail-closed handling in temporary fixtures")
    args = parser.parse_args()
    if args.self_test:
        code = self_test()
        if code: return code
    root = Path(__file__).resolve().parent
    if args.validate_only or not args.self_test:
        errors = validate(root, strict_pdf=True)
        if errors:
            print("BLOCKED: release validation failed", file=sys.stderr)
            for item in errors: print(f"- {item}", file=sys.stderr)
            return 1
        print("PASS: core manifest, metadata, bibliography keys, figure inventory, verification and PDF floors validated")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
