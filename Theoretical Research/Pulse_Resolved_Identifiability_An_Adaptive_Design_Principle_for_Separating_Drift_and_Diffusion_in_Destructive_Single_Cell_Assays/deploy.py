#!/usr/bin/env python3
"""Fail-closed local validation and deployment helper for this release."""
from __future__ import annotations
import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CORE = {
    "paper.tex", "paper.pdf", "verify_model.py", "README.md",
    "CITATION.cff", "paper.bib", ".zenodo.json",
    "schema_article.jsonld", "deploy.py"
}
EXPECTED_FIGURES = [
    "fig1_mean_response", "fig2_variance_confounding",
    "fig3_information_schedule", "fig4_parameter_rmse",
    "fig5_model_dependency_graph", "fig6_information_coupling_graph",
    "fig7_assay_pipeline", "fig8_adaptive_loop"
]


def run(command, cwd=None):
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(map(str, command))}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def validate_release(root: Path):
    errors = []
    root = root.resolve()
    if not root.is_dir():
        return [f"release directory does not exist: {root}"]
    actual = {p.name for p in root.iterdir() if p.is_file()}
    if actual != CORE:
        errors.append(f"root-level core manifest mismatch: missing={sorted(CORE-actual)}; extra={sorted(actual-CORE)}")
    for folder in ("figures", "results"):
        if not (root / folder).is_dir():
            errors.append(f"required directory missing: {folder}")
    if not (root / "figures").is_dir() or not (root / "results").is_dir():
        return errors
    for name in CORE:
        p = root / name
        if not p.is_file() or p.stat().st_size == 0:
            errors.append(f"missing or empty core file: {name}")
    tex_path = root / "paper.tex"
    bib_path = root / "paper.bib"
    readme_path = root / "README.md"
    pdf_path = root / "paper.pdf"
    if tex_path.is_file():
        tex = tex_path.read_text(encoding="utf-8")
        static_tables = tex.count(r"\begin{table}")
        figure_envs = tex.count(r"\begin{figure}")
        if static_tables + (1 if (root / "results/generated_results.tex").is_file() else 0) < 5:
            errors.append("fewer than five substantive table environments are present")
        if figure_envs < 8:
            errors.append(f"fewer than eight required figure environments: {figure_envs}")
        if "Co-equal research-process credit is given to Devanik Debnath and Xylia" not in tex:
            errors.append("required Xylia research-process credit is missing")
        if r"\input{results/generated_results.tex}" in tex and not (root / "results/generated_results.tex").is_file():
            errors.append("generated results table is missing")
    if readme_path.is_file():
        readme = readme_path.read_text(encoding="utf-8")
        if readme.count("$$") % 2:
            errors.append("README display-math delimiters are unbalanced")
        if r"\operatorname" in readme:
            errors.append("README contains prohibited operatorname macro")
        if r"\n" in readme:
            errors.append("README contains a literal escaped newline sequence")
        for rel in ["paper.pdf", "paper.tex", "verify_model.py", "paper.bib", "CITATION.cff", ".zenodo.json", "schema_article.jsonld", "deploy.py"]:
            if rel not in readme:
                errors.append(f"README does not link or refer to {rel}")
    if bib_path.is_file() and tex_path.is_file():
        bib = bib_path.read_text(encoding="utf-8")
        tex = tex_path.read_text(encoding="utf-8")
        keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bib))
        cited = set()
        for group in re.findall(r"\\cite\w*\{([^}]+)\}", tex):
            cited.update(k.strip() for k in group.split(","))
        missing = sorted(cited - keys)
        if missing:
            errors.append(f"citation keys missing from bibliography: {missing}")
    for name in EXPECTED_FIGURES:
        for ext in (".pdf", ".svg"):
            p = root / "figures" / (name + ext)
            if not p.is_file() or p.stat().st_size < 1000:
                errors.append(f"missing or undersized figure asset: figures/{name}{ext}")
    results = root / "results"
    required_results = [
        "research_log.md", "author_profile_audit.md", "topic_candidates.csv",
        "literature_audit.md", "claim_evidence_matrix.csv", "publication_design_brief.md",
        "verification_report.json", "verification_summary.txt", "environment_report.json",
        "simulation_config.json", "figure_data_manifest.json", "generated_results.tex",
        "pdf_visual_review.json", "publication_quality_report.md", "release_check_matrix.json",
        "sha256_manifest.json"
    ]
    for name in required_results:
        p = results / name
        if not p.is_file() or p.stat().st_size == 0:
            errors.append(f"missing or empty research record: results/{name}")
    for name in [".zenodo.json", "schema_article.jsonld"]:
        p = root / name
        if p.is_file():
            try:
                json.loads(p.read_text(encoding="utf-8"))
            except Exception as exc:
                errors.append(f"invalid JSON metadata {name}: {exc}")
    report_path = results / "verification_report.json"
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("status") != "PASS":
                errors.append("verification_report.json does not report PASS")
        except Exception as exc:
            errors.append(f"invalid verification report: {exc}")
    if pdf_path.is_file() and pdf_path.stat().st_size:
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(pdf_path))
            if len(reader.pages) < 7:
                errors.append(f"PDF has fewer than seven pages: {len(reader.pages)}")
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            if "References" not in text:
                errors.append("PDF text extraction did not find references")
            if "Co-equal research-process credit is given" not in text or "Xylia" not in text:
                errors.append("compiled PDF is missing the Xylia credit")
            if not text.strip():
                errors.append("PDF text extraction is empty")
        except Exception as exc:
            errors.append(f"PDF parsing failed: {exc}")
    return errors


def self_test():
    with tempfile.TemporaryDirectory(prefix="cell-state-deploy-selftest-") as temp:
        root = Path(temp) / "fixture"
        root.mkdir()
        (root / "figures").mkdir()
        (root / "results").mkdir()
        (root / "paper.tex").write_text("incomplete fixture", encoding="utf-8")
        missing_errors = validate_release(root)
        if not any("manifest mismatch" in error for error in missing_errors):
            raise AssertionError("missing-artifact fixture was not blocked")
        (root / "unexpected.txt").write_text("must be rejected", encoding="utf-8")
        extra_errors = validate_release(root)
        if not any("manifest mismatch" in error and "unexpected.txt" in error for error in extra_errors):
            raise AssertionError("extra-root-file fixture was not blocked")
    print("PASS: malformed/missing-artifact fixture is blocked")
    print("PASS: unexpected root-level artifact fixture is blocked")
    print("Deployment self-tests passed; the real archive was not modified.")


def build(root: Path):
    run([sys.executable, str(root / "verify_model.py")], cwd=root)
    for tool in ("pdflatex", "bibtex"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"required LaTeX toolchain is unavailable: {tool}")
    with tempfile.TemporaryDirectory(prefix="cell-state-pdf-build-") as temp:
        work = Path(temp)
        shutil.copy2(root / "paper.tex", work / "paper.tex")
        shutil.copy2(root / "paper.bib", work / "paper.bib")
        shutil.copytree(root / "figures", work / "figures")
        (work / "results").mkdir()
        shutil.copy2(root / "results/generated_results.tex", work / "results/generated_results.tex")
        run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "paper.tex"], cwd=work)
        run(["bibtex", "paper"], cwd=work)
        run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "paper.tex"], cwd=work)
        run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "paper.tex"], cwd=work)
        log = (work / "paper.log").read_text(encoding="utf-8", errors="replace")
        if "Citation" in log and "undefined" in log:
            raise RuntimeError("undefined citation remains in LaTeX log")
        if "Reference" in log and "undefined" in log:
            raise RuntimeError("undefined reference remains in LaTeX log")
        if not (work / "paper.pdf").is_file() or (work / "paper.pdf").stat().st_size == 0:
            raise RuntimeError("LaTeX build did not produce a nonempty PDF")
        shutil.copy2(work / "paper.pdf", root / "paper.pdf")
    print("PDF rebuilt from current source and bibliography.")


def publish(root: Path):
    errors = validate_release(root)
    if errors:
        raise RuntimeError("release validation failed:\n- " + "\n- ".join(errors))
    matrix = json.loads((root / "results/release_check_matrix.json").read_text(encoding="utf-8"))
    if matrix.get("release_status") != "PASS":
        raise RuntimeError("refusing publication: release_check_matrix.json is not PASS")
    review = json.loads((root / "results/pdf_visual_review.json").read_text(encoding="utf-8"))
    if review.get("status") != "PASS":
        raise RuntimeError("refusing publication: PDF visual review is not PASS")
    repo_root = Path(run(["git", "rev-parse", "--show-toplevel"], cwd=root).strip())
    branch = run(["git", "branch", "--show-current"], cwd=repo_root).strip()
    if branch != "main":
        raise RuntimeError(f"refusing publication from branch {branch!r}; checkout main explicitly")
    remote = run(["git", "remote", "get-url", "origin"], cwd=repo_root).strip()
    if remote.rstrip("/").removesuffix(".git") != "https://github.com/Devanik21/The-Invention-Archive":
        raise RuntimeError(f"unexpected origin remote: {remote}")
    tag = "v2026.10.24"
    existing = subprocess.run(["git", "show-ref", "--verify", "--quiet", f"refs/tags/{tag}"], cwd=repo_root)
    if existing.returncode == 0:
        raise RuntimeError(f"tag {tag} already exists; refusing to overwrite it")
    run(["git", "add", str(root.relative_to(repo_root)), "Theoretical Research/Readme.md"], cwd=repo_root)
    staged = run(["git", "diff", "--cached", "--name-only"], cwd=repo_root).splitlines()
    allowed_prefix = str(root.relative_to(repo_root))
    if any(not (p == "Theoretical Research/Readme.md" or p.startswith(allowed_prefix + "/")) for p in staged):
        raise RuntimeError("staged changes include files outside this release and its index")
    if staged:
        run(["git", "commit", "-m", "release(research): pulse-resolved identifiability preprint"], cwd=repo_root)
    run(["git", "push", "origin", "HEAD:main"], cwd=repo_root)
    run(["git", "tag", "-a", tag, "-m", "Pulse-Resolved Identifiability preprint " + tag], cwd=repo_root)
    run(["git", "push", "origin", tag], cwd=repo_root)
    print(f"Published by fast-forward push and tagged {tag}. No force-push was used.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.self_test:
        self_test()
    if args.build:
        build(root)
    if args.validate:
        errors = validate_release(root)
        if errors:
            print("VALIDATION FAILED")
            for error in errors:
                print(" - " + error)
            raise SystemExit(1)
        print("VALIDATION PASS: exact core manifest, evidence directories, metadata, PDF, and research records")
    if args.publish:
        publish(root)
    if not any((args.self_test, args.validate, args.build, args.publish)):
        parser.print_help()


if __name__ == "__main__":
    main()
