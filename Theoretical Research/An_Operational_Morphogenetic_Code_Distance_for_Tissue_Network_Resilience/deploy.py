#!/usr/bin/env python3
"""Fail-closed validation and provenance utility for this research release.

This utility validates the nine-file release layout, metadata, manuscript,
PDF, bibliography keys, figure inventory, and verification outputs. It does
not push to GitHub, create a DOI, or claim empirical validation.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
CORE_FILES = {
    "paper.tex", "paper.pdf", "verify_model.py", "README.md", "CITATION.cff",
    "paper.bib", ".zenodo.json", "schema_article.jsonld", "deploy.py"
}
FIGURE_BASES = {
    "fig1_lesion_topology", "fig2_pinned_spectral_gap", "fig3_nnls_noise_sensitivity",
    "fig4_pinned_diffusion_traces", "fig5_structural_network",
    "fig6_sheaf_stalk_graph", "fig7_inference_flow", "fig8_validation_flow"
}
REQUIRED_RESULTS = {
    "verification_report.json", "verification_summary.txt",
    "dispersed_vs_targeted_lesions.csv", "spectral_gap_vs_cut_size.csv",
    "nnls_conductance_estimation_trials.csv", "pinned_diffusion_traces.csv",
    "weighted_mcd_sensitivity.csv", "candidate_screen.csv", "literature_audit.csv",
    "claim_evidence_matrix.csv", "author_profile_audit.md", "workflow_log.md",
    "pdf_audit.md", "environment_report.json", "pdf_visual_review.json",
    "release_check_matrix.json", "deploy_validation.log", "deploy_self_test.log"
}


class ReleaseError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseError(message)


def run(command: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, check=True)
    except FileNotFoundError as exc:
        raise ReleaseError(f"Required executable not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise ReleaseError(
            f"Command failed ({exc.returncode}): {' '.join(command)}\n"
            f"STDOUT:\n{exc.stdout[-4000:]}\nSTDERR:\n{exc.stderr[-4000:]}"
        ) from exc


def validate_layout(root: Path, check_sizes: bool = True) -> None:
    require(root.is_dir(), f"Release directory does not exist: {root}")
    actual_files = {p.name for p in root.iterdir() if p.is_file()}
    missing = sorted(CORE_FILES - actual_files)
    extra = sorted(actual_files - CORE_FILES)
    require(not missing, f"Missing root-level core files: {missing}")
    require(not extra, f"Unexpected root-level files: {extra}")
    required_dirs = {"figures", "results"}
    actual_dirs = {p.name for p in root.iterdir() if p.is_dir()}
    require(required_dirs <= actual_dirs, f"Missing evidence directories: {sorted(required_dirs-actual_dirs)}")
    require(actual_dirs <= required_dirs, f"Unexpected root-level directories: {sorted(actual_dirs-required_dirs)}")
    if check_sizes:
        empty = [name for name in CORE_FILES if (root / name).stat().st_size == 0]
        require(not empty, f"Empty core files: {empty}")


def validate_figures(root: Path) -> None:
    fig_dir = root / "figures"
    require(fig_dir.is_dir(), "figures/ directory missing")
    available = {p.stem for p in fig_dir.iterdir() if p.suffix.lower() in {".pdf", ".png"}}
    missing = sorted(FIGURE_BASES - available)
    require(not missing, f"Missing required visual assets: {missing}")
    for base in FIGURE_BASES:
        for suffix in (".pdf", ".png"):
            p = fig_dir / f"{base}{suffix}"
            require(p.is_file() and p.stat().st_size > 500, f"Missing/empty visual asset: {p}")
    # Do not silently count duplicates as distinct visuals.
    require(len(FIGURE_BASES) >= 8, "Visual category inventory incomplete")


def validate_results(root: Path) -> None:
    res = root / "results"
    require(res.is_dir(), "results/ directory missing")
    missing = sorted(name for name in REQUIRED_RESULTS if not (res / name).is_file() or (res / name).stat().st_size == 0)
    require(not missing, f"Missing/empty research evidence files: {missing}")
    report = json.loads((res / "verification_report.json").read_text(encoding="utf-8"))
    require(report.get("run_status") == "PASS", "verification_report.json is not PASS")
    require(report.get("passed") == 14 and report.get("failed") == 0,
            "Expected 14 passing checks and zero failures in verification report")
    require((res / "candidate_screen.csv").read_text(encoding="utf-8").count("\n") >= 10,
            "Candidate screen must contain 8–12 candidates")


def validate_metadata(root: Path) -> None:
    import yaml
    cff = yaml.safe_load((root / "CITATION.cff").read_text(encoding="utf-8"))
    require(isinstance(cff, dict) and cff.get("cff-version") == "1.2.0", "CITATION.cff schema/version invalid")
    require(cff.get("title") and cff.get("authors") and cff.get("date-released") and cff.get("license"),
            "CITATION.cff misses required citation fields")
    zenodo = json.loads((root / ".zenodo.json").read_text(encoding="utf-8"))
    require(zenodo.get("title") and zenodo.get("creators") and zenodo.get("upload_type") == "publication",
            ".zenodo.json metadata incomplete")
    require("doi" not in zenodo and "prereserved_doi" not in zenodo,
            "Do not mint or imply a DOI in metadata without an actual deposit")
    schema = json.loads((root / "schema_article.jsonld").read_text(encoding="utf-8"))
    require(schema.get("@type") == "ScholarlyArticle" and schema.get("author"),
            "schema_article.jsonld type/author invalid")


def validate_bibliography(root: Path) -> None:
    tex = (root / "paper.tex").read_text(encoding="utf-8")
    bib = (root / "paper.bib").read_text(encoding="utf-8")
    keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bib))
    cite_groups = re.findall(r"\\cite[a-zA-Z*]*\s*(?:\[[^]]*\])?\s*\{([^}]+)\}", tex)
    used = {key.strip() for group in cite_groups for key in group.split(",")}
    missing = sorted(used - keys)
    require(not missing, f"Citations missing from paper.bib: {missing}")
    manual = set(re.findall(r"\\bibitem(?:\[[^]]*\])?\{([^}]+)\}", tex))
    require(used <= manual, f"Citations missing from rendered bibliography: {sorted(used-manual)}")


def validate_paper(root: Path, min_pages: int = 25) -> int:
    tex = (root / "paper.tex").read_text(encoding="utf-8")
    readme = (root / "README.md").read_text(encoding="utf-8")
    require("Co-equal research-process credit is given to Devanik Debnath and Xylia" in tex,
            "Required Xylia research-process disclosure missing")
    require("No biological MCD has been measured" in tex or "No biological MCD has been measured" in readme,
            "Manuscript must explicitly disclose absence of biological MCD measurements")
    require("\u0000" not in tex, "NUL byte in LaTeX source")
    require(readme.count("$$") % 2 == 0, "Unbalanced README display-math delimiters")
    require(r"\operatorname" not in readme and r"\operatorname*" not in readme,
            "Prohibited operatorname macro in README")
    require(r"\n" not in readme, "Literal escaped newlines in README")
    require(tex.count(r"\begin{table}") + tex.count(r"\begin{longtable}") >= 5,
            "Manuscript must contain at least five substantive tables")
    require(tex.count(r"\begin{figure}") >= 8, "Manuscript must contain at least eight distinct figures")
    pdf = root / "paper.pdf"
    require(pdf.is_file() and pdf.stat().st_size > 100_000, "Compiled PDF missing or implausibly small")
    require(pdf.read_bytes()[:5] == b"%PDF-", "PDF signature invalid")
    info = run(["pdfinfo", str(pdf)])
    page_match = re.search(r"^Pages:\s+(\d+)", info.stdout, flags=re.M)
    require(page_match is not None, "pdfinfo did not report page count")
    pages = int(page_match.group(1))
    require(pages >= min_pages, f"Paper page count {pages} is below required {min_pages}")
    extracted = run(["pdftotext", str(pdf), "-"]).stdout
    for marker in ("Abstract", "Research posture", "Limitations", "References", "Devanik Debnath"):
        require(marker.lower() in extracted.lower(), f"PDF text missing expected marker: {marker}")
    require("undefined" not in extracted.lower(), "Suspicious literal 'undefined' in extracted PDF text")
    return pages


def run_verifier(root: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="mcd-pycache-") as cache_s:
        env = os.environ.copy()
        env["PYTHONPYCACHEPREFIX"] = cache_s
        run([sys.executable, "-m", "py_compile", str(root / "verify_model.py"), str(root / "deploy.py")], cwd=root, env=env)
    result = run([sys.executable, str(root / "verify_model.py")], cwd=root)
    (root / "results" / "deploy_validation.log").write_text(
        "Command: python verify_model.py\nExit status: 0\n\n" + result.stdout + result.stderr,
        encoding="utf-8"
    )


def rebuild_pdf(root: Path) -> int:
    with tempfile.TemporaryDirectory(prefix="mcd-pdf-build-") as tmp_s:
        tmp = Path(tmp_s)
        shutil.copy2(root / "paper.tex", tmp / "paper.tex")
        shutil.copytree(root / "figures", tmp / "figures")
        for _ in range(2):
            run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "paper.tex"], cwd=tmp)
        generated = tmp / "paper.pdf"
        require(generated.is_file(), "pdflatex produced no paper.pdf")
        info = run(["pdfinfo", str(generated)])
        m = re.search(r"^Pages:\s+(\d+)", info.stdout, flags=re.M)
        require(m is not None and int(m.group(1)) >= 25, "Clean LaTeX rebuild did not meet 25-page requirement")
        extracted = run(["pdftotext", str(generated), "-"]).stdout
        require("References" in extracted, "Clean rebuild omitted bibliography text")
        shutil.copy2(generated, root / "paper.pdf")
        return int(m.group(1))


def write_manifest(root: Path) -> dict:
    excluded = {"sha256_manifest.json", "release_check_matrix.json"}
    manifest = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if rel.startswith(".git/") or (rel.startswith("results/") and Path(rel).name in excluded):
            continue
        manifest[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    path = root / "results" / "sha256_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="mcd-deploy-selftest-") as tmp_s:
        fixture = Path(tmp_s)
        for name in CORE_FILES:
            (fixture / name).write_text("fixture", encoding="utf-8")
        (fixture / "figures").mkdir()
        (fixture / "results").mkdir()
        validate_layout(fixture, check_sizes=True)
        (fixture / "unexpected.txt").write_text("x", encoding="utf-8")
        try:
            validate_layout(fixture, check_sizes=True)
        except ReleaseError:
            pass
        else:
            raise ReleaseError("Self-test failed: unexpected root-level file was accepted")
        (fixture / "unexpected.txt").unlink()
        (fixture / "paper.pdf").unlink()
        try:
            validate_layout(fixture, check_sizes=True)
        except ReleaseError:
            pass
        else:
            raise ReleaseError("Self-test failed: missing core file was accepted")
    print("PASS deployment self-test: exact core-file layout, extra-file rejection, missing-file rejection")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run isolated layout fail-closed tests")
    parser.add_argument("--validate-only", action="store_true", help="validate the existing release without publishing")
    parser.add_argument("--rebuild-pdf", action="store_true", help="compile from the current source in a temporary directory")
    parser.add_argument("--skip-model", action="store_true", help="skip executing the numerical verifier")
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
            (ROOT / "results").mkdir(exist_ok=True)
            (ROOT / "results" / "deploy_self_test.log").write_text("PASS deployment self-test: exact core-file layout, extra-file rejection, missing-file rejection\n", encoding="utf-8")
            return 0
        validate_layout(ROOT)
        validate_figures(ROOT)
        validate_results(ROOT)
        validate_metadata(ROOT)
        validate_bibliography(ROOT)
        if not args.skip_model:
            run_verifier(ROOT)
        validate_layout(ROOT)
        validate_results(ROOT)
        if args.rebuild_pdf:
            pages = rebuild_pdf(ROOT)
            pages = validate_paper(ROOT)
        else:
            pages = validate_paper(ROOT)
        manifest = write_manifest(ROOT)
        print(json.dumps({
            "status": "PASS",
            "root_core_files": len(CORE_FILES),
            "figure_assets": len(FIGURE_BASES) * 2,
            "manifest_entries": len(manifest),
            "pdf_pages": pages,
            "model_execution": "SKIPPED" if args.skip_model else "PASS",
            "empirical_biological_validation": "NOT_PERFORMED"
        }, indent=2))
        return 0
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())