#!/usr/bin/env python3
"""Validate and safely deploy this release from a clean local Git clone.

Usage:
  python deploy.py --check-only   # run tests, metadata and PDF preflight only
  python deploy.py                # rebuild, stage only this release + index, commit and push

The script never force-pushes, never overwrites an existing release directory, and
refuses to deploy if the local main branch is not at the fetched origin/main head.
Credentials are supplied by the user's normal Git credential helper; no token is stored.
"""
from __future__ import annotations
import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TITLE = "The Identifiability Budget: An Audit Framework for Mechanistic AI and Experimental Design"
SLUG = "The_Identifiability_Budget_An_Audit_Framework_for_Mechanistic_AI_and_Experimental_Design"
VERSION = "v2026.10.11.2"
REPO_URL = "https://github.com/Devanik21/The-Invention-Archive"
CORE = ["paper.tex", "paper.pdf", "verify_model.py", "README.md", "medium_story.md",
        "x_thread.md", "CITATION.cff", "paper.bib", ".zenodo.json",
        "schema_article.jsonld", "deploy.py"]


def run(cmd: list[str], cwd: Path, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd, text=True,
                           stdout=subprocess.PIPE if capture else None,
                           stderr=subprocess.STDOUT if capture else None)
    if capture and result.stdout:
        print(result.stdout[-6000:])
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")
    return result


def git_root() -> Path:
    p = subprocess.run(["git", "rev-parse", "--show-toplevel"], text=True, capture_output=True)
    if p.returncode:
        raise RuntimeError("Run deploy.py from inside a local clone of The-Invention-Archive.")
    return Path(p.stdout.strip()).resolve()


def validate_metadata(release: Path) -> None:
    with (release / ".zenodo.json").open(encoding="utf-8") as f:
        z = json.load(f)
    with (release / "schema_article.jsonld").open(encoding="utf-8") as f:
        s = json.load(f)
    if z.get("title") != TITLE or s.get("headline") != TITLE:
        raise RuntimeError("Metadata title mismatch.")
    cff = (release / "CITATION.cff").read_text(encoding="utf-8")
    if "cff-version: 1.2.0" not in cff or "Devanik" not in cff:
        raise RuntimeError("CITATION.cff is missing required fields.")
    if "10.5281/zenodo." in cff or "10.5281/zenodo." in json.dumps(z):
        raise RuntimeError("A DOI is present although no Zenodo DOI has been issued by this script.")


def check_word_counts(release: Path) -> None:
    medium = (release / "medium_story.md").read_text(encoding="utf-8")
    words = len(re.findall(r"\b[\w’'-]+\b", medium))
    if not 1500 <= words <= 2000:
        raise RuntimeError(f"medium_story.md has {words} words; required 1,500–2,000.")
    thread = (release / "x_thread.md").read_text(encoding="utf-8")
    posts = re.findall(r"(?m)^\*\*(\d+/\d+)\*\*", thread)
    if not 10 <= len(posts) <= 14:
        raise RuntimeError(f"x_thread.md has {len(posts)} numbered posts; required 10–14.")


def pdf_pages(pdf: Path) -> int:
    pdfinfo = shutil.which("pdfinfo")
    if pdfinfo:
        r = subprocess.run([pdfinfo, str(pdf)], text=True, capture_output=True, check=True)
        m = re.search(r"^Pages:\s+(\d+)", r.stdout, re.M)
        if m:
            return int(m.group(1))
    try:
        import fitz  # type: ignore
        return len(fitz.open(pdf))
    except Exception as exc:
        raise RuntimeError("Need pdfinfo or PyMuPDF to validate PDF page count.") from exc


def rebuild_pdf(release: Path) -> None:
    pdflatex = shutil.which("pdflatex")
    if not pdflatex:
        raise RuntimeError("pdflatex is required to rebuild paper.pdf from paper.tex.")
    with tempfile.TemporaryDirectory(prefix="identifiability-build-") as td:
        build = Path(td)
        shutil.copy2(release / "paper.tex", build / "paper.tex")
        # The checked reference list is embedded in paper.tex; paper.bib remains a reusable
        # bibliography source and is not needed by pdflatex for the reproducible build.
        for iteration in range(3):
            run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "paper.tex"], build, capture=True)
        built = build / "paper.pdf"
        if not built.exists() or built.stat().st_size < 1000:
            raise RuntimeError("Build did not produce a nonempty PDF.")
        # Fail on unresolved citation/reference markers after repeated passes.
        log = (build / "paper.log").read_text(encoding="utf-8", errors="replace")
        unresolved = re.findall(r"(?:Citation|Reference) `[^']+' .*undefined", log)
        if unresolved:
            raise RuntimeError("Unresolved citation/reference warnings remain: " + "; ".join(unresolved[:8]))
        pages = pdf_pages(built)
        if pages < 7:
            raise RuntimeError(f"Compiled PDF has {pages} pages; at least 7 substantive pages are required.")
        shutil.copy2(built, release / "paper.pdf")
        print(f"PDF rebuilt and checked: {pages} pages, {built.stat().st_size} bytes.")


def validate_release(release: Path) -> None:
    missing = [name for name in CORE if not (release / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing core files: {missing}")
    extra = [p.name for p in release.iterdir() if p.is_file() and p.name not in CORE]
    if extra:
        raise RuntimeError(f"Unexpected files inside the release directory: {extra}")
    run([sys.executable, "verify_model.py"], release, capture=True)
    validate_metadata(release)
    check_word_counts(release)
    pages = pdf_pages(release / "paper.pdf")
    if pages < 7:
        raise RuntimeError(f"paper.pdf has {pages} pages; expected at least 7.")
    tex = (release / "paper.tex").read_text(encoding="utf-8")
    if "Co-equal research-process credit is given to Devanik Debnath and Xylia" not in tex:
        raise RuntimeError("Required Xylia research-process credit/disclosure is missing.")
    readme = (release / "README.md").read_text(encoding="utf-8")
    if r"\operatorname" in readme or r"\operatorname*" in readme:
        raise RuntimeError("README contains a prohibited operator macro.")
    # Simple delimiter balance is a smoke check, not a substitute for rendered inspection.
    if readme.count("$$") % 2:
        raise RuntimeError("README has unbalanced display-math delimiters.")
    print(f"Release preflight PASS: {len(CORE)} core files, PDF {pages} pages, metadata valid.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true", help="validate artifacts without Git writes")
    args = parser.parse_args()
    root = git_root()
    release_rel = Path("Theoretical Research") / SLUG
    release = root / release_rel
    index = root / "Theoretical Research" / "Readme.md"
    if not release.is_dir():
        raise RuntimeError(f"Expected release directory not found: {release}")
    if not index.is_file():
        raise RuntimeError(f"Parent release index not found: {index}")
    validate_release(release)
    if args.check_only:
        print("Check-only complete; no Git or remote writes performed.")
        return 0

    # Never overwrite a release directory or publish unrelated dirty changes.
    run(["git", "fetch", "origin", "main"], root)
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=root, text=True, capture_output=True, check=True).stdout.strip()
    if branch != "main":
        raise RuntimeError(f"Expected branch main, found {branch!r}.")
    local = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True).stdout.strip()
    remote = subprocess.run(["git", "rev-parse", "origin/main"], cwd=root, text=True, capture_output=True, check=True).stdout.strip()
    if local != remote:
        raise RuntimeError("Local main differs from origin/main. Reconcile the clone explicitly, then rerun.")
    status_lines = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=root, text=True, capture_output=True, check=True).stdout.splitlines()
    # Allow this release's expected untracked files; reject every unrelated local change.
    allowed_status_prefix = "?? " + str(release_rel) + "/"
    unrelated = [line for line in status_lines if not line.startswith(allowed_status_prefix)]
    if unrelated:
        raise RuntimeError(f"Working tree has unrelated changes before staging: {unrelated}")
    existing = subprocess.run(["git", "ls-tree", "HEAD", str(release_rel)], cwd=root, text=True, capture_output=True, check=True).stdout.strip()
    if existing:
        raise RuntimeError("The target release path already exists in Git; refusing to overwrite it.")

    contents = index.read_text(encoding="utf-8")
    if SLUG not in contents:
        row = ("\n| 2026.10.11 | The Identifiability Budget | Mechanistic-AI identifiability and intervention-design audit | "
               f"[{SLUG}]({SLUG}/) |\n")
        if "| Release | Title | Type | Path |" in contents:
            contents = contents.rstrip() + row
        else:
            contents = contents.rstrip() + "\n\n" + row
        index.write_text(contents, encoding="utf-8")
    run(["git", "add", "--", str(release_rel), str(Path("Theoretical Research") / "Readme.md")], root)
    staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=root, text=True, capture_output=True, check=True).stdout.splitlines()
    allowed_prefix = str(release_rel) + "/"
    allowed_index = str(Path("Theoretical Research") / "Readme.md")
    if any(not (p.startswith(allowed_prefix) or p == allowed_index) for p in staged):
        raise RuntimeError(f"Unrelated path staged; refusing commit: {staged}")
    run(["git", "commit", "-m", "Add identifiability-budget research release"], root)
    run(["git", "push", "origin", "main"], root)
    tag = VERSION
    tag_exists = subprocess.run(["git", "rev-parse", "-q", "--verify", f"refs/tags/{tag}"], cwd=root, text=True, capture_output=True)
    if tag_exists.returncode == 0:
        suffix = 1
        while subprocess.run(["git", "rev-parse", "-q", "--verify", f"refs/tags/{tag}.{suffix}"], cwd=root, capture_output=True).returncode == 0:
            suffix += 1
        tag = f"{tag}.{suffix}"
    run(["git", "tag", "-a", tag, "-m", TITLE], root)
    run(["git", "push", "origin", tag], root)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True).stdout.strip()
    print("Deployment commands completed; independently verify the remote files and tag before describing the release as complete.")
    print(f"Commit: {REPO_URL}/commit/{sha}")
    print(f"Release directory: {REPO_URL}/tree/main/{release_rel.as_posix().replace(' ', '%20')}")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
