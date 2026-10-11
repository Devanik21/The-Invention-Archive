#!/usr/bin/env python3
"""Validate, rebuild, and optionally deploy the CryoWeave release.

Examples:
  python deploy.py --skip-compile
  python deploy.py --repo-root /path/to/The-Invention-Archive --publish

Publishing uses an ordinary commit, an annotated version tag, fast-forward push,
and post-push checks. It refuses dirty clones, directory collisions, and existing
tags. Git credentials must come from the user's configured credential manager;
no access tokens or passwords are stored in this script.
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

FILES = [
    "paper.tex", "paper.pdf", "verify_model.py", "README.md", "medium_story.md",
    "x_thread.md", "CITATION.cff", "paper.bib", ".zenodo.json",
    "schema_article.jsonld", "deploy.py",
]
TITLE = "CryoWeave: Risk-Budgeted AI Co-Design of Cryoprotectant Loading and Nanowarming for Organ-Scale Cryobanking"
SLUG = "CryoWeave_Risk_Budgeted_AI_Co_Design_of_Cryoprotectant_Loading_and_Nanowarming_for_Organ_Scale_Cryobanking"
VERSION = "v2026.10.11.3"
REPOSITORY = "https://github.com/Devanik21/The-Invention-Archive"
INDEX_REL = Path("Theoretical Research/Readme.md")
RELEASE_PARENT = Path("Theoretical Research")


def run(command: list[str], cwd: Path, *, capture: bool = False, check: bool = True) -> str:
    result = subprocess.run(command, cwd=cwd, check=False, text=True,
                            stdout=subprocess.PIPE if capture else None,
                            stderr=subprocess.STDOUT if capture else None)
    if check and result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}\n{result.stdout or ''}")
    return (result.stdout or "").strip() if capture else ""


def compile_pdf(source: Path) -> None:
    latexmk = shutil.which("latexmk")
    pdflatex = shutil.which("pdflatex")
    if latexmk is None and pdflatex is None:
        raise RuntimeError("Neither latexmk nor pdflatex is installed; no PDF was rebuilt.")
    with tempfile.TemporaryDirectory(prefix="cryoweave-build-") as tmp:
        build = Path(tmp)
        shutil.copy2(source / "paper.tex", build / "paper.tex")
        if latexmk:
            run([latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", "paper.tex"], build)
        else:
            for _ in range(3):
                run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", "paper.tex"], build)
        pdf = build / "paper.pdf"
        if not pdf.is_file() or pdf.stat().st_size < 1000 or not pdf.read_bytes().startswith(b"%PDF-"):
            raise RuntimeError("Compilation did not produce a valid, nonempty PDF.")
        shutil.copy2(pdf, source / "paper.pdf")


def metadata_check(source: Path) -> None:
    json.loads((source / ".zenodo.json").read_text(encoding="utf-8"))
    schema = json.loads((source / "schema_article.jsonld").read_text(encoding="utf-8"))
    if schema.get("@type") != "ScholarlyArticle":
        raise RuntimeError("schema_article.jsonld must describe a ScholarlyArticle.")
    cff = (source / "CITATION.cff").read_text(encoding="utf-8")
    for marker in ("cff-version: 1.2.0", "title:", "authors:", "date-released:", "version:", "license: MIT"):
        if marker not in cff:
            raise RuntimeError(f"CITATION.cff lacks required marker: {marker}")


def validate(source: Path, *, build: bool) -> None:
    missing = [name for name in FILES if not (source / name).is_file()]
    extras = [item.name for item in source.iterdir() if item.is_file() and item.name not in FILES]
    if missing or extras:
        raise RuntimeError(f"Manifest mismatch: missing={missing}; extras={extras}")
    run([sys.executable, "verify_model.py"], source)
    metadata_check(source)
    medium = (source / "medium_story.md").read_text(encoding="utf-8")
    word_count = len(re.findall(r"\b[\w'-]+\b", medium))
    if not 1500 <= word_count <= 2000:
        raise RuntimeError(f"Medium draft must contain 1500-2000 words; found {word_count}.")
    thread = (source / "x_thread.md").read_text(encoding="utf-8")
    if len(re.findall(r"(?m)^\*\*\d+/12\*\*", thread)) != 12:
        raise RuntimeError("X thread must contain exactly twelve numbered posts.")
    readme = (source / "README.md").read_text(encoding="utf-8")
    if readme.count("$$") != 8 or r"\operatorname" in readme or r"\n" in readme:
        raise RuntimeError("README equation preflight failed: require four display blocks, no operatorname, no literal escaped newlines.")
    tex = (source / "paper.tex").read_text(encoding="utf-8")
    if "Co-equal research-process credit is given to Devanik Debnath and Xylia" not in tex:
        raise RuntimeError("The required research-process credit is missing from paper.tex.")
    bib = (source / "paper.bib").read_text(encoding="utf-8")
    bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bib))
    cited = set()
    for group in re.findall(r"\\cite\{([^}]+)\}", tex):
        cited.update(key.strip() for key in group.split(","))
    missing_keys = sorted(cited - bib_keys)
    if missing_keys:
        raise RuntimeError(f"paper.bib lacks cited keys: {missing_keys}")
    if build:
        compile_pdf(source)
    pdf = source / "paper.pdf"
    if not pdf.read_bytes().startswith(b"%PDF-"):
        raise RuntimeError("paper.pdf is not a valid PDF signature.")
    pdfinfo = shutil.which("pdfinfo")
    if not pdfinfo:
        raise RuntimeError("pdfinfo is required to check the seven-page minimum.")
    info = run([pdfinfo, str(pdf)], source, capture=True)
    match = re.search(r"(?m)^Pages:\s+(\d+)\s*$", info)
    if not match or int(match.group(1)) < 7:
        raise RuntimeError(f"PDF must contain at least seven pages; found {match.group(1) if match else 'unknown'}.")
    if pdf.stat().st_size < 1000:
        raise RuntimeError("PDF is suspiciously small.")
    print(f"PASS: exact 11-file manifest; Medium={word_count} words; X=12 posts; PDF={match.group(1)} pages; metadata and citations consistent.")


def locate_repo(explicit: Path | None) -> Path:
    if explicit is None:
        raise RuntimeError("--publish requires --repo-root pointing to a clean local repository clone.")
    root = explicit.resolve()
    if not (root / ".git").exists():
        raise RuntimeError(f"Not a Git repository: {root}")
    return root


def update_index(index: Path) -> None:
    if not index.exists():
        raise RuntimeError(f"Parent research index is missing: {index}")
    content = index.read_text(encoding="utf-8")
    if SLUG in content or TITLE in content:
        raise RuntimeError("The release already appears in the parent index; refusing to duplicate it.")
    row = f"| 2026.10.11 | CryoWeave | Robust CPA--nanowarming co-design proposal | [CryoWeave release]({SLUG}/) |"
    if "| Release | Title | Type | Path |" not in content:
        raise RuntimeError("Unknown parent index format; update manually rather than rewriting it.")
    index.write_text(content.rstrip() + "\n" + row + "\n", encoding="utf-8")


def publish(source: Path, root: Path) -> None:
    status = run(["git", "status", "--porcelain"], root, capture=True)
    if status:
        raise RuntimeError("Refusing to publish from a dirty working tree. Commit or stash existing changes first.")
    origin = run(["git", "remote", "get-url", "origin"], root, capture=True)
    if "Devanik21/The-Invention-Archive" not in origin:
        raise RuntimeError(f"Unexpected origin remote: {origin}")
    branch = run(["git", "branch", "--show-current"], root, capture=True)
    if branch != "main":
        raise RuntimeError(f"Expected to publish on main, current branch is {branch!r}.")
    target = root / RELEASE_PARENT / SLUG
    if target.exists():
        raise RuntimeError(f"Release destination already exists; refusing overwrite: {target}")
    if not source.is_dir() or source == target:
        raise RuntimeError("--source-dir must be a separate staging directory, not the final release directory.")
    run(["git", "fetch", "origin", "main", "--tags"], root)
    tag_test = run(["git", "ls-remote", "--exit-code", "--tags", "origin", f"refs/tags/{VERSION}"], root, capture=True, check=False)
    if tag_test:
        raise RuntimeError(f"Release tag already exists: {VERSION}")
    run(["git", "pull", "--ff-only", "origin", "main"], root)
    target.mkdir(parents=True)
    for filename in FILES:
        shutil.copy2(source / filename, target / filename)
    update_index(root / INDEX_REL)
    validate(target, build=False)
    run(["git", "add", "--", str(target.relative_to(root)), str(INDEX_REL)], root)
    run(["git", "commit", "-m", f"Add CryoWeave research release {VERSION}"], root)
    run(["git", "tag", "-a", VERSION, "-m", f"CryoWeave preprint {VERSION}"], root)
    run(["git", "push", "origin", "main"], root)
    run(["git", "push", "origin", f"refs/tags/{VERSION}"], root)
    run(["git", "fetch", "origin", "main", "--tags"], root)
    remote_head = run(["git", "rev-parse", "origin/main"], root, capture=True)
    local_head = run(["git", "rev-parse", "HEAD"], root, capture=True)
    if remote_head != local_head:
        raise RuntimeError("Remote main does not match the pushed release commit.")
    for filename in FILES:
        remote_blob = run(["git", "rev-parse", f"origin/main:{RELEASE_PARENT.as_posix()}/{SLUG}/{filename}"], root, capture=True)
        local_blob = run(["git", "hash-object", str(target / filename)], root, capture=True)
        if remote_blob != local_blob:
            raise RuntimeError(f"Remote file verification failed: {filename}")
    print(f"REMOTE VERIFIED: {REPOSITORY}/tree/main/{RELEASE_PARENT.as_posix()}/{SLUG}")
    print(f"COMMIT: {local_head}\nTAG: {VERSION}\nNote: GitHub Actions may separately rebuild paper.pdf if a repository workflow is configured.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--skip-compile", action="store_true", help="Validate the existing PDF without rebuilding it.")
    parser.add_argument("--publish", action="store_true", help="Copy release into a clean local clone, commit, push and verify.")
    args = parser.parse_args()
    source = args.source_dir.resolve()
    validate(source, build=not args.skip_compile)
    if args.publish:
        publish(source, locate_repo(args.repo_root))
    else:
        print(f"VALIDATED: {TITLE}\nArtifacts validated locally; no remote write requested.")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
