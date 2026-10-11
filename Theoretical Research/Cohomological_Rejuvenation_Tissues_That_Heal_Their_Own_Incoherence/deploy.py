#!/usr/bin/env python3
"""Validate, rebuild, and optionally publish this release from a local Git clone.

Examples:
  python deploy.py
  python deploy.py --repo-root /path/to/The-Invention-Archive --publish

The publish mode uses normal Git commits and fast-forward pushes only. It refuses
an existing release tag, never force-pushes, checks the remote branch after push,
and compares all 11 release-file blobs against the remote branch. Credentials must
be provided by the user's normal Git credential manager; this script stores none.
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
TITLE = "Cohomological Rejuvenation: Tissues That Heal Their Own Incoherence"
SLUG = "Cohomological_Rejuvenation_Tissues_That_Heal_Their_Own_Incoherence"
VERSION = "v2026.10.11"
INDEX_REL = Path("Theoretical Research/Readme.md")


def run(cmd: list[str], cwd: Path, *, capture: bool = False) -> str:
    print("+", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd, check=True, text=True,
                            stdout=subprocess.PIPE if capture else None,
                            stderr=subprocess.STDOUT if capture else None)
    return result.stdout.strip() if capture else ""


def compile_pdf(src: Path) -> None:
    latexmk = shutil.which("latexmk")
    pdflatex = shutil.which("pdflatex")
    if latexmk is None and pdflatex is None:
        raise RuntimeError("Neither latexmk nor pdflatex is installed; PDF was not rebuilt.")
    with tempfile.TemporaryDirectory(prefix="cohomological-rejuvenation-") as td:
        build = Path(td)
        shutil.copy2(src / "paper.tex", build / "paper.tex")
        shutil.copy2(src / "paper.bib", build / "paper.bib")
        if latexmk:
            run([latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", "paper.tex"], build)
        else:
            for _ in range(3):
                run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", "paper.tex"], build)
        pdf = build / "paper.pdf"
        if not pdf.is_file() or pdf.stat().st_size < 1000 or not pdf.read_bytes().startswith(b"%PDF-"):
            raise RuntimeError("PDF build failed or produced an invalid document.")
        shutil.copy2(pdf, src / "paper.pdf")


def check_metadata(src: Path) -> None:
    json.loads((src / ".zenodo.json").read_text(encoding="utf-8"))
    json.loads((src / "schema_article.jsonld").read_text(encoding="utf-8"))
    cff = (src / "CITATION.cff").read_text(encoding="utf-8")
    for required in ("cff-version: 1.2.0", "title:", "authors:", "date-released:", "version:"):
        if required not in cff:
            raise RuntimeError(f"CITATION.cff lacks required field marker: {required}")


def validate(src: Path, *, build: bool) -> None:
    missing = [name for name in FILES if not (src / name).is_file()]
    extras = [p.name for p in src.iterdir() if p.is_file() and p.name not in FILES]
    if missing or extras:
        raise RuntimeError(f"Manifest mismatch: missing={missing}; extras={extras}")
    run([sys.executable, "verify_model.py"], src)
    check_metadata(src)
    medium_words = len(re.findall(r"\b[\w'-]+\b", (src / "medium_story.md").read_text(encoding="utf-8")))
    if not 1500 <= medium_words <= 2000:
        raise RuntimeError(f"Medium draft must contain 1500-2000 words; found {medium_words}.")
    thread = (src / "x_thread.md").read_text(encoding="utf-8")
    if len(re.findall(r"(?m)^\*\*\d+/12\*\*", thread)) != 12:
        raise RuntimeError("X thread must contain exactly 12 numbered posts.")
    readme = (src / "README.md").read_text(encoding="utf-8")
    if readme.count("$$") != 6 or r"\n" in readme:
        raise RuntimeError("README math typography check failed.")
    tex = (src / "paper.tex").read_text(encoding="utf-8")
    bib = (src / "paper.bib").read_text(encoding="utf-8")
    bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bib))
    cited = set()
    for group in re.findall(r"\\cite\{([^}]+)\}", tex):
        cited.update(x.strip() for x in group.split(","))
    if cited - bib_keys:
        raise RuntimeError(f"paper.bib is missing cited keys: {sorted(cited-bib_keys)}")
    if build:
        compile_pdf(src)
    if not (src / "paper.pdf").read_bytes().startswith(b"%PDF-"):
        raise RuntimeError("paper.pdf has an invalid PDF signature.")
    print(f"PASS: exact 11-file manifest; Medium={medium_words} words; X=12 posts; metadata and bibliography valid.")


def locate_repo(start: Path, explicit: Path | None) -> Path:
    if explicit:
        root = explicit.resolve()
        if not (root / ".git").exists():
            raise RuntimeError(f"--repo-root is not a Git repository: {root}")
        return root
    for parent in (start, *start.parents):
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("No local Git repository found. Supply --repo-root.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--skip-compile", action="store_true", help="Only validate the current PDF; do not rebuild it.")
    parser.add_argument("--publish", action="store_true", help="Commit, push, and verify the release on origin/main.")
    args = parser.parse_args()
    src = args.source_dir.resolve()
    validate(src, build=not args.skip_compile)
    if not args.publish:
        print(f"VALIDATED: {TITLE}\nLocal artifacts only; no remote write requested.")
        return 0

    repo = locate_repo(src, args.repo_root)
    target = repo / "Theoretical Research" / SLUG
    try:
        src.relative_to(target)
        source_is_target = True
    except ValueError:
        source_is_target = (src == target.resolve())
    if not source_is_target:
        if target.exists():
            raise RuntimeError(f"Target exists; refusing to overwrite: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, target)

    index = repo / INDEX_REL
    index.parent.mkdir(parents=True, exist_ok=True)
    old = index.read_text(encoding="utf-8") if index.exists() else (
        "# Theoretical Research\n\n| Release | Title | Type | Path |\n|---|---|---|---|\n")
    if SLUG not in old:
        row = f"| 2026.10.11 | Cohomological Rejuvenation | Sheaf-theoretic aging proposal | [{SLUG}]({SLUG}/) |"
        index.write_text(old.rstrip() + "\n" + row + "\n", encoding="utf-8")

    # Do not stage over a pre-existing user staging area or overwrite a tag.
    staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repo)
    if staged.returncode != 0:
        raise RuntimeError("Index already contains staged changes; commit or unstage them before publishing.")
    run(["git", "fetch", "origin", "main", "--tags"], repo)
    exists = subprocess.run(["git", "ls-remote", "--exit-code", "--tags", "origin", f"refs/tags/{VERSION}"], cwd=repo,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if exists.returncode == 0:
        raise RuntimeError(f"Tag {VERSION} already exists remotely; refusing to overwrite it.")
    run(["git", "add", str(target.relative_to(repo)), str(INDEX_REL)], repo)
    run(["git", "diff", "--cached", "--check", "--", ".", f":!{(target / 'README.md').relative_to(repo)}"], repo)
    pending = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repo)
    if pending.returncode == 0:
        print("No staged release changes; checking the existing remote release.")
    else:
        run(["git", "config", "user.name", "Devanik Debnath"], repo)
        run(["git", "config", "user.email", "162272415+Devanik21@users.noreply.github.com"], repo)
        run(["git", "commit", "-m", "Add cohomological rejuvenation research release"], repo)
    run(["git", "fetch", "origin", "main", "--tags"], repo)
    # Rebase is non-destructive; any conflict aborts instead of overwriting remote work.
    behind = subprocess.run(["git", "merge-base", "--is-ancestor", "origin/main", "HEAD"], cwd=repo)
    if behind.returncode != 0:
        run(["git", "rebase", "origin/main"], repo)
    run(["git", "push", "origin", "HEAD:main"], repo)
    check_tag = subprocess.run(["git", "ls-remote", "--exit-code", "--tags", "origin", f"refs/tags/{VERSION}"], cwd=repo,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if check_tag.returncode == 0:
        raise RuntimeError(f"Tag {VERSION} appeared during publish; release files were pushed but the tag was not changed.")
    run(["git", "tag", "-a", VERSION, "-m", f"Cohomological Rejuvenation preprint {VERSION}"], repo)
    run(["git", "push", "origin", f"refs/tags/{VERSION}"], repo)
    run(["git", "fetch", "origin", "main", "--tags"], repo)
    if run(["git", "rev-parse", "HEAD"], repo, capture=True) != run(["git", "rev-parse", "origin/main"], repo, capture=True):
        raise RuntimeError("Remote main does not match the just-pushed local HEAD.")
    remote_paths = run(["git", "ls-tree", "-r", "--name-only", "origin/main", "--", str(target.relative_to(repo))], repo, capture=True).splitlines()
    if len(remote_paths) != 11:
        raise RuntimeError(f"Remote manifest contains {len(remote_paths)} files, expected 11.")
    for name in FILES:
        remote_sha = run(["git", "rev-parse", f"origin/main:{(target / name).relative_to(repo)}"], repo, capture=True)
        local_sha = run(["git", "hash-object", str(target / name)], repo, capture=True)
        if remote_sha != local_sha:
            raise RuntimeError(f"Remote blob mismatch: {name}")
    print(f"PUBLISHED: {TITLE}\nBranch: main\nCommit: {run(['git','rev-parse','HEAD'], repo, capture=True)}\nTag: {VERSION}\nVerified: 11/11 remote files match local artifacts.")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)