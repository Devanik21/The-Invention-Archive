#!/usr/bin/env python3
"""Master deployment engine for the Mortal Computation release.

Run from the release directory or from the parent `Theoretical Research` directory.
The script validates the 11-file release payload, runs numerical verification, compiles the
LaTeX preprint, patches the parent index, writes deterministic SHA-256 hashes into README.md,
and creates a local git tag vYYYY.MM.DD when the repository is available.

No LinkedIn artifact or cross-promotion is generated.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

TITLE = "Mortal Computation via Multi-Scale Interoceptive Morphogenesis"
TAG = "v2026.10.04"
CORE = [
    "paper.tex", "paper.pdf", "verify_model.py", "README.md", "medium_story.md", "x_thread.md",
    "CITATION.cff", "paper.bib", ".zenodo.json", "schema_article.jsonld", "deploy.py"
]
ROOT_NAME = Path("Theoretical Research")


def run(cmd: list[str], cwd: Path) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def locate_release() -> Path:
    here = Path.cwd()
    candidates = [
        here,
        here / ROOT_NAME / TITLE,
        here / TITLE,
    ]
    for p in candidates:
        if all((p / f).exists() for f in CORE if f != "paper.pdf"):
            return p
    raise SystemExit("Could not locate the release directory. Run deploy.py from the release directory or repository root.")


def patch_parent_index(release: Path) -> None:
    parent = release.parent
    index = parent / "Readme.md"
    if not index.exists():
        index.write_text("# Theoretical Research\n\n| Date | Paper | PDF | Code | Status |\n|---|---|---|---|---|\n", encoding="utf-8")
    text = index.read_text(encoding="utf-8")
    if "| Date | Paper | PDF | Code | Status |" not in text:
        text = "# Theoretical Research\n\n| Date | Paper | PDF | Code | Status |\n|---|---|---|---|---|\n" + text
    row = f"| 2026-10-04 | [{TITLE}](./{TITLE}/) | [paper.pdf](./{TITLE}/paper.pdf) | [verify_model.py](./{TITLE}/verify_model.py) | Theoretical preprint |"
    lines = [ln for ln in text.splitlines() if TITLE not in ln]
    if row not in lines:
        lines.append(row)
    index.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def refresh_hashes(release: Path) -> None:
    readme = release / "README.md"
    text = readme.read_text(encoding="utf-8")
    marker = "## File SHA-256 manifest"
    section = marker + "\n\n| File | SHA-256 |\n|---|---|\n"
    for name in CORE:
        if name == "README.md":
            continue
        path = release / name
        if path.exists():
            section += f"| `{name}` | `{sha256(path)}` |\n"
    if marker in text:
        text = text.split(marker)[0].rstrip() + "\n\n" + section.rstrip() + "\n"
    else:
        text = text.rstrip() + "\n\n" + section.rstrip() + "\n"
    readme.write_text(text, encoding="utf-8")


def git_tag(repo_root: Path) -> None:
    try:
        run(["git", "rev-parse", "--is-inside-work-tree"], repo_root)
        exists = subprocess.run(["git", "rev-parse", "-q", "--verify", f"refs/tags/{TAG}"], cwd=repo_root, check=False).returncode == 0
        if not exists:
            run(["git", "tag", TAG], repo_root)
        print(f"Semantic tag ready: {TAG}")
    except Exception as exc:
        print(f"Git tag skipped: {exc}")


def main() -> int:
    release = locate_release()
    required_text = [f for f in CORE if f != "paper.pdf"]
    missing = [f for f in required_text if not (release / f).exists()]
    if missing:
        raise SystemExit(f"Missing release files: {missing}")

    if "linkedin" in " ".join(CORE).lower():
        raise SystemExit("LinkedIn invariant violated.")

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run([sys.executable, str(release / "verify_model.py")], cwd=release, check=True, env=env)
    run(["pdflatex", "-interaction=nonstopmode", "paper.tex"], release)
    bib = "bibtex" if __import__("shutil").which("bibtex") else "bibtex8"
    run([bib, "paper"], release)
    run(["pdflatex", "-interaction=nonstopmode", "paper.tex"], release)
    run(["pdflatex", "-interaction=nonstopmode", "paper.tex"], release)

    if not (release / "paper.pdf").exists():
        raise SystemExit("paper.pdf was not generated.")
    patch_parent_index(release)
    refresh_hashes(release)

    repo_root = release.parent.parent
    git_tag(repo_root)
    print(f"Release validated: {release}")
    print(f"Core file count: {sum((release / f).exists() for f in CORE)} / 11")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
