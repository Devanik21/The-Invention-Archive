#!/usr/bin/env python3
"""Validate, rebuild, index, commit, tag, push, and verify this release.

Run from any directory inside a clean clone of Devanik21/The-Invention-Archive:
    python "Theoretical Research/Source_Channel_Identifiability_in_Aging_Biomolecular_Communication_An_Information_Theoretic_Audit/deploy.py"

Requires Python >=3.10, NumPy >=1.24, Git, and (for PDF rebuild) pdflatex.
No credentials are embedded; Git uses the user's configured authentication.
The script stages only this release directory and the parent research index.
"""
from __future__ import annotations
import argparse, datetime as dt, json, os, re, shutil, subprocess, sys, tempfile
from pathlib import Path

EXPECTED = {"paper.tex", "paper.pdf", "verify_model.py", "README.md", "medium_story.md", "x_thread.md", "CITATION.cff", "paper.bib", ".zenodo.json", "schema_article.jsonld", "deploy.py"}
RELATIVE_DIR = Path("Theoretical Research/Source_Channel_Identifiability_in_Aging_Biomolecular_Communication_An_Information_Theoretic_Audit")
INDEX = Path("Theoretical Research/Readme.md")

def run(cmd, cwd=None, capture=False, check=True):
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=capture, check=check)

def git(root: Path, *args, capture=True, check=True):
    return run(["git", *args], cwd=root, capture=capture, check=check)

def die(msg: str):
    raise SystemExit(f"ERROR: {msg}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="release tag; defaults to vYYYY.MM.DD and adds -02/-03 if already present")
    parser.add_argument("--skip-push", action="store_true", help="validate/build/index locally without creating a release commit or pushing")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root_text = git(here, "rev-parse", "--show-toplevel").stdout.strip()
    root = Path(root_text).resolve()
    rel = here.relative_to(root)
    expected_dir = (root / RELATIVE_DIR).resolve()
    if expected_dir != here:
        die(f"this script is expected at {RELATIVE_DIR}; found {rel}")
    present = {p.relative_to(here).as_posix() for p in here.rglob("*") if p.is_file()}
    if present != EXPECTED:
        die(f"recursive file manifest mismatch: missing={sorted(EXPECTED-present)}, extra={sorted(present-EXPECTED)}")
    try:
        import numpy as np
    except ImportError:
        die("NumPy >=1.24 is required to run verify_model.py")
    print(f"Python={sys.version.split()[0]}; NumPy={np.__version__}")
    run([sys.executable, str(here / "verify_model.py")], cwd=here)
    json.loads((here / ".zenodo.json").read_text(encoding="utf-8"))
    json.loads((here / "schema_article.jsonld").read_text(encoding="utf-8"))
    cff=(here / "CITATION.cff").read_text(encoding="utf-8")
    if "cff-version: 1.2.0" not in cff or "authors:" not in cff: die("CITATION.cff structural checks failed")
    medium=(here / "medium_story.md").read_text(encoding="utf-8")
    words=re.findall(r"\b[\w'-]+\b", medium)
    if not 1500 <= len(words) <= 2000: die(f"Medium draft has {len(words)} words, expected 1500-2000")
    posts=re.findall(r"(?m)^\*\*\d+/12\*\*", (here / "x_thread.md").read_text(encoding="utf-8"))
    if len(posts) != 12: die(f"X thread contains {len(posts)} numbered posts, expected 12")
    readme=(here / "README.md").read_text(encoding="utf-8")
    if len(re.findall(r"(?m)^\$\$$", readme)) != 6: die("README display-math delimiter check failed")
    if r"\n" in readme: die("README contains literal escaped newline sequence")
    if shutil.which("pdflatex") is None: die("pdflatex unavailable; cannot confirm PDF from current source")
    with tempfile.TemporaryDirectory(prefix="source-channel-build-") as tmp:
        tmpdir=Path(tmp)
        shutil.copy2(here / "paper.tex", tmpdir / "paper.tex")
        # The manuscript embeds a matching thebibliography; paper.bib is supplied as reusable metadata.
        for pass_no in (1,2):
            proc=run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "paper.tex"], cwd=tmpdir, capture=True, check=False)
            if proc.returncode != 0:
                print(proc.stdout[-8000:]); print(proc.stderr[-2000:]); die(f"LaTeX pass {pass_no} failed")
        built=tmpdir / "paper.pdf"
        if not built.is_file() or built.stat().st_size < 1000 or not built.read_bytes().startswith(b"%PDF-"):
            die("compiled PDF is absent, empty, or invalid")
        shutil.copy2(built, here / "paper.pdf")
    print(f"PASS: exact file manifest; medium_words={len(words)}; x_posts={len(posts)}; rebuilt_pdf_bytes={(here/'paper.pdf').stat().st_size}")
    if args.skip_push:
        print("LOCAL VALIDATION COMPLETE; push intentionally skipped")
        return
    branch=git(root, "branch", "--show-current").stdout.strip()
    if not branch: die("detached HEAD; check out the intended branch before deployment")
    status_lines=git(root, "status", "--porcelain").stdout.splitlines()
    rel_prefix=str(rel).replace("\\", "/")
    unexpected=[]
    for line in status_lines:
        changed=line[3:].strip().strip('"').replace("\\", "/")
        if changed == str(INDEX).replace("\\", "/") or changed == rel_prefix or changed.startswith(rel_prefix + "/") or changed.startswith(rel_prefix + " -> "):
            continue
        unexpected.append(line)
    if unexpected:
        die(f"unrelated working-tree changes detected; refusing to stage them: {unexpected}")
    # Refresh tags and observe remote branch before choosing the release tag. A non-fast-forward
    # push is never forced; if main advances later, Git rejects the push without losing history.
    run(["git", "fetch", "origin", branch, "--tags"], cwd=root)
    index_path=root/INDEX
    if not index_path.exists(): die(f"parent index does not exist: {INDEX}")
    index_text=index_path.read_text(encoding="utf-8")
    slug=here.name
    if slug not in index_text:
        row=f"| {dt.date.today().strftime("%Y.%m.%d")} | Source-Channel Identifiability in Aging Biomolecular Communication | Information-theoretic audit | [{slug}]({slug}/) |"
        index_path.write_text(index_text.rstrip()+"\n"+row+"\n", encoding="utf-8")
    run(["git", "add", "--", str(rel), str(INDEX)], cwd=root)
    staged=git(root, "diff", "--cached", "--name-only").stdout.splitlines()
    allowed={str(rel / name) for name in EXPECTED} | {str(INDEX)}
    if set(staged)-allowed: die(f"unexpected staged paths: {sorted(set(staged)-allowed)}")
    if not staged: die("no staged changes; this may already be deployed")
    run(["git", "commit", "-m", "Add source-channel identifiability research release"], cwd=root)
    commit=git(root, "rev-parse", "HEAD").stdout.strip()
    base_tag=args.tag or f"v{dt.date.today():%Y.%m.%d}"
    tags=set(git(root, "tag", "--list", capture=True).stdout.splitlines())
    remote_tags=set(git(root, "ls-remote", "--tags", "origin", capture=True).stdout.splitlines())
    remote_tag_names={line.split("refs/tags/",1)[1].removesuffix("^{}") for line in remote_tags if "refs/tags/" in line}
    tags |= remote_tag_names
    tag=base_tag
    if tag in tags and args.tag is None:
        suffix=2
        while f"{base_tag}-{suffix:02d}" in tags: suffix+=1
        tag=f"{base_tag}-{suffix:02d}"
    if tag in tags: die(f"tag already exists: {tag}; choose a different --tag")
    run(["git", "tag", "-a", tag, "-m", f"Source-channel identifiability preprint {tag}"], cwd=root)
    # Non-force pushes only. If remote advanced, Git rejects; refetch/reconcile manually and rerun.
    run(["git", "push", "origin", branch], cwd=root)
    run(["git", "push", "origin", f"refs/tags/{tag}"], cwd=root)
    run(["git", "fetch", "origin", branch, "--tags"], cwd=root)
    remote_commit=git(root, "rev-parse", f"origin/{branch}").stdout.strip()
    if remote_commit != commit: die(f"remote branch head {remote_commit} does not match release commit {commit}")
    remote_tree=git(root, "ls-tree", "-r", "--name-only", f"origin/{branch}", "--", str(rel)).stdout.splitlines()
    remote_expected={str(rel/name) for name in EXPECTED}
    if set(remote_tree) != remote_expected: die(f"remote manifest mismatch: missing={sorted(remote_expected-set(remote_tree))}, extra={sorted(set(remote_tree)-remote_expected)}")
    for name in EXPECTED:
        local_sha=git(root, "hash-object", str(rel/name)).stdout.strip()
        remote_sha=git(root, "rev-parse", f"origin/{branch}:{rel/name}").stdout.strip()
        if local_sha != remote_sha: die(f"remote file differs: {name}")
    print("DEPLOYMENT VERIFIED")
    print(f"branch={branch}; commit={commit}; tag={tag}; remote_head={remote_commit}")
    print(f"directory_url=https://github.com/Devanik21/The-Invention-Archive/tree/{branch}/Theoretical%20Research/{slug}")
    print(f"commit_url=https://github.com/Devanik21/The-Invention-Archive/commit/{commit}")

if __name__ == "__main__":
    main()
