"""
deploy.py -- release packager for the Biohybrid Infrastructure Viability v1.0 folder.

What it does (local only; it does NOT push to GitHub or Zenodo):
  1. Checks that the nine root-level core files of the archive standard are present.
  2. Runs verify_model.py and stops if any analytic check fails.
  3. Writes release_bundle.zip containing every present file, plus figures/ and results/
     if they exist in the working tree.

Usage:  python3 deploy.py
"""
import os
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))

CORE_FILES = [
    "paper.tex",            # LaTeX source: NOT produced; PDF is built by ReportLab (see MANIFEST.md)
    "paper.pdf",            # built by build_pdf.py
    "verify_model.py",
    "README.md",
    "CITATION.cff",
    "paper.bib",
    ".zenodo.json",
    "schema_article.jsonld",
    "deploy.py",
]
EXTRA_DIRS = ["figures", "results"]


def main():
    missing = [f for f in CORE_FILES if not os.path.exists(os.path.join(HERE, f))]
    print("Core files missing:" if missing else "All nine core files present.")
    for f in missing:
        print("  -", f)

    print("\nRunning verify_model.py ...")
    rc = subprocess.call([sys.executable, os.path.join(HERE, "verify_model.py")])
    if rc != 0:
        print("Verification failed; bundle not written.")
        return 1

    out = os.path.join(HERE, "release_bundle.zip")
    count = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in CORE_FILES + ["MANIFEST.md"]:
            p = os.path.join(HERE, f)
            if os.path.exists(p):
                zf.write(p, f); count += 1
        for d in EXTRA_DIRS:
            base = os.path.join(HERE, d)
            if not os.path.isdir(base):
                continue
            for root, _, files in os.walk(base):
                for name in files:
                    full = os.path.join(root, name)
                    zf.write(full, os.path.relpath(full, HERE)); count += 1
    print(f"Wrote {out} with {count} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
