#!/usr/bin/env python3
"""Fail-closed local release validator; no credentials and no remote writes."""
from pathlib import Path
import tempfile, shutil, sys
ROOT=Path(__file__).resolve().parent
CORE=["paper.tex","paper.pdf","verify_model.py","README.md","CITATION.cff","paper.bib",".zenodo.json","schema_article.jsonld","deploy.py"]
def validate(root):
    missing=[x for x in CORE if not (root/x).is_file()]
    if missing: raise RuntimeError("missing core files: "+", ".join(missing))
    for ext in ["*.png","*.pdf","*.svg"]:
        if len(list((root/"figures").glob(ext)))<8: raise RuntimeError("expected eight figure assets for "+ext)
    if len(list((root/"results").glob("*")))<5: raise RuntimeError("research evidence directory incomplete")
    from pypdf import PdfReader
    reader=PdfReader(str(root/"paper.pdf"))
    if len(reader.pages)<20: raise RuntimeError("PDF must contain at least 20 pages")
    text="\n".join(p.extract_text() or "" for p in reader.pages)
    if "Co-equal research-process credit" not in text: raise RuntimeError("AI research-process disclosure missing")
    return len(reader.pages)
def self_test():
    with tempfile.TemporaryDirectory(prefix="plasma-echo-fixture-") as td:
        dst=Path(td)
        for f in CORE: shutil.copy2(ROOT/f,dst/f)
        for folder in ["figures","results"]: shutil.copytree(ROOT/folder,dst/folder)
        pages=validate(dst)
        print("PASS isolated fixture self-test; PDF pages:",pages)
if __name__=="__main__":
    if "--self-test" in sys.argv: self_test()
    else: print("Usage: python deploy.py --self-test")
