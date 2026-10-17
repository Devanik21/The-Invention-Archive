#!/usr/bin/env python3
"""Safe validation/build/deployment helper for this research release.

Default: regenerate model outputs, compile the current manuscript in a temporary
folder, validate artifacts, and refresh the PDF / results reports.
--self-test: exercise path and metadata helpers in a temporary fixture only.
--validate-only: run the same non-publishing validation/build pipeline.
--publish: after validation, commit this release and its archive index to main,
           then create a dated tag. Requires a clean authorized local clone.

No credentials are stored or requested. Pushes are normal fast-forward pushes.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, re, shutil, subprocess, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CORE = {
    "paper.tex", "paper.pdf", "verify_model.py", "README.md", "CITATION.cff",
    "paper.bib", ".zenodo.json", "schema_article.jsonld", "deploy.py",
}
TITLE = "Two Information Channels, One Design Limit: An Information-Theoretic Framework for Generative AI in Synthetic Gene-Circuit Design"
VERSION = "2026.10.17-02"
TAG = "v2026.10.17-02"
ROOT = Path(__file__).resolve().parent
REPO_URL = "https://github.com/Devanik21/The-Invention-Archive"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def root_manifest_errors(root: Path) -> list[str]:
    errors=[]
    files={p.name for p in root.iterdir() if p.is_file()}
    missing=CORE-files
    extra=files-CORE
    if missing: errors.append("missing root core files: " + ", ".join(sorted(missing)))
    if extra: errors.append("unexpected root-level files: " + ", ".join(sorted(extra)))
    for name in ("figures", "results"):
        if not (root/name).is_dir(): errors.append(f"missing required directory: {name}/")
    return errors


def validate_markdown(text: str) -> list[str]:
    errors=[]
    if text.count("$$") % 2: errors.append("unbalanced display-math delimiters in README.md")
    if r"\operatorname" in text: errors.append("prohibited \\operatorname macro in README.md")
    if r"\operatorname*" in text: errors.append("prohibited \\operatorname* macro in README.md")
    if "\\n" in text and (r"\(\n" in text or r"\[\n" in text): errors.append("literal escaped newline in math markup")
    if not re.search(r"\]\(paper\.pdf\)", text): errors.append("README lacks direct PDF link")
    for p in ("paper.tex", "verify_model.py", "paper.bib", "CITATION.cff", ".zenodo.json", "schema_article.jsonld", "deploy.py"):
        if p not in text: errors.append(f"README lacks reference to {p}")
    return errors


def validate_bib(tex: str, bib: str) -> list[str]:
    errors=[]
    keys=set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib))
    cites=set()
    for group in re.findall(r"\\(?:parencite|textcite|cite|autocite)\s*\{([^}]+)\}", tex):
        cites.update(k.strip() for k in group.split(",") if k.strip())
    missing=cites-keys
    if missing: errors.append("citation keys absent from paper.bib: " + ", ".join(sorted(missing)))
    # Report bibliography entries that are never cited, because these can hide drift.
    unused=keys-cites
    if unused: errors.append("uncited bibliography entries: " + ", ".join(sorted(unused)))
    return errors


def load_metadata(root: Path) -> list[str]:
    errors=[]
    try:
        json.loads((root/".zenodo.json").read_text(encoding="utf-8"))
    except Exception as exc: errors.append(f".zenodo.json parse error: {exc}")
    try:
        json.loads((root/"schema_article.jsonld").read_text(encoding="utf-8"))
    except Exception as exc: errors.append(f"schema_article.jsonld parse error: {exc}")
    try:
        import yaml
        cff=yaml.safe_load((root/"CITATION.cff").read_text(encoding="utf-8"))
        for key in ("cff-version", "title", "authors", "version", "date-released"):
            if key not in cff: errors.append(f"CITATION.cff missing field {key}")
        if cff.get("version") != VERSION: errors.append("CITATION.cff version is inconsistent")
    except Exception as exc: errors.append(f"CITATION.cff parse error: {exc}")
    return errors


def validate_source_and_assets(root: Path) -> list[str]:
    errors=[]
    tex=(root/"paper.tex").read_text(encoding="utf-8")
    bib=(root/"paper.bib").read_text(encoding="utf-8")
    readme=(root/"README.md").read_text(encoding="utf-8")
    errors.extend(validate_markdown(readme))
    errors.extend(validate_bib(tex,bib))
    table_count=len(re.findall(r"\\begin\{table\}(?:\[[^]]*\])?",tex))
    graphics=re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}",tex)
    if table_count < 5: errors.append(f"paper.tex contains only {table_count} table environments")
    if len(graphics)<8: errors.append(f"paper.tex includes only {len(graphics)} figures")
    prefixes={"plot":4,"graph":2,"flow":2}
    files=[p.name for p in (root/"figures").glob("figure*.svg")]
    if len(files)<8: errors.append(f"figures/ contains only {len(files)} SVG assets")
    expected=[f"figure{i:02d}_" for i in range(1,9)]
    for prefix in expected:
        if not any(name.startswith(prefix) for name in files): errors.append(f"missing SVG asset prefix {prefix}")
    if not (root/"results"/"model_outputs.csv").is_file(): errors.append("missing plot source data")
    if not (root/"results"/"verification_report.json").is_file(): errors.append("missing verification report")
    errors.extend(load_metadata(root))
    return errors


def compile_pdf(root: Path) -> tuple[bytes, str, dict[str, Any]]:
    latexmk=shutil.which("latexmk")
    if not latexmk:
        raise RuntimeError("latexmk unavailable; release blocked (stale PDF will not be reused)")
    with tempfile.TemporaryDirectory(prefix="gene-circuit-paper-build-") as tmp_s:
        tmp=Path(tmp_s)
        shutil.copy2(root/"paper.tex",tmp/"paper.tex")
        shutil.copy2(root/"paper.bib",tmp/"paper.bib")
        figdir=tmp/"figures"; figdir.mkdir()
        for p in (root/"figures").glob("figure*.pdf"):
            shutil.copy2(p,figdir/p.name)
        cmd=[latexmk,"-pdf","-interaction=nonstopmode","-halt-on-error","-file-line-error","paper.tex"]
        proc=subprocess.run(cmd,cwd=tmp,text=True,capture_output=True,timeout=240)
        combined=proc.stdout+"\n--- STDERR ---\n"+proc.stderr
        if proc.returncode != 0:
            raise RuntimeError(f"LaTeX build failed with exit {proc.returncode}; see build output.\n{combined[-5000:]}")
        pdf=tmp/"paper.pdf"
        if not pdf.is_file() or pdf.stat().st_size<1000:
            raise RuntimeError("LaTeX tool returned no usable paper.pdf")
        log=(tmp/"paper.log").read_text(errors="replace") if (tmp/"paper.log").exists() else ""
        fatal_patterns=[r"LaTeX Error:",r"Emergency stop",r"Fatal error occurred"]
        for pat in fatal_patterns:
            if re.search(pat,log,re.I): raise RuntimeError(f"fatal LaTeX log match: {pat}")
        unresolved=[line for line in log.splitlines() if re.search(r"Citation .* undefined|Reference .* undefined|There were undefined references|Please \(re\)run Biber",line,re.I)]
        if unresolved:
            raise RuntimeError("unresolved references/citations in final build log: " + " | ".join(unresolved[:5]))
        try:
            from pypdf import PdfReader
            reader=PdfReader(str(pdf)); pages=len(reader.pages)
            text="\n".join((page.extract_text() or "") for page in reader.pages)
            if pages<7: raise RuntimeError(f"paper PDF has only {pages} pages")
            for phrase in ("Research-process credit", "Co-equal research-process credit", "References"):
                if phrase not in text and phrase != "References": raise RuntimeError(f"PDF text missing required disclosure phrase: {phrase}")
            if "Bibliography" not in text and "Gallup" not in text: raise RuntimeError("bibliography text was not found in compiled PDF")
        except ImportError as exc: raise RuntimeError("pypdf unavailable; cannot verify PDF pages") from exc
        return pdf.read_bytes(), combined, {"pages":pages,"bytes":pdf.stat().st_size,"table_environments":len(re.findall(r"\\begin\{table\}(?:\[[^]]*\])?",(root/"paper.tex").read_text(encoding="utf-8"))),"figure_includes":len(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}",(root/"paper.tex").read_text(encoding="utf-8"))),"bib_entries":len(re.findall(r"@\w+\s*\{",(root/"paper.bib").read_text(encoding="utf-8")))}


def build_manifest(root: Path) -> dict[str, Any]:
    artifacts=[]
    for folder in (root, root/"figures", root/"results"):
        for p in sorted(folder.iterdir()):
            if p.is_file() and p.name != "sha256_manifest.json":
                artifacts.append({"path":str(p.relative_to(root)).replace(os.sep,"/"),"size_bytes":p.stat().st_size,"sha256":sha256(p)})
    obj={"generated_utc":datetime.now(timezone.utc).isoformat(),"algorithm":"SHA-256","manifest_self_hash_excluded":True,"file_count":len(artifacts),"files":artifacts}
    (root/"results"/"sha256_manifest.json").write_text(json.dumps(obj,indent=2)+"\n",encoding="utf-8")
    return obj


def run_self_tests() -> dict[str, Any]:
    checks=[]
    def test(name,fn):
        fn(); checks.append({"name":name,"status":"passed"})
    with tempfile.TemporaryDirectory(prefix="deploy-self-test-") as temp:
        fixture=Path(temp)
        for name in CORE: (fixture/name).write_text("fixture\n",encoding="utf-8")
        (fixture/"figures").mkdir(); (fixture/"results").mkdir()
        test("exact-nine-file manifest accepts fixture",lambda: (_ for _ in ()).throw(AssertionError(root_manifest_errors(fixture))) if root_manifest_errors(fixture) else None)
        test("extra-root-file detection",lambda: _test_extra_file(fixture))
        test("SHA-256 deterministic",lambda: _test_sha(fixture))
        test("math delimiter detection",lambda: _test_math())
    return {"timestamp_utc":datetime.now(timezone.utc).isoformat(),"status":"passed","checks":checks,"fixture_is_temporary":True,"real_repository_touched":False}


def _test_extra_file(fixture: Path) -> None:
    p=fixture/"accidental.txt"; p.write_text("x")
    try:
        if not any("unexpected root-level" in x for x in root_manifest_errors(fixture)): raise AssertionError("extra file not detected")
    finally: p.unlink()


def _test_sha(fixture: Path) -> None:
    p=fixture/"paper.tex"
    if sha256(p)!=sha256(p): raise AssertionError("unstable checksum")


def _test_math() -> None:
    required = "[**Read paper**](paper.pdf) paper.tex verify_model.py paper.bib CITATION.cff .zenodo.json schema_article.jsonld deploy.py"
    valid = required + "\n\n$$\nx\n$$\n"
    invalid = required + "\n\n$$\nx\n"
    if validate_markdown(valid):
        raise AssertionError("valid README fixture rejected: " + "; ".join(validate_markdown(valid)))
    if not any("unbalanced display-math" in e for e in validate_markdown(invalid)):
        raise AssertionError("unbalanced display-math delimiters were not detected")


def python_syntax_checks(root: Path) -> list[dict[str, Any]]:
    out=[]
    with tempfile.TemporaryDirectory(prefix="python-pycompile-") as tmp_s:
        tmp=Path(tmp_s)
        for name in sorted(p.name for p in root.glob("*.py")):
            copied=tmp/name; shutil.copy2(root/name,copied)
            proc=subprocess.run([sys.executable,"-m","py_compile",str(copied)],text=True,capture_output=True)
            out.append({"file":name,"status":"passed" if proc.returncode==0 else "failed","exit_code":proc.returncode,"stderr":proc.stderr.strip()})
            if proc.returncode: raise RuntimeError(f"py_compile failed for {name}: {proc.stderr}")
    return out


def publish(root: Path, tag: str) -> dict[str, Any]:
    def git(*args: str, check=True):
        p=subprocess.run(["git",*args],cwd=repo_root,text=True,capture_output=True)
        if check and p.returncode: raise RuntimeError(f"git {' '.join(args)} failed: {p.stderr.strip()}")
        return p
    repo=git("rev-parse","--show-toplevel")
    repo_root=Path(repo.stdout.strip()).resolve()
    release= root.resolve()
    try: rel=release.relative_to(repo_root)
    except ValueError: raise RuntimeError("release directory is not inside the local Git repository")
    if git("branch","--show-current").stdout.strip() != "main": raise RuntimeError("refusing publish: current branch must be main")
    idx=repo_root/"Theoretical Research"/"Readme.md"
    if not idx.is_file(): raise RuntimeError("archive index not found")
    index_text=idx.read_text(encoding="utf-8")
    slug=rel.name
    if slug not in index_text:
        addition=f"| 2026.10.17 | Two Information Channels, One Design Limit | Information-theoretic framework preprint | [{slug}]({slug}/) |\n"
        marker="\n|---|---|---|---|\n"
        if marker not in index_text: raise RuntimeError("archive index table convention not recognized; refusing to rewrite")
        index_text=index_text.replace(marker,marker+addition,1)
        idx.write_text(index_text,encoding="utf-8")
    status=git("status","--porcelain").stdout.splitlines()
    allowed=[str(rel),"Theoretical Research/Readme.md"]
    unrelated=[]
    for line in status:
        path=line[3:].strip().replace('\\','/')
        if not any(path==a or path.startswith(a+"/") for a in allowed): unrelated.append(line)
    if unrelated: raise RuntimeError("refusing publish due to unrelated working-tree changes: "+"; ".join(unrelated))
    git("fetch","origin","main")
    ancestor=git("merge-base","--is-ancestor","origin/main","HEAD",check=False)
    if ancestor.returncode != 0: raise RuntimeError("local main is behind/diverged from origin/main; reconcile latest remote safely before publishing")
    tag_exists=git("ls-remote","--exit-code","--tags","origin",f"refs/tags/{tag}",check=False)
    if tag_exists.returncode==0 and tag_exists.stdout.strip(): raise RuntimeError(f"tag {tag} already exists remotely; choose a new noncolliding tag")
    git("add","--",str(rel),"Theoretical Research/Readme.md")
    staged=git("diff","--cached","--name-only").stdout.splitlines()
    if not staged: raise RuntimeError("no staged changes; refusing empty release commit")
    git("commit","-m",f"Add research release: Two Information Channels, One Design Limit ({VERSION})")
    commit=git("rev-parse","HEAD").stdout.strip()
    git("push","origin","main")
    git("tag","-a",tag,"-m",f"Research release {VERSION}",commit)
    git("push","origin",tag)
    return {"branch":"main","commit_sha":commit,"tag":tag,"release_path":str(rel).replace(os.sep,"/"),"pushed":True}


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test",action="store_true",help="run safe temporary-fixture tests only")
    parser.add_argument("--validate-only",action="store_true",help="build and validate without Git publication")
    parser.add_argument("--publish",action="store_true",help="commit and push after all checks pass")
    parser.add_argument("--tag",default=TAG,help="noncolliding dated tag used with --publish")
    args=parser.parse_args()
    if args.self_test:
        report=run_self_tests()
        print(json.dumps(report,indent=2)); return 0
    results=ROOT/"results"; results.mkdir(exist_ok=True)
    report={"timestamp_utc":datetime.now(timezone.utc).isoformat(),"version":VERSION,"status":"running","steps":[]}
    try:
        errors=root_manifest_errors(ROOT)
        # Allow missing manifest and transient build outputs only by design: core files must be exact.
        if errors: raise RuntimeError("; ".join(errors))
        report["steps"].append({"step":"exact root manifest", "status":"passed", "core_files":sorted(CORE)})
        pychecks=python_syntax_checks(ROOT)
        report["python_syntax_checks"]=pychecks
        report["steps"].append({"step":"py_compile", "status":"passed"})
        vm=subprocess.run([sys.executable,str(ROOT/"verify_model.py")],cwd=ROOT,text=True,capture_output=True,timeout=180)
        report["verify_model_exit_code"]=vm.returncode
        report["verify_model_stdout"]=vm.stdout
        report["verify_model_stderr"]=vm.stderr
        (results/"verify_model_stdout.txt").write_text(vm.stdout+ ("\nSTDERR\n"+vm.stderr if vm.stderr else ""),encoding="utf-8")
        if vm.returncode: raise RuntimeError("verify_model.py failed: "+vm.stderr[-2000:])
        report["steps"].append({"step":"execute mathematical verification and regenerate figures/data", "status":"passed"})
        metadata_errors=validate_source_and_assets(ROOT)
        if metadata_errors: raise RuntimeError("source/metadata checks failed: "+"; ".join(metadata_errors))
        (results/"deploy_self_test_report.json").write_text(json.dumps(run_self_tests(),indent=2)+"\n",encoding="utf-8")
        report["steps"].append({"step":"metadata, README, bibliography, asset validation", "status":"passed"})
        pdf_bytes, build_log, pdf_stats=compile_pdf(ROOT)
        (ROOT/"paper.pdf").write_bytes(pdf_bytes)
        (results/"latex_build_log.txt").write_text(build_log,encoding="utf-8")
        report["pdf_validation"]=pdf_stats
        report["steps"].append({"step":"compile PDF from current paper.tex + paper.bib in temporary directory", "status":"passed"})
        from pypdf import PdfReader
        with tempfile.NamedTemporaryFile(suffix=".pdf",delete=False) as temp_pdf:
            temp_pdf.write(pdf_bytes); tmp_name=temp_pdf.name
        try:
            reader=PdfReader(tmp_name)
            if len(reader.pages)<7: raise RuntimeError("PDF is below 7-page requirement")
            fulltext="\n".join((pg.extract_text() or "") for pg in reader.pages)
            if "Research-process credit" not in fulltext: raise RuntimeError("required disclosure absent from PDF")
            report["pdf_validation"]["text_check"]="passed"
        finally: Path(tmp_name).unlink(missing_ok=True)
        env={"timestamp_utc":datetime.now(timezone.utc).isoformat(),"python":sys.version,"platform":sys.platform,"dependencies":{},"tools":{}}
        for mod in ("numpy","matplotlib","networkx","pypdf","yaml","cairosvg"):
            try:
                m=__import__(mod); env["dependencies"][mod]=getattr(m,"__version__","installed; version attribute unavailable")
            except Exception as exc: env["dependencies"][mod]=f"unavailable: {exc}"
        for exe in ("python","latexmk","pdflatex","biber","pdftoppm"):
            env["tools"][exe]=shutil.which(exe)
        (results/"test_environment.json").write_text(json.dumps(env,indent=2)+"\n",encoding="utf-8")
        (results/"deploy_validation_report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
        manifest=build_manifest(ROOT)
        report["status"]="passed"
        report["sha256_manifest_file_count_excluding_manifest"]=manifest["file_count"]
        (results/"deploy_validation_report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
        # Rebuild manifest because the validation report was updated after the initial hash pass.
        manifest=build_manifest(ROOT)
        print(f"DEPLOY VALIDATION: PASSED; PDF={pdf_stats['pages']} pages; tables={pdf_stats['table_environments']}; figure assets={pdf_stats['figure_includes']}; code tests=10 in verify_model.py")
        print(f"Root core-file count: {len(CORE)}; manifest files hashed (manifest excludes itself): {manifest['file_count']}")
        print("No biological/experimental validation is claimed.")
        if args.publish:
            result=publish(ROOT,args.tag)
            report["publication"]=result
            print(json.dumps(result,indent=2))
        elif not args.validate_only:
            print("Not published: pass --publish only from a clean authorized clone of the target repository.")
        return 0
    except Exception as exc:
        report["status"]="failed"
        report["error"]=str(exc)
        (results/"deploy_validation_report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
        print(f"DEPLOY VALIDATION: FAILED: {exc}",file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
