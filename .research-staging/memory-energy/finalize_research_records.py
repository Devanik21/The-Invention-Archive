#!/usr/bin/env python3
"""Finalize auditable candidate scores and closure hashes after the reproducible build."""
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
res = root / "results"

candidate_rows = [
    ("Memory-energy frontier for noisy recurrent inference", [4,4,4,5,5,4,4,5,5,5], "SELECTED",
     "Survived overlap screening; exact scalar derivation, deterministic tests, and a direct route to hardware calibration."),
    ("Morphogenetic error-correcting codes after cell loss", [5,3,5,3,5,4,5,3,5,4], "RESERVE",
     "High potential, but overlaps tissue-coherence/control themes and requires biological calibration."),
    ("Spectral repairability of lesioned neural circuits", [5,3,4,4,5,4,5,4,5,5], "RESERVE",
     "Tractable, but overlaps network-control and observability lines."),
    ("Active sensing for exoplanet biosignatures under abiotic mimicry", [5,3,5,3,4,5,4,3,5,4], "REJECTED_OVERLAP",
     "Adjacent to the existing alien-life release; needs instrument-specific validation."),
    ("Epigenetic reset control under latent cell-state uncertainty", [5,3,5,2,4,4,5,2,4,2], "REJECTED_OVERLAP",
     "Overlaps FER and cell-state-control releases; real biological data unavailable."),
    ("Mitochondrial heteroplasmy control with adaptive intervention", [4,3,4,3,4,4,4,3,4,3], "REJECTED_OVERLAP",
     "Overlaps existing heteroplasmy work and requires empirical parameters."),
    ("Active scientific inference under simulator misspecification", [5,4,4,5,5,5,5,5,5,5], "REJECTED_OVERLAP",
     "High raw score, but too close to the existing Identifiability Budget and information-theoretic papers."),
    ("AI co-design of cryoprotectant loading and nanowarming", [5,3,5,3,4,5,5,3,5,3], "REJECTED_OVERLAP",
     "Overlaps CryoWeave; validation requires physical measurements."),
    ("Diatom photonic crystals as paleoclimate sensors", [4,3,4,4,4,4,4,4,5,5], "REJECTED_OVERLAP",
     "Overlaps the recent Silica Light Meters release."),
    ("Information-theoretic limits of synthetic gene-circuit design", [5,3,4,5,5,5,5,5,5,5], "REJECTED_OVERLAP",
     "Overlaps the recent synthetic gene-circuit information-limit release."),
]
criteria = ["problem_importance","gap_evidence","boldness_interdisciplinary","mechanistic_plausibility",
            "mathematical_core","substantive_AI_role","human_benefit_path","first_validation_feasibility",
            "falsifiability","safety_ethics"]
with (res / "candidate_topics.csv").open("w", newline="", encoding="utf-8") as f:
    fields = ["candidate"] + criteria + ["mean_score_1_to_5","decision","selection_rationale"]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for title, scores, decision, rationale in candidate_rows:
        writer.writerow({"candidate":title, **dict(zip(criteria,scores)),
          "mean_score_1_to_5":f"{sum(scores)/len(scores):.2f}","decision":decision,"selection_rationale":rationale})

audit = """# Author-profile audit

**Audit date:** 2026-10-24. **Coverage:** targeted public-web and repository checks; not exhaustive.

- The public repository Devanik21/The-Invention-Archive was inspected through its release index. Existing releases include information-theoretic gene-circuit limits, FER, an identifiability audit, CryoWeave, mitochondrial heteroplasmy, an alien-life synthetic benchmark, and Silica Light Meters. Candidate screening explicitly considered overlap with those releases.
- The profile description supplied for this audit lists self-reported labels including Samsung Fellow @IISc, ISRO Hackathon Winner, AI Researcher, and ECE '26. This run did not independently corroborate those labels; they are not used as endorsements or credentials in the manuscript.
- The supplied ORCID URL opened a JavaScript application but did not expose machine-readable identity details in this run. It is recorded as author-supplied, not independently verified.
- Targeted searches for a confidently matched Zenodo record, OpenAIRE person/publication record, and Google Scholar author profile did not yield a verified match. This is not proof that no such record exists; no DOI, index entry, or citation count is asserted.
- No new DOI, preprint record, citation, or formal academic appointment was established. The manuscript uses the supplied NIT Agartala ECE affiliation and does not state that a degree has been awarded.
"""
(res / "author_profile_audit.md").write_text(audit, encoding="utf-8")

claim_path = res / "claim_evidence_matrix.csv"
with claim_path.open("r", newline="", encoding="utf-8") as f:
    claims = list(csv.DictReader(f))
for row in claims:
    if row["claim_id"] in {"C1","C2","C3"}: row["status"] = "PASS"
    elif row["claim_id"] == "C4": row["status"] = "NOT_TESTED"
    elif row["claim_id"] == "C5": row["status"] = "HYPOTHESIS"
with claim_path.open("w", newline="", encoding="utf-8") as f:
    writer=csv.DictWriter(f,fieldnames=list(claims[0].keys())); writer.writeheader(); writer.writerows(claims)

report=json.loads((res/"verification_report.json").read_text(encoding="utf-8"))
if report.get("status")!="PASS" or report.get("checks_failed")!=0:
    raise SystemExit("Refusing to finalize records because numerical verification is not PASS.")

tex=(root/"paper.tex").read_text(encoding="utf-8")
quality_path=res/"publication_quality_report.json"
quality=json.loads(quality_path.read_text(encoding="utf-8"))
quality["table_environments"]=len(re.findall(r"\\begin\{table\}",tex))
quality["figure_environments"]=len(re.findall(r"\\begin\{figure\}",tex))
quality["vector_figure_assets"]=len(list((root/"figures").glob("figure*.pdf")))
quality["status"]="BLOCKED"
quality["all_pages_visually_reviewed"]=False
quality["critical_major_defects"]="NOT_YET_ASSESSED"
quality["final_disposition"]="Pending independent page-by-page review of the exact PDF hash."
quality_path.write_text(json.dumps(quality,indent=2)+"\n",encoding="utf-8")

matrix_path=res/"release_check_matrix.json"
matrix=json.loads(matrix_path.read_text(encoding="utf-8"))
matrix["candidate_score_dimensions"]=criteria
matrix["release_status"]="BLOCKED"
matrix["blockers"]=[
    "Independent page-by-page visual review of the exact PDF",
    "Remote read-back and final-SHA workflow reconciliation",
    "Semantic tag awaits all gates"
]
for check in matrix.get("checks",[]):
    if check.get("name")=="Python syntax and deployment fixture tests":
        check["evidence"]="results/deploy_self_test.log; exact-SHA validation workflow"
    if check.get("name")=="LaTeX build and PDF preflight":
        check["evidence"]="results/latex_build.log; results/pdffonts_report.txt; results/publication_quality_report.json"
md=["# Release check matrix summary","","**Release status: BLOCKED**","","| Check | Status | Evidence |","|---|---|---|"]
md += [f"| {c['name']} | {c['status']} | {c['evidence']} |" for c in matrix.get("checks",[])]
md += ["","The numerical/build checks ran in this cycle. Independent page-by-page review of the exact PDF and remote closure remain pending."]
(res/"release_check_matrix.md").write_text("\n".join(md)+"\n",encoding="utf-8")

excluded={"results/sha256_manifest.txt","results/release_check_matrix.json"}
entries=[]
for p in sorted(root.rglob("*")):
    if not p.is_file() or "__pycache__" in p.parts: continue
    rel=p.relative_to(root).as_posix()
    if rel in excluded: continue
    entries.append((rel,hashlib.sha256(p.read_bytes()).hexdigest()))
manifest="# SHA-256 manifest; excludes itself and mutable release_check_matrix.json to avoid a self-referential hash.\n"
manifest += "".join(f"{digest}  {rel}\n" for rel,digest in entries)
(res/"sha256_manifest.txt").write_text(manifest,encoding="utf-8")
tree_hash=hashlib.sha256("".join(f"{rel}:{digest}\n" for rel,digest in entries).encode()).hexdigest()
matrix["artifact_tree_sha256"]=tree_hash
matrix["manifested_file_count"]=len(entries)
matrix["pdf_sha256"]=quality.get("pdf_sha256")
matrix["pdf_pages"]=quality.get("page_count")
matrix_path.write_text(json.dumps(matrix,indent=2)+"\n",encoding="utf-8")
print(f"FINALIZE_RECORDS: PASS; candidates={len(candidate_rows)}; criteria={len(criteria)}; manifest entries={len(entries)}")
