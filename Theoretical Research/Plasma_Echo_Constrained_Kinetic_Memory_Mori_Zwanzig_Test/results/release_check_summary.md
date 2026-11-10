# Release check summary

**Status: BLOCKED — source package committed; PDF publication incomplete.**

Committed: manuscript source, bibliography, model code, validator, metadata, author audit, literature audit, candidate comparison, claim/evidence matrix and research log.

Not completed: GitHub Actions build, paper.pdf, generated figures, rendered-page review, remote artifact manifest and final CI closure.

Evidence:
- Workflow file: .github/workflows/build-plasma-echo-release.yml
- Workflow file commit: e2b616a60bf8b880ca6902c520ab739802458d62
- Latest release sentinel commit: 8c85bcc09ca58555822fab970c83fd737bebb5b8
- Sentinel remains at .research-staging/plasma-echo/READY.
- Remote paper.pdf lookup returned 404.
- Combined status for the sentinel commit returned no status contexts.

Next action: open GitHub Actions and manually run "Build and verify plasma-echo research release" using workflow_dispatch, or enable Actions if disabled. Do not call this release published until the PDF and evidence artifacts are remotely present and verified.
