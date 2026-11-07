# Standalone verifier workflow history

- Run [42357547119](https://github.com/Devanik21/The-Invention-Archive/actions/runs/42357547119) checked the final visual-review commit but stopped at the PDF page-count preflight because `pdfinfo` was not installed in that workflow runner.
- This was an environment-dependency omission in `.github/workflows/verify-mcd-release.yml`, not a PDF, manifest, or numerical-model failure.
- The standalone verifier now installs `poppler-utils` before running the exact-commit checks.
- The build workflow's independent post-commit archive verification had already passed in run [42357164937](https://github.com/Devanik21/The-Invention-Archive/actions/runs/42357164937).
- A new manifest-changing commit triggers the corrected standalone verifier; its outcome is to be recorded after the run completes.
