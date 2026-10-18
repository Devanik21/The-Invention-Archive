# Full Research Package Import

The repository-connected GitHub tool can commit text files, but it cannot transfer a local binary archive directly. To complete the exact local release without re-encoding or silently omitting binaries:

1. Open this folder's [upload page](https://github.com/Devanik21/The-Invention-Archive/upload/main/staging).
2. Upload the ZIP named `Speleothems_as_Noisy_Channels_Research_Release.zip` from the ChatGPT conversation.
3. Commit the ZIP to `main`.

The `.github/workflows/import-speleothem-release.yml` workflow will check ZIP integrity and every SHA-256 listed in the package manifest, extract the full PDF/source/figures/results release into `Theoretical Research/Decoding_Deep_Time_Through_a_Diagenetic_Channel/`, then create a separate automatic commit removing the staging ZIP. It fails closed if required files or hashes do not match.

Do not treat the repository as fully published until that workflow finishes successfully and the generated commit is visible.
