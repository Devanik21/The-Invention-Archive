# TeX toolchain compatibility adjustment

The source generator initially selects newtxtext/newtxmath, which are not installed by the TeX Live package set on the GitHub runner. The release workflow deterministically replaces that stanza with mathptmx and explicitly loads amssymb for mathbb and related AMS symbols. Both changes are reflected in released paper.tex; the PDF is compiled from this exact patched source. No scientific content or numerical result is changed.
