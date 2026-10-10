# Contributing to VoiceLLM

VoiceLLM is experimental. Contributions to alignment, synthesis, evaluation, reproducibility and documentation are welcome.

1. Open an issue describing the bug or proposed improvement.
2. Fork the repository and make a focused change on a branch.
3. Install `requirements-dev.txt` and run `python -m pytest -q` from the repository root.
4. Open a pull request explaining the change, evidence and limitations.

Use synthetic fixtures in tests. Do not commit recordings, personal data, secrets, datasets, model weights or local experiment reports. Public datasets must have documented provenance and redistribution terms. Preserve the CMU dictionary license. Never present reduced loss or passing code tests as proof of intelligible speech; perceptual quality claims need independent listening evidence and held-out evaluation.

By contributing, you agree that your original contributions are licensed under the repository's MIT license.
