# Repository instructions

- Write project documentation, comments, user-facing messages, commit messages, and pull request descriptions in English. Preserve proper names, third-party licenses, and intentionally multilingual test data where needed for coverage.
- After completing and verifying each coherent change, commit it and push it to the configured GitHub remote. The owner has authorized this workflow. Inspect the diff first and include only files belonging to the change. Do not force-push or rewrite published history.
- Keep personal recordings, datasets, checkpoints, optimizer states, secrets, and local experiment outputs out of Git. Preserve the exclusions in `.gitignore`.
- Use only the project's native models and public, publisher-paired training data. Do not introduce external pretrained weights, distillation, or cloud synthesis.
- Keep one active model. Development checkpoints are experimental until acceptance conditions pass. Passing tests or reduced losses do not establish intelligible speech.
- Run checks appropriate to the change before committing. Report the commit, push result, and any remaining limitation accurately.
