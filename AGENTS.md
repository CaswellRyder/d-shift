# Version control

Use Jujutsu (`jj`) for local version-control work in this repository going
forward. This checkout is colocated with Git; preserve Git interoperability,
existing history, tags, and remotes. See CONTRIBUTING.md for the workflow.

Inspect `jj status` and `jj diff` before changing or checkpointing work. Preserve
unrelated or concurrent edits and commit only owned paths or selected hunks.
Do not rewrite published history or push without user authorization. Keep
datasets, recordings, credentials, and generated model artifacts ignored.

Use plain descriptive pull request titles, without AI provider or model labels.
