# Version control

Use Jujutsu (`jj`) for local version-control work in this repository going
forward. This checkout is colocated with Git; preserve Git interoperability,
existing history, tags, and remotes. See CONTRIBUTING.md for the workflow.

Inspect `jj status` and `jj diff` before changing or checkpointing work. Preserve
unrelated or concurrent edits and commit only owned paths or selected hunks.
Do not rewrite published history or push without user authorization. Keep
datasets, recordings, credentials, and generated model artifacts ignored.

Use plain descriptive pull request titles, without AI provider or model labels.

# Dataset boundary

In the EngDes2 red/blue export, keep the entire `IMG_` filename family (all splits
and augmentations) out of training, calibration and negative mining. It is reserved
for development evaluation in `configs/engdes2-development-family.json`, not the
final flight-qualification test. Preserve the existing reserved test separately.
