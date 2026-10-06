# D-SHIFT model registry

## v1.0.0 — research/integration snapshot

Selected goal candidate: **D-SHIFT Context-16K**, FP32, 16,183 parameters.
Companion balloon baseline: **Tiny-6K**, INT8, 6,131 parameters.
This version freezes existing weights; it does not retrain or grant flight approval.

Small selected students, original export metadata, training provenance, epoch
histories, and benchmark summaries are committed under `v1.0.0/`.
The [model card and complete lineage](../docs/MODEL_V1.md) explain how to use them.
The [private release](https://github.com/CaswellRyder/d-shift/releases/tag/v1.0.0)
contains the pretrained backbone, both initial and proposal-adapted teachers,
the selected students, and a separate ARMv6 runtime with dependency notices.

Datasets, crops, cached teacher targets, recordings, and the other team's pixel
implementation are not distributed. Never treat a release version as flight approval.
Do not overwrite a version's weights or edit historical metadata: create a new
version and retain its parent hashes instead.

```bash
python3 scripts/verify_release.py
```

`manifest.json` binds file sizes/hashes and source-file hashes. A release tag pins
the complete code/documentation commit. `lineage.json` connects training-stage
checkpoints to their exact parents and records data-manifest hashes without data.
