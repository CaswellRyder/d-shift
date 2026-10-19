# D-SHIFT model registry

## v1.0.0 — research/integration snapshot

The selected goal candidate is D-SHIFT Context-16K (FP32, 16,183 parameters). The
companion balloon baseline is Tiny-6K (INT8, 6,131 parameters). This version
freezes existing weights. Nothing was retrained, and the version number is not a
flight approval.

The small selected students, original export metadata, training provenance, epoch
histories and benchmark summaries are committed under `v1.0.0/`. The
[model card and complete lineage](../docs/MODEL_V1.md) explain how to use them.
The [private release](https://github.com/CaswellRyder/d-shift/releases/tag/v1.0.0)
holds the pretrained backbone, both the initial and proposal-adapted teachers, the
selected students, and a separate ARMv6 runtime with its dependency notices.

We do not distribute datasets, crops, cached teacher targets, recordings or the
other team's pixel implementation. Do not overwrite a version's weights or edit
its historical metadata. Create a new version and record its parent hashes instead.

```bash
python3 scripts/verify_release.py
```

`manifest.json` binds file sizes and hashes plus source-file hashes. A release tag
pins the full code and documentation commit. `lineage.json` links each
training-stage checkpoint to its exact parents and records data-manifest hashes
without including the data.
