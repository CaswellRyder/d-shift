# New public red/blue corpus: downloaded, not yet admitted

2026-10-08. Source: [EngDes2 RED BLUE BALLON DATASET](https://universe.roboflow.com/engdes2/red-blue-ballon-dataset-mj745),
version 1. The public project page lists CC BY 4.0 and 4,589 source images.
The actual version API/export contains 11,057 images after augmentation, not
11,057 independent scenes. Attribute EngDes2 and Roboflow Universe if used.
The listing alone does not settle rights to every underlying image: previews
include stock-photo watermarks and mixed-source material. Do not redistribute
images or assert clean provenance from the aggregate license label.

## Acquisition receipt

Authorized API export using the existing user-provided key, hidden terminal input,
in-memory only. No credentials or signed links saved. Bounded downloader stores
a ZIP without extracting or executing its contents; 1 GB download ceiling and
2 GB remaining-disk margin. No paid inference or model API called.

- Local archive: `data/raw/engdes2-red-blue-v1-20261008/dataset.zip`
- Download: 582,557,232 bytes; advertised uncompressed contents: 584,654,322 bytes.
- SHA-256: `da21324671a143c3783685443e913a6a611c5b3aaa5249231b34567bbdf84c5e`
- Receipt: adjacent `download-receipt.json`, `training_approved: false`.
- Actual preprocessing: EXIF orientation plus 640x640 stretching; augmentation
  metadata lists three image versions, horizontal/vertical flips, noise and blur.

## Audit

`scripts/audit_public_balloon_export.py` inspected all split metadata and only
30 bounded training-image previews (six sampled images per populated class,
unioned by image ID). Validation/test image pixels were not opened.

| Upstream split | Images | Annotations | Filename groups | Images without annotations |
| --- | ---: | ---: | ---: | ---: |
| Train | 9,702 | 22,662 | 2,073 | 819 |
| Valid | 879 | 1,912 | 726 | 75 |
| Test | 476 | 1,103 | 429 | 39 |

**516 basename groups span more than one split.** This is a warning from a
filename heuristic that strips Roboflow `.rf.HASH` suffixes, not proof that every
group is the same scene. Conversely, different basenames do not prove independence.
Do not use the upstream split scores as an independent acceptance benchmark.

Observed categories: `NEWEST` (no annotations), `Kirmizi Balon`, `balloon`,
`blue_ballon`, `blue_baloon`, `red_ballon`. The 30 previews contain useful indoor
red/blue balloons, strong geometric/photometric augmentations, cropped targets,
stock photography, and inconsistent taxonomy. Generic `balloon` examples include
orange-looking targets; never silently map the generic class to red or background.
Some multi-balloon photos are incompletely or inconsistently labeled.

The audit is in `data/engdes2-red-blue-audit-20261008/audit.json`, SHA-256
`e7303fc34437343c990418f5fb7f9dd97c90a94ddd8d504f20a496d280e958cb`.
All three contact sheets were visually inspected. No class mapping or individual
training admission has been approved. New data, screenshots and ZIP remain ignored.

## Required before retraining

1. Identify indoor source families and augmentation siblings; review source-group
   boundaries, actual content duplication and overlap with the existing corpus.
2. Build an explicit label map with visual evidence. Leave generic/ambiguous
   annotations unknown. Missing boxes are not negative labels.
3. Quarantine source groups that intersect any frozen holdout. Split by recording
   or source family rather than random augmented images, and record uncertainty.
4. Review a bounded training subset, including non-balloon clutter and missed
   labels, then train with a matched baseline. Preserve the old corpus/weights.
5. Reserve separate representative live Pi sessions for the real 94% gate.

Reproduction:

```sh
PYTHONPATH=.:src .venv/bin/python scripts/download_public_balloon.py \
  --project engdes2/red-blue-ballon-dataset-mj745 --version 1 \
  --output data/raw/engdes2-red-blue-v1-20261008
PYTHONPATH=.:src .venv/bin/python scripts/audit_public_balloon_export.py \
  --root data/raw/engdes2-red-blue-v1-20261008 \
  --output data/engdes2-red-blue-audit-20261008
```

Commands refuse existing outputs. This is a new data source for investigation,
not evidence of improved model accuracy or flight readiness.
