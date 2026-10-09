# Balloon-part selection: training evidence audit

2026-10-08. Research data coverage only; no new model trained or deployed.

## Finding

The current reviewed data does **not** adequately support the next proposed
learned parent/child selector. After applying the actual frozen classifier at
0.8 and ordinary duplicate suppression, both seeds have **zero blue examples
where the smaller box should win**. Their larger-box preference examples come
from only three or four source photographs, all from Matterport.

This is a concrete data gap, not evidence that another model architecture
cannot work. More optimization epochs on these pairs would not add the missing
blue containment cases. The previous box-refinement experiment identified the
selection tradeoff; this audit identifies the missing supervision for it.

## Method

`scripts/audit_balloon_pair_supervision.py` inspects only existing approved
training sources: 23 Matterport source photos and 38 reviewed EngDes2 frames.
The Matterport manifest and source archive are hash-bound, source pixels are
checked against their recorded RGB hashes, and conflicting annotation views
are rejected. Held-out source names and pixel hashes are forbidden. Only train
members are opened; the test fixture deliberately omits the reserved image.

For Matterport, regenerate the existing capped MSER proposals on 320x240 scans.
For EngDes2, use only already admitted quality-review proposals and known target
boxes, retaining the existing source/review/pixel guards. Both student seeds run
fresh FP32 inference on their identical candidate crops. Then run ordinary NMS
before auditing the remaining same-class accepted containment pairs.

An auditable preference requires:

- Existing nested-box geometry: smaller area <=35% of larger, >=95% contained.
- Both candidates accepted as the same red/blue class at the unchanged 0.8.
- Each candidate overlaps the **same** uniquely best known same-color target.
- Exactly one box reaches IoU 0.5, with an IoU difference of at least 0.1.

Unmatched proposals, ambiguous target ties, different-target pairs and pairs
without a clear preference are excluded. Unmatched is **not** background; two
different known balloons are **not** duplicate-training labels. This is an
automatic audit of existing labels, not newly reviewed pair-label admission.
The report explicitly retains `training_approved: false` and
`pair_labels_reviewed: false`.

## Results

| Seed | Preference | Pairs | Source groups | Red pairs | Blue pairs |
| --- | --- | ---: | ---: | ---: | ---: |
| 42 | Keep larger parent | 15 | 3 | 6 | 9 |
| 42 | Keep smaller child | 3 | 3 | 3 | 0 |
| 43 | Keep larger parent | 13 | 4 | 8 | 5 |
| 43 | Keep smaller child | 4 | 4 | 4 | 0 |

For seed 42, one parent-preference photo supplies 7/15 pairs (46.7%). For seed
43, one supplies 6/13 (46.2%). Crops/pairs from one photo are not independent
examples. EngDes2 contributes two child-preference source groups to each seed,
but no accepted parent-preference pairs under these rules.

Seed 42 excludes four no-clear-preference pairs, three unmatched pairs and one
different-target pair. Seed 43 excludes three, five and two respectively.

Both seeds fail the audit's planning floor of ten source groups **per choice**.
That floor is a conservative acquisition check, not a statistical accuracy
guarantee; even passing it would not approve training labels, prove recording
independence or establish flight readiness. The more important observed failure
is the missing blue child-preference category.

An initial raw-acceptance exploratory count was larger (27/3 and 23/4 parent/
child pairs). The retained report uses the correct **post-ordinary-NMS stage**,
yielding 15/3 and 13/4. These denominators must not be interchanged.

Machine-readable evidence: [pair-supervision-coverage.json](pair-supervision-coverage.json).
It includes per-pair boxes, IoUs, model scores, source groups/domains, exclusions,
input/model/code identities and explicit non-deployment status.

## Next acquisition target

Search additional permissible **training-only** sources for:

1. Blue balloons inside oversized blue/clutter proposals where the smaller box
   correctly localizes the target; include red counterparts.
2. Real balloons containing highlights, printed details or fragments where the
   larger box should remain selected, especially indoor blimp-camera imagery.
3. Nearby or overlapping distinct same-color balloons where suppressing either
   is wrong. These need an explicit keep-both category, not forced binary labels.

Quarantine new candidate pairs for visual review before fitting. Keep whole
source groups together; do not mine any IMG development-family images or the
reserved test. The current binary-preference audit intentionally excludes
known distinct-target pairs rather than pretending it handles every selection
case. Independent future qualification footage remains separate from training.

## Hardware state and verification

A read-only batch SSH attempt reached `pacman@vision.local`, but authentication
failed with `Permission denied (publickey,password)`. This proves reachability,
not an authenticated hardware/model/camera check. No installs, deployment,
network configuration, camera use or motor commands occurred. Restored access
is still needed for isolated original-ARMv6 timing and live-camera checks.

Reproduce without overwriting evidence:

```sh
PYTHONPATH=.:src .venv/bin/python scripts/audit_balloon_pair_supervision.py \
  --output runs/pair-supervision-reproduction.json
```

20 new tests cover same-target preferences, unknown/different-target exclusions,
source grouping, inference acceptance, missing held-out pixels, manifest/archive
identity, annotation conflicts and prohibited source families. The script does
not evaluate development or reserved-test accuracy and does not produce a new
training dataset or model artifact.

Full repository suite: **537 passed, 2 skipped**, six existing TensorFlow
warnings. Scoped Ruff passes. No model weights or deployed thresholds changed.
