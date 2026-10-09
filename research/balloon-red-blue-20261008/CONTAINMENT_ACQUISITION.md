# Targeted containment acquisition and Pi reconnect

2026-10-08. Additional reviewed training-label candidates; no retraining or
deployment in this step.

## Acquisition result

Scanned the remaining permitted source groups in the locally available,
hash-bound EngDes2 red/blue archive with both existing student seeds.

- 788 unused eligible source groups considered.
- 711 frames processed by the proposal/classification pipeline.
- Three frames had no annotations; 74 were near-duplicates of previously
  reviewed training frames and were excluded before inference.
- 55 candidate pairs found; 48 representative source/choice/color pairs queued
  for review. Both seeds' observations are retained for shared pairs.
- Reviewed all eight contact sheets and native source images for IDs 0, 38, 42.
- **46 pair labels accepted; two rejected.**

Accepted labels: **29 blue parent**, **14 red child**, **3 red parent**.
There are still **zero blue child** labels and **zero clean keep-both** labels.
The accepted pairs come from 46 filename/pixel source groups, not 46 verified
independent recording sessions. Many show related environments or nearby frames.

Do not fit the proposed selector to this collection alone: preference is strongly
confounded with color. A classifier could learn “blue means keep the parent”
without learning the intended instance/part distinction. More source groups
help, but they do not repair the absent blue-child category.

## Why two automatic labels were rejected

- ID 0, provisionally `red_balloon:keep_both`: the large proposal spans two
  distinct red balloons while the smaller proposal localizes one. An IoU just
  over 0.5 allowed the larger box to match the other annotation, but this is not
  a clean pair of separate instance boxes. Reject ambiguous supervision.
- ID 42, provisionally `red_balloon:parent`: the large proposal groups two red
  balloons while the child covers one. The automatic largest-overlap preference
  should not become a label teaching the system to discard that real instance.

This demonstrates why public box matching alone is insufficient for admission.
ID 38's distant red balloon was checked in the native image and retained as a
visible parent/part preference. Other accepted pairs include highlights, edge
fragments and clipped visible balloons; they are not metric-range labels.

## Guarded search and provenance

`scripts/mine_balloon_containment.py`:

1. Checks the exact archive and original approved manifest identities and loads
   the existing quality-review guards before searching.
2. Reads split metadata to exclude every source group appearing in validation or
   test; opens only train images in the approved four-/six-digit `frame_` families.
   All IMG families stay out, regardless of split name.
3. Excludes prior reviewed source groups, chooses one Roboflow sibling per group,
   checks source dimensions/annotation geometry, and filters prior-training
   perceptual duplicates plus exact hashes from the original reserved metadata.
4. Uses the fixed 320x240 MSER proposal pipeline, maximum 12 proposals, both
   64-pixel student models, acceptance 0.8, and ordinary NMS before pair mining.
5. Proposes parent/child preferences only for same-class nested candidates
   associated with the same known target. It can propose keep-both for distinct
   known targets each reaching IoU 0.5. All proposals remain quarantined.

Perceptual matching compares training images only. No held-out pixels were read
for deduplication, inference, label mining or fitting in this step. Source-family,
cross-split group and exact-hash checks are not proof of independent sessions or
exhaustive near-duplicate elimination against every holdout.

Queue: `data/balloon-containment-mining-20261008/review.json`, SHA-256
`5b079c0cfd2a914bfd9be451a915a71e25533a3e50a6236c31f6ce4e3dbb8f61`.

Explicit decisions: `configs/balloon-containment-review-20261008.json`.
The review has `label_review_approved: true`, **`training_approved: false`** and
`deployment_approved: false`. Accepted preference labels do not certify
complete-frame annotations, background negatives or flight performance.

Tracked evidence: [containment-acquisition.json](containment-acquisition.json)
retains accepted pair coordinates/scores/source identities, counts, review and
queue hashes, exclusions and model identities. Images remain in the ignored
data directory; the original archive remains untouched. No credentials or model
weights are committed.

## Pi is accessible again

After the user rebooted the Pi, password-based SSH succeeded. The previous
failure was from a batch/non-interactive authentication attempt; a reboot alone
is not evidence that key-based authentication was repaired.

Read-only checks confirmed:

- `Raspberry Pi Zero W Rev 1.1`, architecture `armv6l`.
- Approximately 426 MiB reported RAM; 212 MiB available at the sampled check,
  no swap used.
- Existing red-blue v3 and native-lookup benchmark folders remain present.
- `rpicam-hello` is installed. Camera detection/capture was **not** tested here.
- Initial 1-minute load average was 4.38 shortly after reboot, later falling to
  0.85. A process snapshot showed PackageKit and desktop activity; that snapshot
  is not an isolated CPU benchmark or a measurement of vision-model speed.

No Pi files, installed packages, network configuration, services, camera settings
or actuators were changed. No process was stopped. No new timing claim is made
from boot-time or desktop-contended measurements. SSH access is no longer the
current blocker, but clean timing and representative live footage remain needed.

## Verification and next steps

17 added tests cover source-family/cross-split exclusion, annotation bounds,
parent/child/keep-both distinction, unaccepted/unknown candidates, bounded scans
and refusal to overwrite an existing review queue. Full suite: **554 passed,
2 skipped**, with six existing TensorFlow warnings. Scoped Ruff passes.

Reproduce into a fresh queue:

```sh
PYTHONPATH=.:src .venv/bin/python scripts/mine_balloon_containment.py \
  --output data/balloon-containment-reproduction
```

Next model-data work should address missing blue child-preference and clean
distinct-instance examples. Controlled synthetic counterfactuals may supplement
training, but must be labeled synthetic and cannot replace independent real
qualification footage. The available real source pool was scanned under the
current strict mining rules; rerunning it unchanged will not fill the gap.
Separately, restored SSH enables an isolated Pi runtime baseline once boot-time
activity settles, using one implementation at a time.
