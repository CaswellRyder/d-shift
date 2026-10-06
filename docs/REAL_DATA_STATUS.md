# Real-data acquisition — 2026-10-03

**2026-10-05 follow-up:** proposal-adapted teachers are now the research-viewer defaults.
Full-frame validation improved, with class-level regressions and frequent false detections still
present. See [current visual-testing
instructions and full-frame results](TESTING.md). The older proposal measurements below are
retained as the original baseline; the original teacher artifacts below remain unchanged.
## Current outcome: DTR labeled-data blocker resolved

The user supplied authorized Roboflow export access. Downloaded **V10 and V11** of
[Cats-and-Dogs by Cheese](https://universe.roboflow.com/cheese-geozd/cats-and-dogs-bmity).
Both include 6,180 real arena images and all eight target classes. Export README and COCO
license entries state **CC BY 4.0**. Original archives, labels and license files are preserved.
The key was used in memory for downloading and was not saved in repository files or receipts.
It was supplied in chat; rotate it through Roboflow after downloading.

**Selected V10 (640x640), not V11 (180x180).** Both share all 6,054 original filename stems;
combining versions would duplicate the same examples. V10 retains more detail for small crops.
No synthetic data or Matterport photos are included in these new DTR manifests.

### Data preparation

Upstream train/valid/test directories mixed sequential frames from the same filename families.
The replacement policy was fixed before model training:

- Test: the entire `Highbay_*` family, 1,946 frames. No predictions or model selection on it.
- Validation: clips with filename start times 19:20:00–19:24:59 on 2024-10-02, 595 frames.
- Train: remaining dated clips, 2,951 frames after exclusions.
- Guard: excluded clips starting during the minute before/after that validation block.
- Same-name/full-pixel duplicates deduplicated; any such group with conflicting box/class
  labels excluded entirely. Cross-split 64-bit dHash distance <=4 removes lower-priority
  train/validation images, never moving test images into training.

**5,492 retained frames; 688 excluded:** 214 conflicting duplicate-label records, 16 duplicate
records, 435 temporal-guard frames, 22 unrecognized filenames, and one cross-split near duplicate.
Groups are **inferred from filenames, not verified recording sessions**. Unknown clip lengths,
same-arena/event correlation, and undetected near duplicates remain limitations. dHash is a
conservative heuristic, not proof of independence. Test counts do not imply 1,946 independent scenes.

AI visual review covered 64 sampled target crops (eight per class) and 40 background crops,
all from train/validation. Samples matched upstream labels and looked usable. This is not a
complete annotation review. Generic `Balloons` labels are excluded as positives but included
in background-exclusion masks. Tiny (<8px) boxes are omitted. Background crops are real image
regions outside all annotated boxes; missing upstream boxes can still cause incorrect negatives.

| Task / split | Background | Green | Purple | Total |
|---|---:|---:|---:|---:|
| Balloon train | 2,951 | 1,115 | 1,148 | 5,214 |
| Balloon validation | 594 | 485 | 99 | 1,178 |
| Balloon test, reserved | 1,946 | 332 | 715 | 2,993 |

| Goal split | Background | Orange circle | Orange square | Orange triangle | Yellow circle | Yellow square | Yellow triangle | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 2,951 | 395 | 654 | 556 | 483 | 389 | 336 | 5,764 |
| Validation | 594 | 249 | 200 | 199 | 157 | 255 | 119 | 1,773 |
| Test, reserved | 1,946 | 373 | 488 | 530 | 631 | 670 | 468 | 5,106 |

9,385 balloon and 12,643 goal crops total, including reserved splits. Background crops overlap
between task datasets; these are not 22,028 independent photographs.

### Training and artifacts

Training uses the local Mac, verified ImageNet V4 weights and unchanged teacher configs
(5 head + up to 15 fine-tuning epochs with early stopping).

Balloon teacher completed 5 head + 9 fine-tuning epochs. Selected fine-tuning epoch 4 by
validation loss. On 1,178 validation crops: **1,177 correct (99.915%), macro-F1 0.99938**.
At the unchanged provisional 0.8 threshold, 583/584 true target crops were correctly accepted,
with zero incorrect acceptances in this validation sample. One green balloon was missed.
These are correlated arena crop results, not full-frame detection or independent competition proof.

Goal teacher completed 5 head + 7 fine-tuning epochs. Selected fine-tuning epoch 2 by
validation loss. On 1,773 validation crops: **1,710 correct (96.447%), macro-F1 0.96027**.
At threshold 0.8, 1,056/1,179 true target crops were correctly accepted (89.57%); 15 accepted
predictions had the wrong target class. No background crop was accepted as a target in this
validation sample. The largest raw confusion was 33 orange circles classified as orange squares.
Further goal fine-tuning and proposal-stage work are warranted before distillation.

Both runs are complete, marked `synthetic_training: false`, `deployment_approved: false`,
`distillation_started: false`, and `test_evaluated: false`. Final `.keras` checksums and dataset
manifest checksums match provenance. Dataset qualification limits survive in metadata and reports.

- `runs/balloon-dtr-v10-20261003/`
- `runs/goal-dtr-v10-20261003/`
- `data/roboflow-dtr-v10-grouped/{balloon,goal}/manifest.json`
- `data/roboflow-dtr-v10-grouped/{audit,retained,excluded,groups}.json`
- `data/roboflow-dtr-v10-grouped/review-*.jpg`
- `configs/roboflow-dtr-review.json` records the representative visual review.
- `scripts/download_dtr.py`, `scripts/prepare_roboflow_dtr.py` reproduce acquisition/curation.

No distillation, test evaluation, Pi deployment, camera activation, or MacBook sync is implied.
No competition accuracy, full-frame proposal recall, tracking, capture or flight readiness is established.
`make test` passed Ruff and all **28 tests**, with four existing TFLite warnings. Both raw archive
SHA-256 values matched their retained download receipts; every input frame is accounted for as
retained or excluded. Both crop manifests pass class coverage, hashes and cross-split checks.

Use the original teachers in the viewer with explicit paths (current defaults load the newer
proposal-adapted real teachers):

```bash
.venv/bin/python webcam_app.py --allow-unvalidated \
  --balloon-model runs/balloon-dtr-v10-20261003/teacher.keras \
  --goal-model runs/goal-dtr-v10-20261003/teacher.keras
```

The viewer backend successfully loaded both actual completed models. On one real validation
frame, it returned both masks, zero balloon proposals and three finite goal predictions,
with no flight commands. This confirms backend integration, not rendered browser behavior,
live webcam performance, detection accuracy or laptop synchronization.

### Measured next bottleneck: proposal coverage

Ran the existing OpenCV candidate generator on all 595 validation frames, resized to the
viewer's 320x240 resolution with its unchanged limit of three candidates per task. This is
**class-agnostic box coverage**, not classifier/detector precision or end-to-end recall. It
includes tiny annotated targets that crop training omitted. Browser JPEG/resizing can differ.
No teacher inference, test-set evaluation, or runtime threshold changes were performed here.

| Labeled target | Targets | Any proposal overlaps at IoU >=0.5 |
|---|---:|---:|
| Green balloon | 541 | 49.2% |
| Purple balloon | 107 | 21.5% |
| Orange circle | 250 | 0.0% |
| Orange square | 204 | 1.5% |
| Orange triangle | 201 | 0.0% |
| Yellow circle | 159 | 23.3% |
| Yellow square | 255 | 75.3% |
| Yellow triangle | 119 | 0.0% |

Thus the classifier can be accurate on annotated crops while the viewer fails to find useful
boxes. **Before distillation: improve proposal coverage, collect hard-negative proposal crops,
and validate complete-frame teacher behavior on separate recordings.** No proposal algorithm
change is included in this data-acquisition task.

Receipt: `data/roboflow-dtr-v10-grouped/proposal-validation.json`. Reproduction (new output path):

```bash
.venv/bin/python scripts/evaluate_dtr_proposals.py \
  --source data/roboflow-dtr-v10-grouped/coco \
  --output data/roboflow-dtr-v10-grouped/proposal-validation-recheck.json
```

### Reproduction (fresh output paths only)

```bash
.venv/bin/python scripts/download_dtr.py
.venv/bin/python scripts/prepare_roboflow_dtr.py
# Inspect generated review sheets before proceeding.
.venv/bin/python scripts/prepare_roboflow_dtr.py --prepare
.venv/bin/python scripts/prepare_roboflow_dtr.py --review-backgrounds
# Inspect backgrounds; then use the teacher commands in TRAINING.md.
```

The downloader uses the export endpoint used by the official Roboflow SDK and the existing
requests dependency, avoiding dependency changes. It prompts for a key without echoing or saving
it. Scripts refuse to overwrite existing output roots. Raw archives are immutable inputs; derived
data removes/re-splits records and extracts crops. Retain Cheese attribution and CC BY 4.0 terms.

---

## Earlier acquisition attempt (before authorized Roboflow access)

**Balloon real-photo bootstrap acquired, AI color-labeled, validated, and trained. Full DTR
data blocker was then only partially resolved: no usable six-class goal training set yet.**

No Roboflow authentication, account creation, paid acquisition, cloud inference, or external
messages were used. No synthetic recoloring was represented as real photography.

## Delivered balloon dataset

Source: [Matterport Mask R-CNN v2.1 official balloon release](https://github.com/matterport/Mask_RCNN/releases/tag/v2.1).
The publisher distributes the archive for its [balloon training example](https://github.com/matterport/Mask_RCNN/tree/master/samples/balloon).

- Downloaded 74 real photographs with 305 upstream balloon polygon annotations (38.7 MB archive).
- AI assistant visually inspected all 305 region crops in four contact sheets and assigned a
  conservative subset to green, purple, or non-target-color balloons; unlisted crops excluded.
- Final dataset: **191 crops from 69 photographs**. No AI-generated imagery. These are ordinary
  balloon photos, **not DTR arena footage**, and color labels are AI-reviewed, not human-certified.
- Split before training, grouped by full source image, with exact full-image/crop hash checks.
  Original recording/photographer session independence is unknown; near-duplicate or same-event
  images may remain. This is a bootstrap, not a trustworthy competition benchmark.

| Split | Non-target/background | Green | Purple | Total |
|---|---:|---:|---:|---:|
| Train | 123 | 15 | 20 | 158 |
| Validation | 11 | 2 | 2 | 15 |
| Test, unevaluated | 9 | 4 | 5 | 18 |

`background` currently means other-color balloons. Empty rooms, people, blimps, nets, glare,
and colored arena objects still need hard-negative examples. Positive class counts and validation
coverage are very small; do not use aggregate accuracy to claim robustness.

Files:

- `data/raw/matterport-balloon/balloon_dataset.zip` (original upstream archive)
- `data/raw/matterport-balloon/review/` (305 crop previews, source index, contact sheets)
- `configs/balloon-public-review.json` (auditable AI label decisions)
- `data/balloon-public-bootstrap-20261003/manifest.json` and `receipt.json`
- `scripts/review_balloon_source.py`, `scripts/build_public_balloon.py` (reproduction)

Archive SHA-256: `053797b8a0f3c210b80d1494af99e2e65c7830c5dccca581285e61048cb0bf1c`.
Builder pins both the archive and crop-index checksums. Dataset checksums and qualification
limits are carried into the trained model metadata, provenance, and reports.

Attribution: Matterport / Waleed Abdulla, Mask R-CNN Balloon Color Splash dataset, release v2.1
(2018). The repository code is MIT; the ZIP does not enumerate individual source-photograph
licenses. The publisher explicitly provides it for training. No claim is made that MIT covers
every photograph, and redistribution is not approved by this project. Retain attribution and
resolve individual photo terms before redistributing any derived dataset/model as required.

## Completed training

`runs/balloon-public-bootstrap-20261003/` contains the real-photo teacher and receipt.
Used the existing pretrained V4 backbone and unchanged balloon config: 5 head epochs,
8 fine-tuning epochs before early stopping. Selected fine-tuning epoch 3 by validation loss.
**Training used 158 crops**, not all 191. No test evaluation or distillation was run.

Validation on just 15 crops: 14/15 correct, macro-F1 0.8744. At the provisional 0.8 acceptance
threshold, only **1 of 4 target crops was correctly accepted**. Thus, despite 93.3% overall
accuracy, this is **not a usable capture policy or a qualified teacher for distillation**.
Do not tune this result against the held-out test set. Additional positives, hard negatives,
and real camera validation matter more than another synthetic run.

Viewer command, from this project root:

```bash
python webcam_app.py --allow-unvalidated \
  --balloon-model runs/balloon-public-bootstrap-20261003/teacher.keras
```

The goal model remains the old synthetic default unless explicitly replaced. The viewer is
Mac-local; no Pi deployment or MacBook sync happened here.

## Real DTR goal draft — quarantined

[IU team's public repository](https://github.com/yk-codez/Autonomous_Aerial_Robotics) supplied
three real 4284x4284 arena photographs. I visually annotated **seven large yellow goals**:
two circles, three squares, two triangles. All images depict the same high-bay scene and must
stay in one group. Distant objects remain unannotated; automatic background mining is unsafe.

- Draft labels: `configs/iu-goal-seed-review.json`
- Draft COCO: `data/raw/iu-public-inspection/goal-draft.coco.json`
- Reproduction: `scripts/build_iu_goal_seed.py`
- No repository license was found. `training_approved: false`; no images were used for training.
- Missing: usable orange-goal coverage, independent scenes, complete labels, clear reuse terms.
- The team's pretrained model exports exist publicly but were not loaded or claimed licensed.

## Rejected / blocked sources

- Lehigh FOMO demo: downloaded pinned public Apache-2.0-repository footage at commit
  `f51c36c5f6aea44c10857f6b6fdd4654e9d7797e`. Visual inspection of 12 spaced frames revealed
  color-coded detection graphics burned into the target images. Excluded from training and
  evaluation to avoid learning overlays. Original video and LICENSE retained in `data/raw/lehigh-fomo/`.
- Roboflow Cheese/Cats-and-Dogs: matches both team colors and all six goal classes; known export
  requires authentication. Public browse request hit an access challenge this session. No bypass.
- Other university GitHub repositories, Hugging Face, and Kaggle searches did not yield a complete,
  unrestricted DTR labeled training set in this search. This is not proof no such set exists.

**Historical next step, now completed above:** authorized COCO ZIP from the Roboflow source in
`DATA_SOURCES.md`, or original arena recordings with permission to use them. The current three
yellow-goal photos are insufficient; they cannot become an independent six-class dataset merely
by splitting crops or recoloring frames.

## Verification

- `make test`: Ruff passed; 22 tests passed, with four existing TFLite warnings.
- `dtr validate`: 191 samples, `synthetic: false`, all class/split checks passed.
- `predict-teacher`: loaded the actual completed checkpoint and returned finite three-class scores.
- Rebuilt the dataset into a fresh temporary directory; manifest SHA matched the retained
  `7477871cf4efb78bbbf184c1219c475100f3b730822f9d36eae43f7b1880d291` exactly.
- Qualification limitations are persisted in model metadata/provenance/reports; test coverage
  verifies preservation and rejects explicitly quarantined COCO sources.
- No flight, camera hardware, Pi latency, or competition accuracy was established.
