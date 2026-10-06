# Visual testing — updated 2026-10-05

## Shape known, color unknown

The crop viewer now keeps strong shape-only evidence visible in purple, labeled
`square — color ?` (or circle/triangle), with its own **Tentative shape-only** count.
The observations table separates joint prediction, shape evidence, color evidence,
and status. Tentative is not accepted; unknown is never assumed orange. Duplicate
suppression and model thresholds still apply. Stop clears the tentative count.
See [shape/color semantics](GOAL_UNCERTAINTY.md) and [current student experiments](CV_PROGRESS.md).
Refresh an existing viewer tab to load the new interface. The teacher checkpoints
and 12-candidate processing budget are unchanged.

## Private laptop access through pacman

With Tailscale connected on your laptop, open:
**https://pacman.allosaurus-typhon.ts.net:8765/**

Use the full HTTPS hostname, not `https://pacman:8765`, so the certificate matches and
the browser permits camera access. This is private Tailscale Serve, not public Funnel.
The server remains bound to loopback. The proxy's exact HTTPS origin is explicitly allowed;
unknown hosts, cross-site origins and missing/incorrect inference tokens remain rejected.
Existing Tailscale mappings on 455, 5174, 8445 and 8446 were preserved.

The current viewer still uses the proposal-adapted MobileNetV4 teachers, NOT the experimental
YOLO checkpoints. Select a saved image or click Start camera and grant permission yourself.
Your laptop's browser sends frames to this Mac for processing; frames are not recorded.

To restart the viewer from this project directory after it stops:

```sh
.venv/bin/python webcam_app.py --allow-unvalidated --no-browser \
  --public-origin https://pacman.allosaurus-typhon.ts.net:8765
```

The persistent private proxy was configured with:

```sh
tailscale --socket=/var/run/tailscaled.socket serve --bg --https=8765 --yes http://127.0.0.1:8765
```

This proxy persists, but the Python viewer is not installed as a login/reboot service.
Browser QA through the HTTPS URL passed saved-image goal/balloon analysis, Stop, desktop
and mobile rendering, secure-context validation, and no console errors. Host/origin tests:
12 passed. Actual laptop connectivity and physical camera permission still require the user.

## Decision and launch

Both orange and yellow goals are required. The [small-region follow-up](ORANGE_REGION_PROGRESS.md)
was rejected for triangle regressions; it does not change the viewer or resolve orange detection.

Follow-up [goal-stage audit](GOAL_STAGE_PROGRESS.md) identifies small-target localization as
the main orange-goal failure. Experimental changes were rejected; the viewer below is unchanged.

Latest 2026-10-05 update: viewer defaults use the **balloon component filter** with the same
teachers, HSV ranges, 12-candidate budgets and duplicate suppression. Balloon precision/recall
is now **77.8%/81.8%**, goal **72.3%/54.6%**. See [current results and optional desktop mode](COMPONENT_PROGRESS.md).
The history below retains previous stages; rejected orange-mask/rim experiments are not defaults.

Keep OpenCV for now. It supplies cheap color segmentation and candidate boxes; MobileNetV4
is a crop classifier and cannot independently locate full-frame targets. The changed component
is our proposal algorithm, not a switch to an unmeasured detector on the original Pi Zero W.
OpenCV itself is replaceable; the localization function is required by this architecture.

From this project root, in the installed environment:

```bash
.venv/bin/python webcam_app.py --allow-unvalidated
```

Open **http://127.0.0.1:8765** on the computer running the server. Defaults now load both real
proposal-adapted V10 `.keras` teachers, not the older synthetic students. To use your conda installation,
activate its compatible project environment and run `python webcam_app.py --allow-unvalidated`.
The code AND the two `runs/{balloon,goal}-proposals-20261005/teacher.{keras,json}` pairs must be
on that computer. No MacBook sync or Pi installation was performed during this change.

1. Select Balloon or Goal.
2. **Test image → Choose File** accepts JPEG/PNG/WebP up to 15 MB. It analyzes once; no camera
   permission is needed. Use a train/validation arena image, not reserved Highbay test images.
3. Change task and click **Analyze image** to reuse the same image.
4. Or click **Start camera**, grant permission yourself, and hold up a real balloon/goal.
5. Inspect colored boxes, class scores, mask, table and processing time. **Stop** releases the
   camera and clears the frame. Images remain in memory and are not recorded by the viewer.
6. Before Stop, click **Download annotated image** below the main view or **Download color mask**
   below the mask. These save PNGs on your device (960x720 annotated, 320x240 mask), with task,
   source mode, UTC timestamp and frame number in their names. A still-image pair shares the
   same filename prefix. During live updates, each click saves the currently displayed result;
   clicks at different moments can therefore capture different frames. Downloads do not add
   server-side recording. Stop or changing tasks clears the result and disables downloads.

Download browser regression check: six PNGs from goal/balloon desktop and mobile image flows
matched the displayed canvas/mask bytes and dimensions exactly. Disabled states before a
result, after task changes and after Stop passed; no console errors or viewport overflow.
This check used saved validation imagery, not a physical camera or independent accuracy test.

The synthetic Demo only tests UI plumbing. A class score above 0.8 is not a calibrated
probability, a confirmed target, or flight permission. IDs remain simple IoU display tracks.

## Changes

- Expanded blue-purple and reflective green ranges; added reddish-orange hue wraparound.
- Kept thin rims/small targets by removing destructive opening; used raw and closed masks.
- Separated color groups, ranked compact contours, and shared the budget between colors.
- Up to **12 candidates**, versus 3 before. This is an offboard research budget, not Pi timing.
- Detection boxes remain tight; classification crops independently receive 12% context padding.
- All teacher crops in a frame run in one bounded batch. Student fallback still predicts serially.
- Viewer supports still images, defaults to real teachers, and shows the current candidate budget.

Morphological opening erodes before dilating; closing does the reverse. See the
[OpenCV reference](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html).

## Proposal coverage, not end-to-end detection accuracy

Same 595 validation frames and IoU>=0.5; public arena data, not independent Pi recordings.
The validation set informed development, so these are development metrics. Test untouched.
The before/after comparison changes both algorithm and budget, and separates boxes from crop
padding; it is not a same-compute comparison. Final 3-candidate receipts are also retained.

| Target | Previous, 3 candidates | Revised, 12 candidates |
|---|---:|---:|
| Green balloon | 49.2% | 79.5% |
| Purple balloon | 21.5% | 96.3% |
| Orange circle | 0.0% | 30.4% |
| Orange square | 1.5% | 27.5% |
| Orange triangle | 0.0% | 49.3% |
| Yellow circle | 23.3% | 98.7% |
| Yellow square | 75.3% | 94.5% |
| Yellow triangle | 0.0% | 89.1% |

## Full-frame reality check

The two unchanged real-data teachers were evaluated after proposals on all 595 validation
frames. Accepted detections use threshold 0.8 and one-to-one, same-class IoU>=0.5 matching.
Duplicates and wrong locations/classes count as false positives. Predictions overlapping generic
uncolored Balloon annotations are ignored for balloon scoring. Missing upstream labels may
also inflate measured false positives. This is not mAP or an independent final test.

| Task | True detections | False detections | Precision | Recall |
|---|---:|---:|---:|---:|
| Balloon | 529 | 1,474 | 26.4% | 81.6% |
| Goal | 578 | 1,025 | 36.1% | 48.7% |

Correction on 2026-10-05: the earlier prose/table incorrectly added the goal true detections
as 678. The retained per-class receipt totals **578**; its underlying results did not change.

**Frequent false positives remain.** Strong annotated-crop accuracy did not generalize to
arbitrary proposals. Do not distill or deploy these results as an autonomous capture system.
The proposal-adaptation follow-up below addresses some of these errors. Keep the held-out test
separate until choices are fixed.

Local M4 Max/Metal processing median/p95: balloon 61/100 ms, goal 39/67 ms. These timings
include warmup, exclude camera/transport and were measured with other local processes running.
They are not Pi, MacBook or controlled latency qualification. No flight commands were sent.

Receipts under `data/roboflow-dtr-v10-grouped/`:

- `proposal-validation.json`: original baseline.
- `proposal-v2-final-validation-{3,12}.json`: final candidate coverage at each budget.
- `full-frame-validation-v2.json`: per-class metrics and per-frame accepted detections.
- Intermediate proposal receipts are development attempts, not final results.

Reproduction, using new output names:

```bash
.venv/bin/python scripts/evaluate_dtr_proposals.py \
  --source data/roboflow-dtr-v10-grouped/coco --limit 12 --output /tmp/dtr-proposals.json
.venv/bin/python scripts/evaluate_dtr_frames.py --output /tmp/dtr-frames.json
make test
```

## Verification

### Proposal-adapted teachers — 2026-10-05

Both teachers were warm-started, including their learned heads, from the original real-data
models. New training examples match the runtime's proposal crop geometry. See
[training and label-review policy](TRAINING.md#proposal-adaptation-2026-10-05).
No proposal, 0.8 acceptance threshold, frame-resolution or validation-label changes were made
for this comparison. Full-frame metrics use the same 595 frames and one-to-one IoU>=0.5 rule.

| Task | Precision before → after | Recall before → after | F1 before → after | False detections before → after |
|---|---:|---:|---:|---:|
| Balloon | 26.4% → 42.6% | 81.6% → 81.8% | 39.9% → 56.0% | 1,474 → 715 |
| Goal | 36.1% → 48.0% | 48.7% → 54.8% | 41.4% → 51.2% | 1,025 → 706 |

Selected for **offboard research testing only**, not deployment. Class tradeoffs remain:
orange-circle recall fell from 22.0% to 17.2%; yellow-square recall fell from 91.4% to 88.2%.
Yellow-circle precision fell from 54.0% to 46.5%. Yellow-triangle recall improved from 35.3%
to 88.2%, but its precision fell from 79.2% to 41.2%. Overall improvements are not uniform gains.
Missing/wrong upstream labels and repeated arena views limit these development measurements.

Original crop-validation accuracy is 99.660% balloon and 95.206% goal after adaptation,
versus 99.915% and 96.447% before. This illustrates why clean-crop accuracy alone is insufficient.
Balloon selected fine-tune epoch 2; goal selected fine-tune epoch 1. All six scheduled epochs
per task ran; the minimum crop-validation-loss checkpoint was retained. No distillation or
reserved-test inference ran. Both original models remain available via explicit viewer flags.

Receipts:

- `runs/{balloon,goal}-proposals-20261005/{report,provenance,teacher}.json`
- `data/dtr-proposals-reviewed-20261005/{balloon,goal}/review-receipt.json`
- `data/dtr-proposals-reviewed-20261005/full-frame-validation.json`
- `data/dtr-proposals-reviewed-20261005/comparison.json`
- `configs/proposal-negative-review-20261005.json` binds AI decisions to review-queue checksums.

Reproduce the comparison with fresh output names:

```bash
.venv/bin/python scripts/evaluate_dtr_frames.py \
  --balloon-model runs/balloon-proposals-20261005/teacher.keras \
  --goal-model runs/goal-proposals-20261005/teacher.keras \
  --output /tmp/dtr-adapted-frames.json
.venv/bin/python scripts/compare_dtr_frames.py \
  --before data/roboflow-dtr-v10-grouped/full-frame-validation-v2.json \
  --after /tmp/dtr-adapted-frames.json --output /tmp/dtr-adapted-comparison.json
```

Next: improve orange-goal proposal coverage, inspect remaining full-frame false positives and
duplicates, and collect separately recorded real Pi-camera footage before distillation.
Original Pi Zero W ARMv6 compatibility, live camera timing, tracking and flight remain unverified.

Verification on 2026-10-05: Ruff and **41 tests passed** (four existing TFLite warnings).
Playwright's command wrapper was unavailable; its already-installed browser library ran
`scripts/check_viewer.mjs` without installing dependencies. Chromium verified both selected
checkpoint hashes through `/api/config`, real saved-image analysis for both tasks, masks,
observations, Stop and mobile reanalysis. No console errors, viewport overflow or camera
activation; responses retained `flight_commands: null` and deployment approval false.
Screenshots were visually inspected. Dense overlay labels still overlap around clustered
proposals; the observation table is the readable detail view. This is a functional viewer
check, not a live-camera, visual-polish or tracking-accuracy claim.

Browser evidence: `output/playwright/viewer-1791177838422/` (report and desktop/mobile images).
New inference timing measurements overlapped crop validation on this Mac and are not a
controlled speed comparison. No laptop sync, Pi installation, camera recording or flight commands.

### Duplicate suppression — 2026-10-05

**Current viewer default:** `v2` proposals, 12-candidate budget, same adapted teacher weights,
threshold 0.8, `nested` duplicate policy. No extra neural inference or retraining was added.
Compare against the proposal-adapted results immediately above, not the original teachers.

| Task | Precision before → after | Recall before → after | F1 before → after | False detections before → after |
|---|---:|---:|---:|---:|
| Balloon | 42.6% → 44.4% | 81.8% → 81.8% | 56.0% → 57.6% | 715 → 663 |
| Goal | 48.0% → 72.3% | 54.8% → 54.6% | 51.2% → 62.2% | 706 → 249 |

Same 595 development-validation frames, fixed teacher checksums, fixed threshold and target
annotations. Goal true positives decrease from 651 to 649: one yellow circle and one yellow
square lost. Other per-class true-positive counts remain unchanged. These scores are still
limited by upstream labeling errors and correlated footage, not independent competition proof.

Algorithm: for accepted predictions of the same class, near-concentric containment keeps
the larger box when the smaller area is at least 35% of the larger and at least 90% contained.
Centers must differ by no more than 15% of the larger width/height. Remaining same-class boxes
use confidence-ordered IoU>0.5 suppression. This preserves outer hoop extent instead of choosing
a high-confidence inner-rim box. Real overlapping targets can still be merged; this heuristic
does not establish instance separation or usable goal-opening geometry.

All proposals remain auditable. `raw_accepted`, `suppressed`, `suppressed_by` (observation index)
and `rejection_reason` distinguish duplicate rejection from classifier rejection. Suppression
references resolve to a final surviving candidate. The viewer table shows **Duplicate suppressed**;
the inspector counts accepted/suppressed boxes. Rejected boxes use short IDs on the image to
reduce label clutter, retaining full class/score details in the table. Display tracks still
cover candidate identities, not validated physical targets.

Orange-filter experiments were evaluated on **training frames only**. Across 208 sampled orange
targets, the original mask covers 151 at IoU>=0.5 with 12 candidates. Widening orange hue to
0–23 and reducing minimum saturation to 25 covers only 7: targets merge with warm-colored
background structures. Reducing saturation alone while keeping hue 0–12 covers 151, with
class tradeoffs and no net gain. Both experimental profiles remain opt-in audit settings and
are **not viewer defaults**. Their diagnostic 64-candidate results are not a deployment budget.
Do not infer that orange-goal coverage has been fixed.

Retained evidence:

- `data/proposal-audit-20261005/report.json` and `missed-*.jpg`: original train audit.
- `data/proposal-audit-orange-{v3,low-sat}-20261005/report.json`: rejected mask experiments.
- `data/proposal-audit-20261005/full-frame-final.json` and `.vision.py`: final results and exact source snapshot.
- `data/proposal-audit-20261005/final-comparison.json`: fixed-teacher before/after comparison.
- `output/playwright/viewer-1791178826526/`: final desktop/mobile screenshots and browser receipt.

Ruff and **47 tests passed**, four existing TFLite warnings. Browser checks used the Playwright
skill's installed-library fallback because its CLI wrapper remains unavailable. Both model
hashes and current profile/policy were verified, plus visible suppression reasons, masks,
saved-image reanalysis and Stop. No console errors, viewport overflow or camera activation.
Screenshots were visually inspected. Physical camera behavior and Pi performance remain untested.
Accepted labels draw last after the visual check caught rejected IDs covering them.

Reproduce into new output paths:

```bash
.venv/bin/python scripts/evaluate_dtr_frames.py \
  --balloon-model runs/balloon-proposals-20261005/teacher.keras \
  --goal-model runs/goal-proposals-20261005/teacher.keras \
  --profile v2 --duplicate-policy nested --output /tmp/dtr-deduplicated.json
.venv/bin/python scripts/compare_dtr_frames.py \
  --before data/dtr-proposals-reviewed-20261005/full-frame-validation.json \
  --after /tmp/dtr-deduplicated.json --allow-vision-change \
  --output /tmp/dtr-deduplicated-comparison.json
```

The comparison requires explicit `--allow-vision-change` for these ablations, and then rejects
changed teacher hashes. Default comparisons still require unchanged vision source/profile/policy.
Reserved test, distillation, camera access, Pi installation and flight control remain untouched.

### Previous viewer verification — 2026-10-04

- Ruff and **31 tests passed**; four existing TFLite warnings. Tests cover thin rims, blue-purple
  balloons, color balancing, box/crop bounds, batch parity and limits, and no flight commands.
- Real browser QA: Chromium via installed Playwright, desktop 1440x1050 and mobile 390x844.
  Browser plugin unavailable; the bundled CLI wrapper failed (`playwright-cli: command not found`),
  so the installed Playwright library was used without installing browser dependencies.
- Flow: real teacher metadata → Goal → real image → masks/boxes/table → Balloon reanalysis →
  Stop → mobile image analysis. Page identity, nonblank UI, no error overlay, console health,
  screenshots and interactions passed. No viewport overflow. Physical camera was not activated.
- Temporary QA evidence: `/tmp/dtr-vision-qa.json`, `/tmp/dtr-vision-desktop.png`,
  `/tmp/dtr-vision-mobile.png`. A warmup frame is visibly slower than later frames.
- Frontend-testing skill guided rendered desktop/mobile checks, not just backend tests.
## Resolution and FPS controls

The macOS web viewer requests a 2592 × 1944 camera source and the selected FPS as
preferences. The browser/camera can negotiate a different mode; inspect **Source
dimensions** and **Camera-reported FPS**, not just the requested values.

Both uploaded images and webcam frames pass through an aspect-preserving source
ceiling of 2592 × 1944. A 5184 × 3888 source becomes exactly 2592 × 1944; a 3840 × 2160
source becomes 2592 × 1458. Smaller sources retain their dimensions: resizing cannot
create real 5 MP detail. This intermediate is then letterboxed into the existing
320 × 240 frame for proposals and model-sized crops. There is no full-resolution
proposal search in this change. Letterboxing replaces the old stretching behavior
for non-4:3 inputs, so earlier viewer results on those inputs are not directly equivalent.

Set **Processing FPS cap** while stopped: default 5, range 0.2–30. It limits sampling
cadence, not the inference time of the model. Each tick waits for its response and
rendering, waits only the remaining frame budget, then samples the current source.
No frame queue or catch-up batch is created. **Delivered FPS** measures recent result
intervals (up to 20 results); individual intervals can vary with processing jitter.
Still images are analyzed once and explicitly have no FPS measurement.

The web viewer continues using its selected checkpoints and simple IoU tracker;
an FPS setting does not turn it into the Pi temporal runtime or simulate Pi latency,
optics, exposure, sensor readout, thermal conditions, or power consumption. Capture
may run faster than processing. The track expiry remains 0.75 seconds, so very low
FPS settings can lose IDs. No flight commands are emitted.

Validation for this change (2026-10-05):

- 210 Python tests passed, two skipped; four Node frame-policy tests passed; Ruff passed.
- Playwright (Browser plugin unavailable), 1440 × 1050 desktop and 390 × 844 mobile:
  correct page, meaningful content, no framework overlay/app console errors, no mobile overflow.
- Large generated image: 5184 × 3888 → 2592 × 1944 → 320 × 240, with actual server inference.
- Public validation image: 640 × 640 retained, letterboxed for actual inference.
- Synthetic 1280 × 720 canvas camera: requested 2592 × 1944 / 3 FPS; actual source stayed
  1280 × 720 with reported 30 FPS, separately paced processing; black letterbox bars verified.
- Slow-response simulation: selected 10 FPS / 350 ms response delay delivered about 2.72 FPS,
  minimum observed request interval 363 ms, maximum concurrent requests one; Stop prevented
  further requests. This is scheduler QA, not model or device performance evidence.
- Physical webcam permission/mode negotiation, Safari, and Pi performance were not tested.

Reproduce policy checks with `make test-web-policy` (Node required only for these tests).
The optional `scripts/check_viewer.mjs` validates real-image UI/download behavior using
an existing Playwright installation. Browser screenshots from this change are temporary
`/tmp/dtr-frame-budget-desktop.png` and `/tmp/dtr-frame-budget-mobile.png`.
