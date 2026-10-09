# Bright-only MSER policy experiment

2026-10-08. **Research only, not a default replacement.** This changes which
regions are searched. A generated dark-on-bright counterexample proves it is
not generally equivalent to the existing two-pass search. No model weights,
deployed policy, flight commands or training data were changed.

## Hypothesis

Red and blue opponent-color planes make strongly colored objects bright. Test
whether the bright-region pass alone preserves useful proposals while reducing
MSER and neural-classification work. This is one fixed ablation, not a sweep of
thresholds against development labels.

The [OpenCV 4.10 implementation](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/mser.cpp#L946-L973)
exposes `setPass2Only(true)`, skipping the darker-to-brighter traversal and
retaining the brighter-to-darker traversal for grayscale inputs. We apply it to
both opponent-color planes. Delta5, minimum6 and maximum25000 area, native
direct-address reduction, proposal scoring/budget, model and suppression stay
unchanged. `bright_only=False` remains the default.

## Development evidence

Same 16 hash-bound development scenes, not a new qualification set. Original:
four off-domain photos, two red/three blue boxes. Indoor: twelve images, eleven
red/twelve blue boxes. IMG-family scenes remain evaluation-only; no reserved
test images were opened and no images were admitted to training.

Both seeds retain the same aggregate TP / FP / FN at threshold0.8 and IoU0.5:

- Seed42 original: red 2 / 2 / 0, blue 3 / 0 / 0.
- Seed42 indoor: red 11 / 0 / 0, blue 9 / 4 / 3.
- Seed43 original: red 2 / 2 / 0, blue 3 / 1 / 0.
- Seed43 indoor: red 10 / 2 / 1, blue 8 / 4 / 4.

Selected candidates decrease from **90 to 72**, hence average neural calls from
5.625 to 4.5 per frame. Geometric proposal coverage is 26 of 28 annotated targets
for both policies (best selected-box IoU at least0.5, irrespective of color).
These counts do not imply unchanged boxes, scores, candidate ordering or recall
on unseen scenes. Host per-detection latency fields are diagnostic measurements,
not deterministic prediction values.

## Explicit failure case

A generated 320x240 image has a solid RGB(230,0,0) background and a radius18
RGB(100,0,0) disk centered at(160,120). Two-pass search yields the disk proposal;
bright-only search yields none. Conversely, isolated red and blue disks on a
black background have identical proposals in both modes. Unit tests retain both
examples for both native-pointer modes.

These are algorithmic fixtures, not photographic training examples or labeled
real balloons. They establish a missing-proposal mechanism, not a measured
flight miss rate. The real application could encounter a target less saturated
or darker than surrounding same-color clutter. Never infer universal safety
from the unchanged small-panel metrics.

## Pi protocol and evidence

The new opt-in `mser_bright` replay mode has its **own** desktop prediction golden.
The builder verifies the original control against the original golden, preserves
that original entry, then adds the new policy entry. Passing the new golden is
cross-runtime consistency, not equivalence to the old search.

An isolated bundle was built at `artifacts/balloon-pi-bright-search-20261008` and
copied to `/home/pacman/balloon-pi-bright-search-20261008`. Bundle SHA:
`254c45afce621817421e4cc7f29a9da6c85244e96da33fac63308ceaf1aa68eb`.
The student, native reducer and native TFLite runtime match the preceding
direct-address experiment. The native reducer compiled to the identical binary.

Trial order: direct A, bright A, bright B, direct B; two timed rounds of the same
16 images per trial. One vision process, one OpenCV/inference thread; no camera
capture, file/model loading or instrumentation in processing timings. Ordinary
Pi system services remain enabled. Independent recording/session coverage is
not established by repeated processing.

All four trials completed. Pooled processing throughput is **3.1901 FPS** for
two-pass direct search and **3.9824 FPS** for bright-only, a **24.84% increase**.
Mean latency fell **19.89%**, from 313.469 to 251.107 ms/frame. Each method has
64 timed samples over 16 unique scenes. Camera capture is excluded.

Per-scene TP/FP/FN counts are unchanged on all 16 frames. Full detection lists
change on six frames, so this must not be described as exact prediction parity.
The known generated counterexample remains a reason not to switch the default.
All recorded throttle flags are `0x0`; no Python/camera process remained at the
post-run check. No profiling or other vision implementation ran concurrently.

`bright-pass-pi-measurements.json` retains hash-bound per-trial reports, pooled
timings, identities and per-scene comparisons. Raw logs stay ignored under
`artifacts/balloon-pi-bright-results-20261008`. Original input identities and all
original golden entries were independently rechecked after bundle generation.

Full current-working-tree suite: **697 passed, 2 skipped**. Scoped Ruff checks
passed. Tests include isolated bright objects, the omitted dark-on-bright
proposal, wrong-policy goldens, changed identities, missing/repeated trials and
the distinction between unchanged counts and semantic equivalence. Unrelated
dirty edits remain uncommitted and were not reviewed as part of this experiment.

The policy comparator checks common model/bundle/input/runtime/native/benchmark
identities, correct per-policy goldens, repeated-prediction consistency, and
per-scene detection counts. It explicitly reports `general_semantic_equivalence`
as false and never grants deployment or flight authority.

Host evidence: `bright-pass-development.json`.
Reproduction with fresh output paths:

```sh
.venv/bin/python -m scripts.research_balloon_mser_pass --base artifacts/balloon-pi-direct-search-20261008 --library artifacts/balloon-regions-mac-20261008/regions.so --output artifacts/fresh-bright-development.json
.venv/bin/python -m scripts.build_bright_balloon_pi_bundle --reference artifacts/balloon-pi-direct-search-20261008 --output artifacts/fresh-bright-bundle --library artifacts/balloon-regions-mac-20261008/regions.so
```

On the Pi, compile the native reducer in the new bundle. Use
`pi_balloon_search_bench.py --search mser_direct` for control or `--search
mser_bright` for intervention, with the existing one-thread/native-runtime
environment, `--region-library native/regions.so`, `--rounds 2`, and unique output
names. Copy the reports/logs and compare with
`scripts.summarize_bright_balloon_search --base ... --results ... --output ...`.

Next potential use is an explicitly bounded multi-rate search: cheap bright
passes between periodic full passes, with track age and reacquisition delay
measured on temporal footage. That scheduler is not implemented or qualified by
this experiment. Keeping bright-only as the sole search would ignore a known
failure mode. Accuracy and the 94% flight-readiness target remain unresolved.
