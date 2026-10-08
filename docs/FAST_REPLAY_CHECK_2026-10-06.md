# Fast temporal replay check — October 6, 2026

This is a new analysis of existing isolated Pi measurements, not a new hardware
run. Pi SSH/name resolution was unavailable during this session. No models,
thresholds, runtime binaries, Pi configuration, or ESP32 connections changed.

## Paired processed-frame localization

Both methods are scored on exactly the same source/clip/frame keys they actually
processed. Ground truth equality is checked before pairing. One-to-one IoU >=0.5
matching prevents duplicate detections from inflating recall. Only yellow goals
are scored; physical range and shape classification are not evaluated.

| Goal size, sqrt(box area) at 320x240 | Pixels A | Hybrid A | Pixels B | Hybrid B |
|---|---:|---:|---:|---:|
| Under 16px | 0/57 | 9/57 | 0/57 | 9/57 |
| 16–32px | 108/510 | 406/510 | 108/507 | 404/507 |
| 32–64px | 6/18 | 18/18 | 8/20 | 20/20 |
| 64px+ | 10/10 | 10/10 | 13/13 | 13/13 |

Pass A has 295 shared processed frames; B has 299. Among shared frames containing
yellow targets, pixels found at least one in 124/217 and 129/221; hybrid found one
in 217/217 and 221/221. This does not mean it found every target.

The fast hybrid retains a clear small-goal advantage, but under-16px all-target
recall is only 15.8% in this replay. Earlier 92.5% static tiny-target recall must
not be described as fast-live performance. Replay uses twelve transformed stills,
different target composition, temporal state, and a four-crop budget; static uses
595 images and twelve proposals. These results alone cannot isolate which factor
caused the gap. Pixels also intentionally selects one goal, unlike the hybrid.

## Scheduling and target return

Each pass schedules 360 frames at 10 FPS. Pixels processes 352/350; hybrid
processes 300/306. Thus 8/10 and 60/54 frames respectively are skipped, including
unprocessed trailing frames. Paired metrics exclude unshared frames and therefore
are not overall scheduled-frame success rates.

After the generated 300ms blackout, hybrid reacquired at least one yellow goal in
all ten yellow-containing clips, in each pass. First correct delivered results
arrived about 203–240ms after the known return time. Pixels reacquired in eight of
ten, with successful delays about 84–1239ms; two clips had no correct reacquisition
before the clip ended. Two additional clips had no observed yellow target after
return and are excluded from those ten. Delays include skipped frames and result
age, not just processing time. These are generated replay timings, not flight data.

## Next experiment, when Pi reconnects

Keep the current baseline frozen. Test whether bounded periodic wider searches
improve tiny-target discovery without unacceptable latency. Use identical clips,
fresh processes, the same runtime and input resolution, and report all-target
recall alongside any-target success and skipped frames. Do not infer that a
larger crop budget will fix this gap until measured. Actual upright camera checks
with physical targets still need staging; neither desk footage nor this replay
establishes orange/balloon range or competition accuracy.

## Reproduction

Run `scripts/analyze_isolated_replay.py` with the retained isolated `pi-results`
directory and a new `--output` JSON path. It accesses no camera or model and refuses
to overwrite an existing output. Source JSONL SHA-256 hashes are included.

Authoritative output for this analysis:
`/Volumes/HDD_Storage_Extended/dtr-isolated-tests-20261006/temporal-size-analysis-20261006-v2.json`.
The first exploratory JSON is retained; v2 corrects skipped-frame accounting to
include unprocessed tails. Detection and reacquisition results are unchanged.
