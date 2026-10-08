# Orange-search experiments — October 7, 2026

## Outcome

No deployment change. The synthetic scene exposed a real engineering question,
but its local-contrast winner did not generalize on the real training screen.
A bounded gap-closing supplement recovered additional goals, but also produced
more unmatched detections and hurt circle recall after duplicate suppression.
It remains a research candidate, not a replacement for the current model path.

## Fixed screen

369 real training images selected by sorted filename, every eighth image, using
the existing audit policy. All six goal classes, 320x240, fixed annotations.
208 orange and 149 yellow goal instances. Reserved test and validation sets were
not used for this tuning experiment. Correlated training images and potentially
incomplete annotations prevent treating these measurements as generalization.

### Proposal coverage at IoU >=0.5

| Method | Candidate cap | Orange covered / 208 | Yellow covered / 149 |
|---|---:|---:|---:|
| Baseline | 12 | 151 | 124 |
| Local contrast replacement | 12 | 92 | 124 |
| Baseline + local supplement | 14 | 151 | 124 |
| Baseline + expanded baseline supplement | 14 | 160 | 124 |
| Baseline + gap9 supplement | 14 | 168 | 124 |

Supplements preserve the twelve baseline candidates and append at most two
nonduplicate orange candidates. The expanded and gap9 methods examine up to 64
candidate boxes internally; this is extra search work, not 64 neural inferences.
The first supplemental implementation runs a second full proposal search and is
not yet optimized for the Pi. The four-crop temporal scheduler was not changed.

Gap9 tiny-orange coverage (longest side under16px) rose from 46/88 to 55/88.
This size convention differs from the square-root-area bins in the earlier replay
analysis. Do not combine the two sets of percentages.

### Frozen classifier check: baseline versus gap9 supplement

The same context FP32 model, SHA-256
`147671bbff83a8d0f57ff01a0fc8836441790b7b5c919f0e1ade1bb16c75fea3`,
was used for both paths at its unchanged 0.8 acceptance threshold. Same-class
one-to-one IoU >=0.5 scoring and existing nested duplicate suppression apply.

| Orange result | Baseline | Supplement |
|---|---:|---:|
| Correct detections | 146 | 158 |
| Unmatched detections (scored FP) | 107 | 129 |
| Recall | 70.2% | 76.0% |
| Precision | 57.7% | 55.1% |
| Correct circles | 33/51 | 31/51 |
| Correct squares | 59/84 | 68/84 |
| Correct triangles | 54/73 | 59/73 |

Yellow correct detections stayed at 116/149; unmatched yellow detections rose
from 68 to 69. Preserving proposals does not guarantee preserving end-to-end
recall: the additional boxes can change duplicate suppression. Do not claim the
supplement is strictly better. Unmatched detections need visual review because
the upstream label set may be incomplete.

This is stateless 12 versus up to14 crop inference on the Mac CPU backend, NOT
four-crop live tracking. No Pi FPS or latency result was obtained. Raw Mac proposal
timings are diagnostic only; some local experiments overlapped in execution and
are not a controlled speed comparison. No ESP32 or motor commands were issued.
SSH/name resolution for vision.local timed out during this session.

## Reproducible artifacts

- `data/proposal-audit-baseline-20261007/report.json`
- `data/proposal-audit-orange-local-20261007/report.json` and `gate.json` (rejected)
- `data/proposal-audit-orange-supplement-20261007/report.json`
- `data/proposal-audit-expanded-supplement-20261007/report.json`
- `data/proposal-audit-gap9-supplement-20261007/report.json` and `classification.json`
- `scripts/research_orange_supplement.py` (train-only proposal experiment)
- `scripts/verify_orange_supplement.py` (frozen-model training screen)

Reports retain hashes, frame identities, counts and classification observations.
Old receipts and the deployment baseline were not overwritten. Orange balloons
remain a separate unconfirmed class; production green/purple balloon labels are
unchanged.

## Next

Review newly accepted false positives and circle suppressions before training.
Use those examples as reviewed hard negatives only where labels are reliable.
Compare supplemental-box admission policies on training data, then select one
fixed candidate for development validation and sequential Pi profiling. A broader
synthetic set should cover those observed failure modes, not just produce more
copies of the single easy gym scene. Actual physical range remains unmeasured.
