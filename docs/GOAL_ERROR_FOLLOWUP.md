# Goal detector error follow-up — 2026-10-05

Development diagnostics only. No validation corrections, threshold changes, test inference,
checkpoint promotion or deployment approval are made by this review.

## Evidence source

Refinement epoch 2, the strongest whole checkpoint through epoch 10:
`runs/goal-detector-refined-20261005/interim-b/development-640.json`.
Report SHA256: `edfc158e08132307e75f4cf0ab8278ef41afcfb8d67bc31fa6a0f70ca62fa40c`.
The fixed operating point is confidence 0.25, same-class one-to-one IoU 0.5.
Cached diagnostics are in `interim-b/errors-640.json`; review contact sheets and frame
identities are in `interim-b/review/`. Sheets show selected examples, not unbiased prevalence.

## Yellow-square misses are concentrated in one close-up sequence

- 27 missed targets: 25 have no prediction overlapping at IoU 0.1 at the fixed confidence,
  one overlaps a wrong-class prediction at matching IoU, and one has same-class localization
  below the required IoU.
- 26 of the 27 misses have filename prefix `20241002_192258`; one has prefix
  `20241002_192234`. Filename prefixes describe source groups, not proven independent sessions.
- The contact sheet shows repeated close-up yellow square views with a green balloon and
  ceiling background. One reviewed crop instead shows an orange circle under a supplied
  yellow-square annotation. This is a label-audit concern, not permission to adjust metrics.
- Under the existing equivalent-320x240 size reporting, 25 misses are in the 32-plus-pixel
  stratum. The dominant yellow-square failure is not distant tiny-object detection.

Train/validation YOLO label counts by longest side as a fraction of the image:

All 3,545 label files were checked against their dataset receipt hashes before confirming
these counts. Dataset receipt SHA256:
`f88cff3e4a0e1d1b75acf988bca62a57c35d7d8a519f1bdea056a062702ef282`.

| Split | At most 25% | Over 25%, at most 50% | Over 50% |
| --- | ---: | ---: | ---: |
| Train | 361 | 1 | 25 |
| Validation | 190 | 38 | 27 |

All 25 training boxes over half-frame belong to prefix `20241002_192641`; all 27 validation
boxes over half-frame belong to `20241002_192258`. The inspected training image
`20241002_192641_018_jpg.rf.f29a5c1ab74999cefe1849c8d7163666.jpg` shows a different angle
and background, without the green balloon. This supports a narrow close-range coverage
hypothesis; it does not establish why the network misses the validation sequence.

## Orange-triangle false positives need box and annotation review

Of 29 false positives, 16 overlap a same-class labeled box below IoU 0.5 and 13 have no
labeled overlap at IoU 0.1. Several reviewed examples visibly contain orange triangular
frames, including clipped close-up triangles. Missing labels and inconsistent partial-object
extent may contribute; neither is automatically inferred from model disagreement.

## Next experiment boundaries

The existing run reached its 12-epoch cap. Do not extend it unchanged. Before choosing another
training change, inspect train-only close-range/partial-goal examples and the annotation
policy. A subsequent experiment may test increased close-range exposure using training
images only, with an explicit new dataset/profile receipt and all-six-class regression checks.
Do not move validation frames into training or tune boxes to individual validation labels.
Any evaluation-label audit needs a separate version and coverage beyond model-selected errors.
Independent recordings and actual Pi Camera evaluation remain necessary.
