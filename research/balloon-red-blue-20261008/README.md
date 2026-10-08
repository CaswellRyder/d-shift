# Red/blue balloon migration checkpoint — 2026-10-08

Latest experiment: [combined retraining and twelve-scene development diagnostic](COMBINED_REFINEMENT.md).
The added hard-negative candidate regresses and is not promoted. Sections below
retain the initial migration checkpoint rather than replacing its historical evidence.

## Implemented

- Additive `balloon_red_blue` profile; red hue wraparound and blue masks.
- Existing 12-candidate limit, per-color selection and connected-component hole rejection.
- Static/detail/temporal paths reject incompatible model/profile combinations.
- Webcam service uses the corrected profile when supplied with red/blue model metadata;
  goal search and old green/purple artifacts remain unchanged.
- No trained red/blue model, new Pi deployment, or competition qualification yet.

## Public real-photo bootstrap

AI assistant reviewed the four existing Matterport crop contact sheets. Labels
are in `configs/balloon-red-blue-public-review.json`. Only upstream **train**
images were selected. Ambiguous colors and many mixed/occluded crops are excluded;
unlisted IDs are NOT background. Labels are not human-certified. These are
generic balloon photos, not DTR camera scenes.

Local generated dataset: `data/balloon-red-blue-seed-20261008/manifest.json`.
Its SHA-256 is `70fe56a37ac6786baba70bd3c9be2ce9c3df9e1aaabeaafa2e83a86463f912aa`.

| Class | Training crops |
|---|---:|
| red_balloon | 24 |
| blue_balloon | 18 |
| background (other-color balloons) | 72 |

114 crops from 49 distinct source images. This is **training only**: no new
validation/test split or accuracy score. Ordinary training entrypoints that
require train/val/test correctly reject this incomplete dataset. Source/crop
hashes and review provenance are retained. Dataset images stay local in ignored
`data/`; redistribution is not approved. The review, builder, and audit are committed.

Rebuild to a fresh output directory:

```sh
.venv/bin/python scripts/build_public_balloon.py \
  --config configs/balloon-red-blue.json \
  --review configs/balloon-red-blue-public-review.json \
  --training-only --output data/balloon-red-blue-seed-20261008

.venv/bin/python scripts/audit_balloon_seed.py \
  --manifest data/balloon-red-blue-seed-20261008/manifest.json \
  --output research/balloon-red-blue-20261008/proposal-audit.json
```

Both commands refuse to overwrite existing outputs.

## Full-image proposal audit

`proposal-audit.json` records the exact manifest/vision hashes and every reviewed
positive target. Search uses full source photos resized to 320×240, 12 candidates,
same-color proposal groups, and bounding-box IoU ≥0.5.

| Reviewed target | Covered / total |
|---|---:|
| Red balloon | 11 / 24 |
| Blue balloon | 7 / 18 |

This reveals missed or poorly localized objects before classification. Crop
training alone cannot fix targets that search fails to propose. The audit is on
selected **training** photos with incomplete scene annotations; it measures no
precision, trained-model accuracy, held-out generalization, or competition range.
Same-color proposal agreement is not a classified identity.

## Next experimental work

1. Diagnose low-IoU boxes and candidate-budget misses on these training frames;
   compare bounded proposal alternatives without silently increasing Pi workload.
2. Add red/blue non-balloon negatives (clothing, signs, reflections) plus
   orange/yellow goals; other-color balloons alone are insufficient negatives.
3. Review independent validation recordings/photos and preserve a separate final
   test set before selecting a trained candidate. Source-image separation is not
   proof of recording-session independence.
4. Train a correctly labeled teacher/candidate, then distill/export with explicit
   class metadata. Never relabel the green/purple logits.
5. Benchmark each method separately on the Pi when reachable; no ESP32/motor work.

The bounded SSH check this turn returned `No route to host`. No Pi timing,
camera, flight or deployment claim is made.
