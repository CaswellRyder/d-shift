# Corrected bootstrap teacher/student — 2026-10-08

**Status: trained/exported, NOT flight ready; 94% end-to-end target unproven.**

Fresh MobileNetV4 head/backbone fine-tuning uses red/blue classes with verified
ImageNet initialization. No legacy logits were renamed. The source-disjoint
bootstrap has 45 training photos, four development-validation photos and 13
reserved-test photos. The two training views per selected object are correlated
views of the same object, not new independent samples.

Training counts: 44 red, 30 blue, 138 background views. Validation: two red,
three blue, three background crops. Reserved test: three red, two blue,
19 background crops, not evaluated during this experiment. All are off-domain
public photos; the small test set cannot qualify competition accuracy.

## Delivered local artifacts

| Artifact | Location | Size |
|---|---|---:|
| Teacher | `runs/balloon-red-blue-teacher-20261008/teacher.keras` | MobileNetV4 |
| Student | `runs/balloon-red-blue-context-20261008/student.keras` | 15,667 parameters |
| INT8 | `runs/balloon-red-blue-context-20261008/student.int8.tflite` | 21,888 bytes |
| FP32 | `runs/balloon-red-blue-context-20261008/student.float32.tflite` | 66,000 bytes |

`training-receipt.json` records hashes, configuration and actual reports. Weights
remain in local ignored `runs/`; this is not a public model/dataset release.

Teacher and both student encodings achieved 8/8 **argmax crop** predictions on
the tiny development set. At the unchanged 0.8 confidence threshold, the teacher
accepted only 3/5 true target crops. Argmax accuracy therefore must not be confused
with usable detector recall. Student threshold behavior and full-frame errors
need separate measurement. No new Pi speed is inferred from desktop exports.

## Search ablations (training photos only)

Same 320×240 search and at most 12 candidate crops. These counts exclude the four
photos assigned to development validation, unlike the earlier 42-object audit.

| Search | Red covered | Blue covered | Desktop mean search ms |
|---|---:|---:|---:|
| Baseline | 9/22 | 5/15 | 1.69 |
| Opening + convex hull | 12/22 | 5/15 | 1.21 |
| Multiple saturation masks | 16/22 | 5/15 | 2.48 |
| MSER color regions | 19/22 | 7/15 | 11.16 |
| Eight baseline + four MSER | 17/22 | 7/15 | 12.37 |

Single-pass desktop timings include native warm-up/order noise and are NOT Pi
benchmarks. Candidate coverage at IoU >=0.5 is not model recall or precision.
The better-coverage MSER approach costs substantially more here; all alternatives
remain research-only and no production/default search changed.

## Reproduction

```sh
.venv/bin/python -m scripts.build_red_blue_bootstrap \
  --output data/balloon-red-blue-bootstrap-20261008
TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=1 .venv/bin/dtr --cpu train-teacher \
  --config configs/balloon-red-blue.json \
  --manifest data/balloon-red-blue-bootstrap-20261008/manifest.json \
  --pretrained artifacts/pretrained/mobilenetv4_conv_small.keras \
  --output runs/balloon-red-blue-teacher-20261008
.venv/bin/python scripts/distill_pi_student.py \
  --config configs/balloon-red-blue.json \
  --manifest data/balloon-red-blue-bootstrap-20261008/manifest.json \
  --teacher runs/balloon-red-blue-teacher-20261008/teacher.keras \
  --output runs/balloon-red-blue-context-20261008 --epochs 25 --student-variant context
.venv/bin/python scripts/export_pi_float.py \
  --run runs/balloon-red-blue-context-20261008 \
  --output runs/balloon-red-blue-context-20261008/student.float32.tflite
```

Use fresh output directories; checkpoints and existing artifacts are preserved.
Next: review non-balloon hard negatives, refine the candidate, measure thresholded
full-frame behavior, and obtain representative independent Pi recordings.
