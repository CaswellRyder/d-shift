# Orange-goal synthetic pilot — October 7, 2026

User confirmed orange GOALS, not orange balloons. Production target taxonomy is
orange/yellow goals and, per the subsequent rule clarification, RED/BLUE balloons.
The older generated orange and green/purple balloons are research distractors,
not competition-positive balloon samples. Historical balloon models still require
migration; see `docs/TARGET_TAXONOMY.md`.

Built-in image_gen produced two independent scenes, preserved here alongside the
exact prompts in `PROMPTS.md` and AI visually reviewed approximate boxes in
`annotations.json`:

- `backlit-goals.png`: three thin suspended orange goals against ceiling lights.
- `foil-clutter-negative.png`: green/purple foil balloons, packaging, cones, and
  railings; visually reviewed as goal-absent. These balloons are NOT negatives
  for a balloon classifier. Background labels apply only to the goal task.

Both are 1254x1254 generated images. They are not real camera captures, calibrated
physical layouts, independent validation, or exact replicas of the arena. Goal
boxes include visible rim geometry, not suspension cords. Asset coordinates and
class labels were reviewed by the assistant, not human-certified.

## Diagnostic and training experiment

`scripts/build_orange_synthetic_seed.py` preprocesses at 320x240 and builds three
labeled goal crops and six goal-background candidate crops from the negative scene.
Manifest, source hashes and frozen-model diagnostics are in
`data/orange-synthetic-seed-20261007/`. Negative labels come from scene review,
never from the model's own predictions. All descendants remain train-only.

`scripts/train_orange_synthetic_pilot.py` ran a bounded paired experiment from the
same context student Keras checkpoint underlying the deployed FP32 TFLite file.
The original model/runtime hash was verified. Two fixed epochs, learning rate
0.0001, same seed, same architecture, same optimizer-update count. The 7890 original
real training crops retain cached teacher logits for those exact crops. New
synthetic crops use hard labels only, with no fabricated teacher targets.

Each epoch has 72 extra slots (nine synthetic crops repeated eight times).
The control fills those slots with sampled original real crops; the intervention
uses synthetic crops. Both train on 7962 entries, 249 batches/epoch. Source groups
and evaluation partitions are unchanged. Final epoch is fixed; no best-validation
checkpoint selection was performed. This is one seed, not a significance study.

## Results — not a deployment improvement

| Measurement | Untouched model | Real-only control | Synthetic intervention |
|---|---:|---:|---:|
| Real crop-validation top-1 accuracy, 1773 crops | 98.31% | 97.80% | 97.69% |
| Synthetic training negatives accepted as goals, six crops | 4 | 3 | 0 |
| Synthetic training positives accepted correctly, three crops | 3 | 2 | 3 |
| Model parameters | 16183 | 16183 | 16183 |

Synthetic acceptance uses the existing 0.8 threshold; validation accuracy is top-1
classification, not detection precision/recall. The synthetic scores measure fit
to training examples, not generalization. The intervention makes two more crop
validation errors than the matched control and eleven more than the untouched
model. No new model was exported to Pi or selected for deployment.

Evidence: `runs/orange-synthetic-pilot-20261007/{provenance,report,status}.json`,
separate control/synthetic `student.keras` artifacts, and per-epoch CSV logs. Base
weights and manifest hashes were rechecked unchanged after training.

## Next

Broaden independent training-scene diversity and include reviewed real negatives;
do not turn these two scenes into a huge near-duplicate dataset. Before retraining,
separate proposal misses from classifier errors and protect real class-wise recall.
Any selected candidate still needs full-frame development evaluation, temporal
replay, and sequential Pi timing. Same parameter count does not prove same measured
FPS. No ESP32, motor commands, or Pi runtime changes were made.
