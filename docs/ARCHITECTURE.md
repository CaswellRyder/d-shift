# Architecture and repository boundaries

D-SHIFT is the perception system. Mission decisions such as lost-goal search,
opponent-goal navigation cues, capture sequencing, and safe movement requests
belong in downstream autonomy software. Stabilization and actuator watchdogs
belong in controller firmware.

| Layer | Main source | Responsibility |
| --- | --- | --- |
| Models | `src/dtr/models.py` | MobileNetV4 teacher and custom tiny/context student architectures |
| Teacher training | `src/dtr/teacher_training.py` | Fine-tuning and checkpoint handling |
| Distillation | `scripts/distill_pi_student.py` | Reviewed crops, frozen teacher targets, student fitting and export |
| Runtime | `src/dtr/runtime.py`, `src/dtr/native_tflite.py` | Metadata/hash checks, preprocessing, TFLite inference |
| Proposals | `src/dtr/vision.py` | Candidate regions and duplicate suppression |
| Temporal inference | `src/dtr/temporal.py` | Budgeted classification, template association, refresh and expiry |
| Evidence | `src/dtr/goal_evidence.py` | Shape/color marginals and conservative observational policy |
| Viewer | `src/dtr/web.py`, `src/dtr/static/` | Local visual debugging; not the Pi temporal deployment loop |

There are no trainable weights for the HSV proposal thresholds, template tracking,
or scheduler. Their source/configuration is part of reproducibility alongside the
CNN weights. The ARMv6 runtime is an upstream TensorFlow Lite build with scoped
build-system adaptations, not a new neural architecture or rewritten kernel set.

## Preserve paths for v1

Source and scripts are intentionally not moved wholesale: the v1 manifest binds
105 implementation files by path and checksum, and historical commands refer to
them. Documentation/navigation is reorganized without breaking that release.
A future package/module reorganization needs its own compatibility/version plan.

## Experimental branches

Smaller students, edge candidates, RGB lookup, hard-negative refinements, and the
optional YOLO detector remain research experiments. Their existence does not make
them selected defaults. Consult the model card and experiment reports rather than
combining scores or speed measurements from different configurations.
