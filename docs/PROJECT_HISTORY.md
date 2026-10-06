# DTR vision project: from MobileNetV4 to the optimized Pi Zero runtime

**History through October 6, 2026.** This document consolidates the project conversation, implementation records, experiment reports, and retained artifacts. Dates below come from recorded runs where available; the early design discussion is described in order rather than assigned invented dates.

This is a history of development, not a claim that the autonomous blimp is finished. Earlier reports describe what was true at their particular milestone; their statements such as “no distillation yet” have since been superseded.

## 1. Where we stand now

We started with the idea of using MobileNetV4 and now have an executable, measured hybrid vision pipeline on the **original Raspberry Pi Zero W**:

```text
OFFBOARD TRAINING — Mac
Public ImageNet MobileNetV4-Conv-Small weights
    → native Keras implementation and numerical checks
    → DTR-specialized balloon and goal teachers
    → knowledge distillation into small custom crop CNNs
    → FP32 and full-INT8 TensorFlow Lite exports

PI PERCEPTION — tested components
Image → OpenCV color/contour proposals → tiny crop classifier
      → bounded temporal tracking and fresh observations
      → boxes, labels, scores, shape/color evidence, track state

LATEST RUNTIME CANDIDATE
Same context-student weights + rebuilt ARMv6 TensorFlow Lite C library
    → about 9.03 processing FPS on the controlled replay
    → not yet a measured live-camera rate for this combination
```

The Pi model is **not the full MobileNetV4**. MobileNetV4 is the offboard teacher. Our stronger goal student is a custom **16,183-parameter context/layout CNN**, with 64×64 RGB input and a 68,064-byte FP32 export.

The latest major improvement came from rebuilding the inference runtime, not retraining the model: context-crop latency fell from **63.82 to 23.64 ms**, and matched whole-pipeline replay throughput rose from **5.42 to 9.03 FPS**. The installed Pi runtime and original comparison implementation were preserved.

Still unfinished: independent arena accuracy validation, live-camera qualification of the newest combination, calibrated distance, confirmed balloon possession, ESP32 integration, and flight behavior.

## 2. The mission and actual hardware

The intended mission is:

1. Take off.
2. Find a balloon of our assigned team color.
3. Approach and capture it.
4. Find a suitable goal of our assigned color.
5. Carry the balloon through the goal.
6. Land.

The vision classes are green/purple balloons and orange/yellow circle, square, and triangle goals. Each task also needs background rejection.

The board was initially described as a Zero 2 W, but device inspection established **Raspberry Pi Zero W Rev 1.1, `armv6l`**. That correction fundamentally changed the deployment expectations. Benchmarks for newer Raspberry Pis or modern mobile processors do not establish performance on this board.

Early setup showed a USB management connection without a default Internet route or DNS configuration. University Wi-Fi registration and DHCP delays were separate from the vision work. The camera initially reported no available cameras; later actual OV5647 capture tests succeeded. These are different points in the project history, not contradictory current diagnoses.

Training belongs on the Mac. The Pi handles constrained perception. The ESP32 is intended to handle its hardware/control responsibilities, but a completed Pi-to-ESP32 perception/control contract is not established by the work below.

## 3. Why we chose MobileNetV4

We explored OpenCV, YOLO-style detectors, MobileNet versions, distillation, and quantization. The user wanted a concrete baseline rather than spending the available time trying every architecture.

The selected approach was **MobileNetV4-Conv-Small in Keras**, trained offboard and used to teach much smaller models. This preserved the ability to build a specialized learned recognizer without assuming that a full modern backbone would run acceptably on ARMv6.

The professor's suggestion about “ROLO” or a public pretrained DTR model prompted source investigation. No verified professor-provided teacher checkpoint became the basis of this pipeline. The implemented teacher instead starts from public ImageNet weights and is specialized with labeled DTR crops.

Importantly, this was a provisional engineering choice—not proof that V4 universally beats V3, or that distillation will automatically outperform a well-calibrated pixel method.

Sources: [public sources](DATA_SOURCES.md), [initial verification](VERIFICATION.md), [student research](STUDENT_RESEARCH.md).

## 4. September 28–29: building the training and inference foundation

We implemented a native Keras MobileNetV4-Conv-Small and imported the public `timm/mobilenetv4_conv_small.e2400_r224_in1k` ImageNet checkpoint. The conversion mapped **93 layers**. Three deterministic inputs passed numerical comparison at 96×96; maximum logit error was approximately **0.0000687**, below the 0.0005 tolerance.

The initial verified Mac environment used Python 3.11.15, TensorFlow 2.18.1, Keras 3.8.0, and TensorFlow Metal 1.2.0. This is the recorded project environment, not a claim that the earlier Python 3.14/Conda setup automatically had compatible dependencies. Early notebook errors such as missing `time` or `cv2` were setup issues, not model failures.

The foundation included:

- Separate balloon and goal configurations, class orders, manifests, and models.
- Teacher head training followed by backbone fine-tuning.
- Frozen-teacher distillation and small student architectures.
- Full-integer export, runtime preprocessing, checksums, and metadata.
- Bounded OpenCV proposals and observational replay.
- Training reports, confusion matrices, TensorBoard output, and regression tests.
- Research opt-in and `deployment_approved: false` metadata.

The first synthetic runs exercised the entire pipeline. They were **software smoke tests**, not DTR accuracy evidence: the one-epoch goal student was effectively at chance. A converter failure involving resource variables was fixed by freezing variables before conversion; the failed run was retained rather than relabeled successful.

On September 29, we separated **teacher development** from automatic distillation. `train-teacher` could train, pause, resume, and select by validation loss without immediately exporting a student or opening the reserved test split. Epoch checkpoints preserved optimizer state and guarded against incompatible resume inputs.

Sources: [verification](VERIFICATION.md), [training](TRAINING.md).

## 5. Building a visual testing workflow

The user needed to see what the camera pipeline was doing, not just receive printed classifications. We built a browser-based testing interface around `webcam_app.py` with webcam and uploaded-image analysis, candidate boxes, masks, labels, scores, display tracking, and timing information.

The interface later gained:

- Private laptop access through the documented `pacman` HTTPS endpoint.
- Downloadable diagnostic images for sharing and review.
- Explicit model/task identification and candidate budgets.
- Explanations for duplicate suppression and tentative goal evidence.
- An aspect-preserving **2592×1944 source ceiling**.
- A selectable processing-FPS cap, default 5, without a growing frame queue.
- Source dimensions, camera-reported FPS, and delivered-result FPS displayed separately.

The 2592×1944 ceiling does **not** mean the classifier searches a full 5-megapixel image. The current viewer then letterboxes the source into **320×240** for proposals and extracts model-sized crops. Smaller sources are not upscaled to invent detail.

The Mac viewer and Pi benchmark are distinct paths. The viewer retains its selected teachers and simple display tracker; changing its FPS cap does not simulate Pi latency, optics, exposure, thermal behavior, or the Pi temporal scheduler. Image downloads are not automatic labeled-data collection. Creating an export also does not prove a separate laptop checkout was synchronized.

Source: [testing and viewer instructions](TESTING.md).

## 6. October 3: resolving the real-data blocker

### 6.1 Initial public-data attempts

Before authorized DTR export access, we inspected public sources instead of treating generated images as competition data.

- **Matterport balloon bootstrap:** 74 photographs, 305 upstream balloon regions; assistant visual review produced 191 selected crops from 69 photographs. Training used 158 crops. Validation was only 15 crops: 14 correct overall, but just **one of four target crops** correctly accepted at threshold 0.8. This was not a usable capture policy.
- **IU public photographs:** seven large yellow goals were annotated in three same-scene images. This draft stayed quarantined because of insufficient classes/session diversity and unresolved reuse terms.
- **Lehigh FOMO demonstration footage:** rejected for training because detection graphics were already burned into the imagery.
- **Roboflow:** public metadata was discoverable, but the tested export initially required authorized access.

These attempts clarified that publicly visible examples, a hosted model demo, downloadable weights, and a properly licensed labeled dataset are different things.

### 6.2 Authorized DTR acquisition

The user supplied authorized access to Cheese's **Cats-and-Dogs** Roboflow project. Despite its name, the exports contain the relevant DTR balloons and goals. We downloaded V10 and V11, retaining original labels, license files, and archive checksums. Credentials are not included in this history.

**V10 was selected:** 640×640 images versus V11's 180×180. Both exports had 6,180 images and shared the same 6,054 original filename stems; combining them would duplicate examples.

The selected source is recorded as **CC BY 4.0**. Derived datasets require attribution to Cheese / Cats-and-Dogs / Roboflow Universe.

### 6.3 Split and label hygiene

We replaced upstream frame-level splits with filename-family/time-block grouping, a temporal guard, duplicate removal, conflict exclusion, and cross-split near-duplicate checks.

| Retained split | Frames | Role |
| --- | ---: | --- |
| Training | 2,951 | Fit models and derive training examples |
| Development validation | 595 | Model and implementation selection |
| Reserved test | 1,946 | Kept out of the reported real-data development experiments |
| Total | 5,492 | After 688 exclusions |

These are **inferred groups**, not verified independent recording sessions. Representative crops were visually reviewed, not every annotation. Generic balloon labels were not silently converted into team colors. Missing annotations remained a known risk for background mining. Later reviewed adaptations excluded an additional conflicting training source, explaining later 2,950-frame counts.

Sources: [real-data acquisition and history](REAL_DATA_STATUS.md), [source and licensing notes](DATA_SOURCES.md).

## 7. Real MobileNetV4 teachers: strong crops, weak initial detection

Teachers used RGB 96×96 input, a newly trained task head, then low-learning-rate fine-tuning with frozen BatchNorm. The starting schedule allowed five head epochs and up to fifteen fine-tuning epochs, with early stopping and validation-loss selection.

| Teacher | Validation crops | Correct | Crop accuracy |
| --- | ---: | ---: | ---: |
| Balloon | 1,178 | 1,177 | 99.915% |
| Goal | 1,773 | 1,710 | 96.447% |

Those results did **not** mean the camera could detect nearly every target. A classifier sees only crops that the proposal stage gives it.

The original 320×240, three-candidate proposal audit covered only 49.2% of green balloons, 21.5% of purple balloons, and essentially none of several orange-goal classes at IoU ≥0.5. Yellow squares were much easier. This established the central recurring bottleneck: **finding a useful box, not merely classifying an already-cropped object**.

Sources: [real-data results](REAL_DATA_STATUS.md), [teacher training](TRAINING.md).

## 8. October 5: adapting to real proposals and reducing false positives

We trained teachers on examples closer to the actual OpenCV proposals. Raw mined backgrounds were quarantined because some supposedly negative regions contained unlabeled targets.

Assistant visual review admitted 161 balloon-task and 176 goal-task negative crops from 256 candidates reviewed per task. These were oversampled, not misrepresented as independent new photographs. Original validation/test crop records remained unchanged.

The following are full-frame development results on the same 595 frames, not crop accuracy:

| Pipeline stage | Balloon precision / recall | Goal precision / recall |
| --- | --- | --- |
| Proposal-adapted teachers | 42.6% / 81.8% | 48.0% / 54.8% |
| With nested same-class duplicate suppression | 44.4% / 81.8% | 72.3% / 54.6% |
| With selected balloon component filtering | 77.8% / 81.8% | 72.3% / 54.6% |

The component fix stopped treating holes/reflections inside a balloon component as extra objects while preserving foreground islands inside hoops. Balloon false positives fell from **663 to 151**, retaining 530 true detections. Plain external-contour filtering was rejected because it also removed real objects.

The selected viewer configuration became `balloon_components`, twelve proposals per task, threshold 0.8, and nested duplicate suppression. This was a concrete improvement without retraining another architecture.

Sources: [testing results](TESTING.md), [component filtering](COMPONENT_PROGRESS.md).

## 9. Orange goals: requirements correction and localization experiments

The user initially suggested yellow-only goals might be sufficient, then corrected the requirement: **both neon orange and neon yellow are necessary**.

A stage-level audit separated missing boxes, budget losses, wrong classes, confidence rejection, background predictions, and suppression. Among 325 orange localization failures, **318 had a longest side below sixteen pixels** at 320×240.

We retained unsuccessful experiments instead of hiding them:

| Attempt | Result and decision |
| --- | --- |
| Wider/consolidated orange masks and local contrast | Failed to provide a reliable same-budget improvement; not selected |
| Raise goal budget from 12 to 24 | Recall 54.6→59.4%, but precision 72.3→62.0% and F1 62.2→60.7%; keep 12 |
| `goal_gap9` morphological closing | Goal F1 62.2→56.8%; strong yellow-triangle regression; rejected |
| Unchanged pixel thresholds at 640×480 | Worse sampled orange proposal coverage; not proof that all high-resolution strategies fail |
| Small-orange MSER/chroma regions | More circles, fewer triangles; total goal F1 62.2→61.1%; rejected |

This led to a stricter train-side regression screen covering each color, shape, and size stratum before spending further validation effort. Aggregate gains were no longer enough to hide a class-specific regression.

Sources: [goal-stage audit](GOAL_STAGE_PROGRESS.md), [orange regions](ORANGE_REGION_PROGRESS.md), [component experiments](COMPONENT_PROGRESS.md).

## 10. The separate YOLO11n detector branch

Because proposal localization was weak, we also investigated an offboard **YOLO11n goal detector** that directly predicts boxes. This was a separate experiment—not a MobileNetV4 rename, not the Pi student, and not evidence that YOLO ran on the Zero.

Work included detector dataset preparation, label review, fixed per-class evaluation, checkpoint snapshots, and robust resume handling. Disk exhaustion and Mac training-memory growth interrupted development; disk guards and epoch-boundary process restarts made continuation auditable. Later resume checks verified actual checkpoint transfer, optimizer state, and recorded RNG state rather than silently starting again from generic pretrained weights.

The baseline completed at epoch 21, with epoch 17 the stronger retained baseline under the selected criterion. A twelve-epoch reviewed-label refinement selected epoch 2 as its strongest complete checkpoint under the weakest-class ranking.

### The table previously discussed for the report

These are that **offboard detector's** development results, not the tiny Pi classifier's results:

| Goal class | Precision | Recall | Both ≥90%? |
| --- | ---: | ---: | --- |
| Orange circle | 94.24% | 91.60% | Yes |
| Orange square | 92.92% | 96.57% | Yes |
| Orange triangle | 86.88% | 95.52% | No |
| Yellow circle | 96.84% | 96.23% | Yes |
| Yellow square | 96.20% | 89.41% | No |
| Yellow triangle | 98.33% | 99.16% | Yes |

The ≥90% standard was a project evaluation criterion, not a claimed competition rule. No checkpoint passed all six classes. Later epochs and a 960-input trial did not solve this.

A close-range exposure experiment repeated 349 selected training frames without creating new independent scenes. Its first checkpoint improved yellow-square recall to 91.37% but reduced orange-triangle precision to 77.11% and yellow-circle precision to 85.00%. Epoch 2 was saved; the experiment was paused to prioritize Pi integration. It was not promoted.

Sources: [detector history](DETECTOR_PROGRESS.md), [refinement](DETECTOR_REFINEMENT.md), [error review](GOAL_ERROR_FOLLOWUP.md), [close-range exposure](DETECTOR_CLOSE_EXPOSURE.md).

## 11. Distillation: the actual small models

To get something running on the real Pi, we distilled the frozen specialized teachers into custom **64×64 RGB crop CNNs**.

The original student has three stride-two convolution layers with 8/16/32 channels, global average pooling, and class logits. Balloon has background/green/purple outputs; goal has background plus six color-shape outputs.

The distillation objective combines:

```text
0.5 × hard-label cross-entropy
  + 0.5 × T² × KL(teacher softmax at T || student softmax at T)
T = 4
```

Teacher logits are cached for identical crops, guarded by teacher/manifest hashes and class order. Augmenting a crop while reusing unrelated cached targets would break that correspondence, so these recorded student runs used no such augmentation. Validation hard-label loss selects checkpoints.

### First eight-epoch students

Adam 0.001, batch 32, equal hard/soft weighting:

| Student | Parameters | INT8 file | FP32 crop accuracy | INT8 crop accuracy |
| --- | ---: | ---: | ---: | ---: |
| Balloon | 6,131 | 10,640 bytes | 99.41% | 99.41% |
| Goal | 6,263 | 10,880 bytes | 80.60% | 80.49% |

The balloon classifier looked strong on crops but still produced full-frame false positives. Goal accuracy required more work. The offboard YOLO precision/recall table did not transfer to either student.

Source: [Pi comparison](PI_COMPARISON.md).

## 12. Quantization: implemented, but not the fastest choice

The full-integer export uses representative **training-only** crops—500 for the original real-data students—deterministically sampled. Conversion requests `TFLITE_BUILTINS_INT8`, with INT8 input and output. The exporter checks for remaining floating-point tensors and records quantization scales, zero points, class order, preprocessing, checksums, and provenance.

At runtime, inputs follow that metadata contract; output logits are dequantized and converted to scores. This is post-training full-integer quantization, not a completed quantization-aware-training campaign.

We also exported FP32 and measured both on the actual Pi. **INT8 made files smaller but was slower for these models on the tested ARMv6 runtimes.** The original student measured about 40.31 ms/crop FP32 versus 54.68 ms INT8. The later rebuilt runtime preserved the same direction of result.

Therefore the current performance-oriented path is FP32. Quantization remains a supported, verified artifact option—not a mandatory speed improvement.

Sources: [export implementation](../src/dtr/export.py), [Pi optimization](PI_OPTIMIZATION.md), [runtime rebuild](TFLITE_REBUILD.md).

## 13. First actual-Pi comparison: executable, but about 10× slower

We preserved the existing pixel implementation at `/home/pacman/Documents/TESTING/demo.py` and built isolated comparison bundles.

The original method used RGB-distance lookup and connected components, configured for yellow objects. It did not classify goal shapes or orange goals. The alternative used OpenCV proposals and the distilled student.

The native adapter called the already installed TensorFlow Lite 2.20.0 **C API**, avoiding dependence on a Python TensorFlow wheel for ARMv6. Golden crops checked Mac/Pi output agreement before reporting timings.

Same 595 development frames, common 320×240 images, yellow localization ignoring shape:

| Metric | Existing pixel baseline | Initial INT8 student pipeline |
| --- | ---: | ---: |
| Precision | 7.87% | 38.39% |
| Recall | 7.69% | 58.91% |
| F1 | 7.78% | 46.48% |
| Mean processing | 71.32 ms | 761.43 ms |
| Processing throughput | 14.02 FPS | 1.31 FPS |

The alternative was approximately **10.7× slower**. The baseline's original color and minimum-area settings were not recalibrated for these public images, so this was a comparison of frozen configurations—not proof that learned vision universally beats pixel analysis.

A twenty-frame actual-camera smoke test also saved same-frame image pairs. It proved capture/inference plumbing, not arena accuracy; false positives were visible in the desk scene.

Source: [original comparison and protocol](PI_COMPARISON.md).

## 14. Making the slow-blimp pipeline practical

We then made three complementary changes:

1. Trained the original goal student for 25 epochs; selected epoch 18. FP32 crop accuracy increased from 80.60% to **93.97%**.
2. Used the measured-faster FP32 runtime path and removed unnecessary contour work while checking proposal equivalence.
3. Added `TemporalVision` to update position between bounded fresh classifications.

The selected experimental temporal policy used a four-crop budget, full search about every 0.5 seconds, class refresh about every 0.6 seconds, and a one-second accepted-label lifetime. High-confidence background could refresh at 1.8 seconds. Scene changes, failed association, missing targets, frame gaps, and resolution changes invalidate relevant state.

This exploits slow motion without pretending that a cached identity is a fresh neural result. Association remains heuristic.

### Controlled generated replay

Twelve development stills were translated into twenty frames each, with inserted black missing-target frames:

| Method | Processing FPS | Yellow precision / recall |
| --- | ---: | --- |
| Original pixel baseline | 13.01 | 0.00% / 0.00% on this selected subset |
| New FP32 student, stateless | 1.78 | 74.80% / 78.63% |
| Temporal, four crops, uniform refresh | 5.42 | 78.45% / 71.58% |
| Temporal with slower background refresh | 5.92 | 78.45% / 71.58% |

The selected temporal policy cut neural calls from 2,592 to 496, but sacrificed recall. Its zero baseline matches must not be hidden or generalized beyond this uncalibrated subset.

### Actual live camera at this milestone

On sixty actual 320×240 OV5647 frames, the selected older student/policy measured **2.85 FPS full-loop**, versus 3.22 FPS processing-only. No accepted goals were present in these unlabeled frames. This is camera/runtime throughput evidence, not detection-accuracy evidence.

The 5.92 replay FPS was never a valid substitute for this live measurement. Later context-model speed estimates around 4–5 FPS were projections, subsequently superseded by direct replay measurements—not by a new live-camera benchmark.

Source: [optimization, replay, and live-camera results](PI_OPTIMIZATION.md).

## 15. Better student architecture: preserve shape context

We compared tiny, spatial-layout, and context students using the same reviewed data and fixed settings. The context variant adds a 32-channel 3×3 convolution and retains a 2×2 spatial layout before classification, rather than collapsing everything immediately into global averages.

| Candidate | Parameters | Crop accuracy | Six-class full-frame F1 |
| --- | ---: | ---: | ---: |
| Earlier 25-epoch student | 6,263 | 93.97% | 51.57% |
| Spatial-layout distilled | 6,935 | 95.71% | 57.22% |
| Context supervised | 16,183 | 96.90% | 60.45% |
| Context distilled | 16,183 | 98.31% | 63.23% |

The controlled tiny reruns were weaker than the earlier incumbent, illustrating initialization/training sensitivity. Distillation helped the context model in this batch; it did not improve every architecture or every class universally.

The context model has approximately 1.73× the original convolution/dense MAC count. That is a computation estimate, not a measured latency multiplier.

### What the oracle and resolution tests taught us

Only **735 of 1,188** goal annotations had a useful top-twelve proposal. For 284 goals under eight pixels, only fifteen had coverage, although the context model could correctly identify 273 when supplied their annotated boxes. This is conditional classification with oracle boxes, not operational range proof.

An opt-in paired-resolution path searched at 320×240 and classified crops from a larger same-frame image. Context F1 improved only 63.23→63.77%; the older student regressed. More pixels alone did not solve missing proposals. This path was not automatically connected to the camera/viewer.

Source: [student architecture and localization research](STUDENT_RESEARCH.md).

## 16. Separating shape recognition from uncertain color

The user reported seeing shapes at approximately 95 feet while color remained uncertain. We treated that as an observation to reproduce, not a measured operating-range claim.

We added shape/color evidence derived from the existing seven-class scores. For example, orange-circle and yellow-circle scores can support a strong circle hypothesis while leaving color unknown. This requires no additional neural invocation.

The crop viewer displays a **purple tentative box** for strong shape/unknown color. It does not silently label “not yellow” as orange. Original joint-class acceptance and expiry remain intact.

An opt-in observational fallback was tested: it added **eleven false positives and zero true positives** on the development audit. It therefore remains disabled for autonomous pursuit. Unknown is useful information, not permission to assume our color.

This also cannot recover a goal omitted by the color-based proposal stage. A separate shape-only head or color-independent localizer would be a different experiment.

Source: [goal uncertainty](GOAL_UNCERTAINTY.md).

## 17. Smaller context models and rejected shortcuts

We replaced the final dense context convolution with depthwise/pointwise operations, producing an **8,279-parameter separable student**. A warm initializer factorized the trained context kernel with rank-one SVD, copied compatible weights, and then distilled again.

SVD alone was inadequate: 77.50% relative kernel error and 49.97% crop accuracy. Fine-tuning mattered. The warm run also inherited prior training, so it was not an equal-total-training-budget comparison with scratch training.

| Candidate | Full-frame F1 | Old-runtime Pi crop latency | Controlled replay FPS |
| --- | ---: | ---: | ---: |
| Context | 63.23% | 64.50 ms | 5.17 |
| Scratch separable | 58.00% | 43.84 ms | Not in this paired replay |
| Warm separable | 62.79% | 43.65 ms | 6.29 |

Warm separable was faster but introduced important square errors: orange-square false positives rose from 28 to 150. Its replay F1 also fell, 63.48→59.34%. We retained it as a speed experiment instead of replacing the stronger context reference.

A grayscale-edge proposal fallback also failed: proposal coverage fell 735→611 and full-frame F1 63.23→54.40%. It stayed disabled.

Source: [CV progress](CV_PROGRESS.md).

## 18. October 6: targeted negatives and lower-overhead tracking

### Hard-negative refinement

We mined 2,950 training frames, producing 476 candidate negatives and a 96-crop review queue. Assistant review admitted fifty clear backgrounds; ambiguous/unannotated goals were excluded. Eight refinement epochs compared added negatives against a no-added-negative control.

Orange-square false positives fell 150→95, but other classes worsened. Overall F1 fell from the warm model's 62.79% to **62.38%**. The control scored 61.55%. Matching epochs did not match optimizer updates because the added-negative dataset was larger. Neither candidate replaced the reference.

No image-generation training samples were added in this round. The user authorized exploration, but targeted real-error review was the work actually performed.

### Exact RGB lookup: correct, but slower

A 16 MiB lookup matched the HSV policy across all 16,777,216 RGB values and reproduced proposal lists on all 595 frames. On the Pi it nevertheless increased search time **112.01→120.72 ms**, about 7.77% slower. Default remained HSV.

### Native tracking difference: retained improvement

Replacing repeated NumPy absolute-difference reductions with equivalent `cv2.norm(..., NORM_L1)` operations preserved fixed-clock replay outputs on both Mac and Pi. Two paired context-model runs improved throughput by **3.6% and 4.6%**, approximately 5.2→5.4 FPS. Tail latency did not improve consistently.

The native difference path became the local default; frozen comparison bundles and the original Pi demo remained preserved.

Source: [CV refinement](CV_REFINEMENT.md).

## 19. TensorFlow Lite rebuild: what finally happened

Until this stage, we had improved models, scheduling, and the C API adapter while using the existing installed TensorFlow Lite library. That was **not** a TensorFlow rebuild. After the user explicitly asked to proceed, we attempted and completed an isolated ARMv6 rebuild.

### Build approach and problems solved

- Pinned TensorFlow **2.20.0**, commit `72fbba3d20f4616d7312b5e2b7f79daf6e82f2fa`.
- Cross-compiled on the Mac using Zig 0.14.1/Clang 19.1.7, CMake 3.31.6, and Ninja 1.11.1.3.
- Targeted ARM1176JZF-S / ARMv6 / VFPv2 hard-float with `-O3` and without fast-math.
- Kept build tools and sources in an isolated external-drive tree.
- Disabled unsupported/unneeded delegate paths; retained required CPU resource support.

Two isolated Linux-container builders failed to start. We switched to native Mac cross-compilation and first proved a small ARMv6 C++ executable ran on the Pi.

Build repairs addressed an absent XNNPACK CMake target, missing version defines, required FP16 headers, and an unresolved resource-variable symbol. The final link restored resource support and enforced no undefined symbols. No inference kernels or model weights were rewritten.

The resulting C API library is **3,361,916 bytes**. It loads on the actual Pi and reports 2.20.0. The old library already had ARMv6/VFPv2 attributes; this was not fixing an accidentally installed ARMv7 binary. Compiler/runtime build differences are confounded, so we do not attribute the entire speedup to one compiler flag.

### Same-device crop measurements

Installed/rebuilt/rebuilt/installed order; two runs per runtime, 63 timed calls per model per run, one inference thread:

| Model | Installed ms/crop | Rebuilt ms/crop | Throughput ratio |
| --- | ---: | ---: | ---: |
| Context FP32 | 63.82 | 23.64 | 2.70× |
| Context INT8 | 85.77 | 51.13 | 1.68× |
| Scratch separable FP32 | 43.61 | 18.15 | 2.40× |
| Scratch separable INT8 | 61.92 | 38.23 | 1.62× |
| Warm separable FP32 | 43.47 | 17.68 | 2.46× |
| Warm separable INT8 | 60.34 | 38.49 | 1.57× |

Every reference label matched across all four runs. Maximum rebuilt FP32 score error was 9.54e-7; INT8 error was 0.00522, within its declared tolerance and also seen with the installed library. Outputs are not bit-identical. FP32 remains faster.

### Whole context-pipeline replay

Same context weights, native tracking differences, four-crop budget, proposal policy, and expiry rules:

| Runtime | Mean processing, two runs | Processing FPS, two runs | p95, two runs |
| --- | --- | --- | --- |
| Installed | 185.06 / 184.12 ms | 5.40 / 5.43 | 411.96 / 418.66 ms |
| Rebuilt | 110.38 / 111.02 ms | 9.06 / 9.01 | 258.29 / 259.42 ms |

Aggregate throughput improved approximately **67%**, from 5.42 to 9.03 FPS—not 2.7× for the entire pipeline. Search and tracking still consume time.

All runs made 439 neural calls and 78 scans. Six additional true-positive observations appeared with the faster runtime because cached classifications stayed below the unchanged one-second expiry. This was not a learned-accuracy improvement.

The rebuilt runtime still exceeded 100 ms on 120–122 of 240 frames, so there is **no guaranteed 10 Hz deadline**. These rates exclude camera capture, decoding/generation, rendering, transport, startup, and output writes.

Whole-process peak RSS in the recorded second runs was 110.8 MiB installed versus 108.3 MiB rebuilt. That includes imports and replay records, not just the library. Both recorded zero major page faults. Final board temperature was 37.9°C with throttling flags `0x0`.

### What was and was not deployed

The candidate resides separately on the Pi at:

```text
/home/pacman/dtr-runtime-build-20261006/libtensorflowlite_c.candidate.so
```

It is selected per process using `DTR_TFLITE_LIBRARY`. The system library was **not replaced**, no startup service was changed, the original demo checksum remained unchanged, and all 640 frozen comparison files still matched.

The stronger context model can now be tested faster without accepting the separable model's square regressions. A live-camera trial of this newest combination remains the next integration step.

Sources: [complete rebuild report](TFLITE_REBUILD.md), [packaged runtime and receipts](../output/tflite-armv6-20261006.zip), [build scripts](../scripts/tflite_rebuild/).

## 20. Navigation ideas: recorded requirements, not completed autonomy

These discussions shaped the intended system but must not be described as working flight features:

| Requirement or idea | Status at this endpoint |
| --- | --- |
| Estimate target bearing and distance | Image-space observations exist; calibrated metric range is not established |
| Choose the cheapest balloon/goal to reach | Planning requirement; needs geometry, uncertainty, and a defined cost |
| Pursue a goal only while carrying a balloon | Required mission-state gate; possession confirmation unresolved |
| Detect capture with current camera/IMU/cone-vacuum hardware | Hardware geometry/visibility unanswered; no proven capture detector |
| Prefer a central shape/“square merchant” strategy | Discussed heuristic, not a verified arena-optimal policy |
| Use opposing goals when our goal is lost | Written bounded-recovery requirement, not implemented |
| Send useful observations to the ESP32 | End-to-end serial protocol, age handling, and controller behavior not validated |
| Takeoff, traverse, deliver, land | Not established by CV benchmarks |

The opposing-goal idea is to use fresh, positively identified opponent goals to bias a bounded search away from them. It is not proof that our goal is exactly 180° away or that the route is clear. Unknown color must not become an opponent identification by default.

An IMU alone does not establish balloon possession. A goal bounding box also does not establish the traversable opening or safe clearance for the blimp and captured balloon.

Source: [navigation implementation note](NAVIGATION_NOTES.md).

## 21. What the numbers mean—and what they do not

| Evidence type | Answers | Does not establish |
| --- | --- | --- |
| Synthetic smoke test | Does the software execute? | Real arena accuracy |
| Crop accuracy | Can the model classify supplied crops? | Can the proposal stage find targets? |
| Full-frame development P/R/F1 | How does this configuration match annotations? | Independent-session generalization |
| Oracle crop test | How well does recognition work with supplied true boxes? | Operational detection range |
| Golden-crop parity | Do tested exports/runtimes agree within tolerance? | Equivalence for every possible input |
| Controlled replay FPS | How fast is the measured processing workload? | Live-camera or controller throughput |
| Unlabeled camera smoke test | Does capture-to-observation execute? | Correct detections or flight safety |
| Unit tests/lint | Are specified software contracts checked? | Competition or flight qualification |

The 595-frame development block has been reused extensively. It is useful for controlled comparisons but is no longer an independent final accuracy estimate. The reported real-data work did not evaluate the reserved test split. Earlier synthetic test evaluations are separate and do not contradict that statement.

Latency tables from different stages use different models, workloads, and policies. Only matched comparisons support a numerical speedup claim. In particular, **14.02, 13.01, 5.92, 5.17, and 9.03 FPS are not one uniform benchmark series**, and none should replace the recorded 2.85 live FPS for the older camera configuration.

## 22. Current artifact map

| Area | Location |
| --- | --- |
| Overall setup | [README](../README.md) |
| Native Keras models | [models.py](../src/dtr/models.py) |
| Teacher training | [training guide](TRAINING.md) |
| Real data and exclusions | [data receipt](REAL_DATA_STATUS.md) |
| Original real teachers | `runs/balloon-dtr-v10-20261003/`, `runs/goal-dtr-v10-20261003/` |
| Proposal-adapted teachers | `runs/balloon-proposals-20261005/`, `runs/goal-proposals-20261005/` |
| Stronger context student | `runs/student-research-20261005/context-kd/student.float.tflite` |
| Warm separable experiment | `runs/goal-separable-warm-20261005/` |
| Native TFLite adapter | [native_tflite.py](../src/dtr/native_tflite.py) |
| Temporal scheduler | [temporal.py](../src/dtr/temporal.py) |
| Shape/color evidence | [goal_evidence.py](../src/dtr/goal_evidence.py) |
| Mac visual testing | [webcam_app.py](../webcam_app.py), [testing guide](TESTING.md) |
| Original Pi comparison | [comparison report](PI_COMPARISON.md) |
| Latest runtime/results | `runs/tflite-rebuild-20261006/` |
| Rebuild distribution | [tflite-armv6-20261006.zip](../output/tflite-armv6-20261006.zip) |
| Dataset/model/runtime notices | [third-party notices](../THIRD_PARTY_NOTICES.md) |

Latest recorded verification at the rebuild endpoint: **231 Python tests passed, two skipped; Ruff and build-shell syntax checks passed**. The preceding refinement also recorded seven viewer tests passing. These are retained results, not a newly executed test run for this documentation update.

## 23. What remains, in practical order

1. **Run the context FP32 student with the rebuilt runtime on live Pi camera input.** Measure capture-to-result age, p95, dropped/stale observations, temperature, and memory—not just crop inference.
2. **Collect labeled, representative Pi-camera recordings.** Include distant orange goals, real approach/retreat, rotations, partial occlusion, exposure changes, and distracting backgrounds. Hold out entire new recording sessions.
3. **Improve bounded small-target localization.** The crop recognizer cannot recover absent proposals. Investigate targeted detail searches or a compact learned localizer before assuming another classifier retrain is the answer.
4. **Give the pixel baseline a fair calibration-only tuning pass.** Then freeze it and compare on new shared recordings for accuracy, latency, and power-relevant workload.
5. **Calibrate camera geometry and define observation freshness.** Metric distance needs known geometry or another validated method; direction needs camera/body alignment.
6. **Agree on capture evidence and the ESP32 contract with hardware/design.** Define fields, units, timestamps, confidence/unknown states, stale-data rejection, and watchdog behavior before control integration.
7. **Validate mission-state behavior in supervised stages.** Detection does not itself authorize capture, goal traversal, or flight.

The largest completed performance advance is now available as an isolated runtime candidate. The largest unresolved accuracy issue remains small-goal localization, particularly orange targets. We have moved from an architectural idea to a reproducible, measured Pi perception system—but not yet a finished autonomous competition vehicle.
