# Third-party notices

MobileNetV4 architecture:
Qin et al., *MobileNetV4 — Universal Models for the Mobile Ecosystem* (2024).
https://arxiv.org/abs/2404.10518

Architecture/block layout and interoperability references (Apache License 2.0):

- TensorFlow Model Garden, Copyright The TensorFlow Authors:
  https://github.com/tensorflow/models/blob/master/official/vision/modeling/backbones/mobilenet.py
  https://github.com/tensorflow/models/blob/master/official/vision/modeling/layers/nn_blocks.py
- timm / PyTorch Image Models, Copyright Ross Wightman and contributors:
  https://github.com/huggingface/pytorch-image-models/tree/v1.0.15

This project's `models.py` is a native Keras implementation of the published Conv-Small
topology, matching timm's symmetric padding, normalization, and naming for weight transfer.
It is not the official Model Garden implementation and does not require legacy tf_keras.
`pretrained.py` converts the timm ImageNet weights, then checks output parity before saving.

Apache-2.0 text: https://www.apache.org/licenses/LICENSE-2.0
The model implementation and interoperability code in this repository are provided under
Apache-2.0. See LICENSE for the full license text. Dataset rights are separate.

The local (git-ignored) DTR dataset derives from **Cats-and-Dogs by Cheese**, Roboflow Universe,
versions 10 and 11: https://universe.roboflow.com/cheese-geozd/cats-and-dogs-bmity .
Their downloaded README and COCO license entries specify **CC BY 4.0**:
https://creativecommons.org/licenses/by/4.0/ . Version 10 supplies the current training crops.
Modifications: crop extraction, background-region sampling, removal of duplicate/conflicting
records, inferred recording-group/time-block splits and temporal exclusions. Original archives,
READMEs and receipts are retained under `data/raw/roboflow-dtr-v{10,11}/` locally. Data is not
bundled in git; retain this attribution and license information with derived distributions.
Dataset licensing is separate from repository code licensing. See docs/DATA_SOURCES.md.

## Experimental rebuilt Pi runtime

The isolated ARMv6 TensorFlow Lite C API runtime is built from TensorFlow v2.20.0,
commit `72fbba3d20f4616d7312b5e2b7f79daf6e82f2fa` (TensorFlow Authors, Apache-2.0):
https://github.com/tensorflow/tensorflow/tree/v2.20.0 . Build-system patches,
compiler settings and dependency revisions are recorded in docs/TFLITE_REBUILD.md
and the build receipt. No TensorFlow inference kernels were locally rewritten.

Retained upstream license/notices for TensorFlow, Abseil, cpuinfo, Eigen, farmhash,
FP16, gemmlowp, ml_dtypes, ruy, Ooura FFT, FlatBuffers and Zig's C++/unwind runtime
are collected alongside the candidate under `runs/tflite-rebuild-20261006/licenses/`.
Those third-party components retain their own notices and licenses. The candidate
does not include the Pi sysroot archive or a copy of the Pi's system glibc.

## Optional offboard detector experiment

Ultralytics YOLO11n and the separately installed `ultralytics==8.3.203` package:
https://docs.ultralytics.com/models/yolo11/ and https://github.com/ultralytics/ultralytics .
Ultralytics identifies its software/models as AGPL-3.0 or separately licensed Enterprise;
they are not relicensed by this repository's Apache-2.0 notice. Keep this experiment's
dependency/model provenance separate and review applicable terms before redistribution.
No commercial license was purchased or accepted by this workflow.

Pretrained checkpoint source:
https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt .
Local SHA-256: `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`.
The detector experiment uses the same attributed CC BY 4.0 DTR source, preserves the grouped
train/validation split, retains tiny positive boxes, and excludes one previously identified
conflicting training source. It does not export or evaluate reserved test images.
