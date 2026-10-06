# Public sources

**Update 2026-10-03:** the user supplied authorized export access. COCO V10 and V11 from
Cheese/Cats-and-Dogs have been downloaded, with license files and archive checksums retained.
Both contain all eight target classes. V10 was deduplicated and re-split into both task datasets.
See [current acquisition and training receipt](REAL_DATA_STATUS.md). The labeled-data access
blocker is resolved; competition qualification is not.

Inspected 2026-09-28. Public metadata/model demos are NOT proof that weights or bulk data
can be downloaded without authentication. No credentials are stored in this project.

## Preferred colored-balloon source

- https://universe.roboflow.com/cheese-geozd/cats-and-dogs-bmity
- Published name: Cats-and-Dogs by Cheese (the labels are balloons/goals, despite the name).
- Project metadata: 6,180 images, CC BY 4.0; green/purple balloons and orange/yellow circle,
  square and triangle goals. Also an ambiguous generic Balloons class.
- Listed model `cats-and-dogs-bmity/8`: YOLOv8s segmentation; version metadata says 2,602 images.
- Do not equate project image count, model-version image count, and distinct recording count.
- Export attempt to `https://api.roboflow.com/cheese-geozd/cats-and-dogs-bmity/8/coco`
  returned HTTP 401: requires an API key. No bypass attempted.
- Authorized V10/V11 COCO exports now live at `data/raw/roboflow-dtr-v{10,11}/` locally.
  Both export READMEs and COCO license entries explicitly identify CC BY 4.0. Credit
  Cheese / Cats-and-Dogs, Roboflow Universe. Derived data modifies crops and splits and
  removes duplicates/conflicts. No source image recoloring or generated imagery was added.
- Both versions contain 6,180 images and the same 6,054 original filename stems.
  V10 uses 640x640 images; V11 uses 180x180. Only V10 feeds this training run.
- The pipeline needs annotations, not necessarily the hosted detector's weights.

## Alternative source

- https://universe.roboflow.com/dtr-z6mpv/dtr-thwjt
- Public DTR project metadata: 592 images, CC BY 4.0, YOLO11 detector listed.
- Classes: Ball, orange/yellow circle/square/triangle goals, unexplained labels 2 and 88.
- Suitable goal-data lead. Cannot use generic Ball as a green/purple class without review.
- Neither project's official competition affiliation nor camera-domain suitability is verified.

## Pretrained general vision backbone (downloaded by `dtr pretrained`)

- https://huggingface.co/timm/mobilenetv4_conv_small.e2400_r224_in1k
- ImageNet-1k checkpoint by Ross Wightman / timm, Apache-2.0 model-card license.
- Not a DTR teacher until specialized on real labeled crops.
- Only safetensors/public model download handled by timm; the converted Keras artifact has
  a checksum and numerical parity receipt. Do not load arbitrary untrusted pickle checkpoints.

## Data hygiene

Keep human-reviewed labels authoritative. A detector's proposed labels are pseudo-labels,
not human truth or an automatic substitute for distillation. Keep recording sessions apart,
deduplicate images, retain attribution, review background crops, and test on separate Pi-camera
recordings. CC BY attribution is required; independently check model-weight terms before reuse.

Generic real-balloon bootstrap artifacts remain separate historical experiments. Do not
mix their validation/test sets into DTR training or treat either source as Pi-camera proof.
