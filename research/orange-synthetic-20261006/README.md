# Orange synthetic research seed

One image generated with the built-in image-generation tool on 2026-10-06,
visually reviewed and approximately boxed by the assistant. Original generated
image is preserved in the workspace as `scene-001.png`; full prompt is in
`prompt.txt`. This is a seed for research, not a useful training dataset by itself.

Three orange latex balloons at different apparent sizes, orange/yellow hollow
goals, and orange cones probe confusion between a filled object and a goal rim.
Production balloon configuration currently supports green/purple, not orange.
The user confirmed orange GOALS on 2026-10-07. Orange balloons are distractors,
not new target labels. An orange balloon must not be labeled an orange goal.

Do not put variants of this scene on both sides of a train/validation split.
Keep all descendants in the same source group. Generated test success is not
evidence of real arena accuracy. Do not automatically use teacher predictions
as ground truth. Training changes require reviewed class mapping, additional
diverse scenes and negatives, and evaluation on a fixed real-data holdout.

For input-pipeline diagnostics only, `probe_synthetic_proposals.py` checks whether
existing proposal methods cover manually annotated boxes at 320x240 and 640x480.
It does not classify objects, train weights, issue flight commands, or claim Pi
timing. Input resizing is preprocessing, not a new independent generated sample.
