# Hard-negative refinement and full-scene development check

**Not flight ready. The 94% per-color precision/recall target is not met.**

## Label review

The initial student accepted seven candidate regions with zero overlap against
upstream balloon boxes in training photos. Full-photo visual review established:

- Admit 0–5 as negatives: patterned trousers, wig, shirt print, red container,
  and two hair patches.
- Exclude 6: two real red balloons missing from upstream annotations. The earlier
  thumbnail impression of lanterns was incorrect. Never train this crop as background.

`configs/balloon-red-blue-negative-review-20261008.json` binds the decision to
the exact review queue. Raw evidence is local in
`data/balloon-red-blue-negative-review-20261008/`. This is AI review, not human certification.

## Matched refinement

Starting from the same 15,667-parameter student, both runs use seed 42, learning
rate 1e-4, eight epochs, 260 entries and 17 optimizer steps per epoch. The control
adds 48 sampled original entries; the intervention adds six reviewed negatives
repeated eight times. Original teacher targets are unchanged. New negatives use
hard labels, not fabricated teacher logits. No reserved-test predictions were made.

| Result | Original | Matched control | Refined |
|---|---:|---:|---:|
| Accepted reviewed negatives (training fit only) | 6/6 | 6/6 | 3/6 |
| Development crop argmax accuracy | 8/8 | 8/8 | 8/8 |
| Full-scene red TP / FP / FN, baseline search | 2 / 3 / 0 | 2 / 4 / 0 | 2 / 1 / 0 |
| Full-scene blue TP / FP / FN, baseline search | 1 / 0 / 2 | 1 / 0 / 2 | 1 / 0 / 2 |

The full-scene check uses four manually inspected, off-domain development photos
with two red and three blue target balloons. The review is recorded in
`configs/balloon-red-blue-development-scenes.json`. Counts include missed proposals
and threshold rejection at 0.8, with same-color one-to-one matching at IoU >=0.5.
Duplicate detections count as false positives. This tiny reused development set
is nowhere near the independent recording evidence required for flight acceptance.

## Search regressions retained, not promoted

MSER with the original student recovered all three blue targets in development,
but also emitted nine red false detections, largely printed details inside a
larger red balloon. The refined model regressed to one blue detection with MSER.
An experimental containment filter removed internal duplicates, but training
proposal coverage fell from 19/22 to 11/22 red; development blue detections also
fell from three to one with the original student. It is not an acceptable general
fix and remains research-only. No default runtime profile was changed.

Reports: `refinement-development.json`, `components-development.json`, and
`search-components-ablation.json`. Initial MSER/threshold results remain preserved
in `search-ablation.json` and the earlier training receipt.

## Remaining work

- More representative red/blue positive scenes and non-balloon colored clutter.
- Better localization and proposal-compatible crop training; 8/8 crop scores do
  not resolve the full-frame errors above.
- Red/blue Pi bundle/runtime parity, isolated timing and live-camera capture.
- Independent complete-frame validation, distance/size bins and the 94% gate.

Pi connection diagnosis: USB Ethernet has `10.12.194.10`, but routing to
`10.12.194.1` selects VPN interface `utun13` instead of USB `en15`. SSH failed by
hostname and IP. User was asked to disconnect the VPN if permitted; no route/VPN
settings were changed and no remote, ESP32 or motor commands executed.
