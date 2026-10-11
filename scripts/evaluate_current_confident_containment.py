"""Fixed postprocessing replay on current TRAIN and repeated development scenes."""
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from scripts.audit_indoor5_outcomes import compare_frames
from scripts.compare_current_blue_search import MODEL, MODEL_SHA
from scripts.evaluate_red_blue_development import LABELS, detection_counts, metrics
from scripts.replay_balloon_suppression import raw
from scripts.research_balloon_confidence_selection import confidence_selection
from scripts.research_confident_containment import select
from scripts.summarize_public_balloon_trial import checked_metrics

REFERENCE = Path('research/balloon-red-blue-20261008/blue-peak-blend-scenes-20261011.json')
PROTOCOL = Path('configs/current-confidence-containment-20261011.json')
OUTPUT = Path('research/balloon-red-blue-20261008/current-confidence-containment-20261011.json')
RAW = Path('runs/current-confidence-containment-20261011.json')


def nonregressing(before, after, comparison):
    return not comparison['lost_parent_targets'] and all(
        after[c]['tp'] >= before[c]['tp'] and after[c]['fp'] <= before[c]['fp'] for c in LABELS)


def checked_reference():
    report = read_json(REFERENCE)
    if (sha256(report['raw']['path']) != report['raw']['sha256']
            or sha256(MODEL) != MODEL_SHA or report['classifier_sha256'] != MODEL_SHA
            or report['script_sha256'] != sha256('scripts/evaluate_blue_peak_blend.py')
            or any(sha256(p) != h for p, h in report['helpers_sha256'].items())
            or any(sha256(p) != h for p, h in report['inputs'].items())):
        raise ValueError('Frozen current-pipeline evidence changed')
    for model in report['models'].values():
        if (sha256(model['path']) != model['sha256']
                or sha256(Path(model['path']).with_suffix('.json')) != model['metadata_sha256']):
            raise ValueError('Localizer identity changed')
    return report, read_json(report['raw']['path'])


def evaluate():
    if OUTPUT.exists() or RAW.exists():
        raise FileExistsError(OUTPUT)
    reference, previous = checked_reference()
    if len(previous) != 4 or 'training' not in previous:
        raise ValueError('Unexpected evaluation panels')
    full, panels = {}, {}
    for panel in ['training']+[p for p in previous if p != 'training']:
        old = previous[panel]['background/blend']
        if checked_metrics(old) != reference['panels'][panel]['background/blend']['metrics']:
            raise ValueError('Frozen aggregate counts changed')
        if panel == 'training' and (len(old['frames']) != 64 or any('IMG_' in f['source'] for f in old['frames'])):
            raise ValueError('TRAIN boundary changed')
        frames, changes = [], []
        for frame in old['frames']:
            observations = raw(frame['detections'])
            control = confidence_selection(observations)
            if control != frame['detections']:
                raise ValueError('Original selection did not reproduce exactly')
            selected = select(observations)
            for index, (a, b) in enumerate(zip(control, selected, strict=True)):
                for key in ('box', 'crop_box', 'label', 'score', 'scores', 'raw_accepted'):
                    if a[key] != b[key]:
                        raise ValueError('Frozen geometry or prediction changed')
                if a['accepted'] != b['accepted']:
                    changes.append(dict(source=frame['source'], index=index, label=a['label'], box=a['box'],
                        before=a['accepted'], after=b['accepted'], score=a['score']))
            frames.append(dict(frame, detections=selected, counts=detection_counts(frame['truth'], selected)))
        total = {c: {k: sum(f['counts'][c][k] for f in frames) for k in ('tp', 'fp', 'fn')} for c in LABELS}
        result = dict(frames=frames, metrics=metrics(total))
        checked_metrics(result)
        comparison = compare_frames(frames, old['frames'])
        gate = nonregressing(old['metrics'], result['metrics'], comparison)
        panels[panel] = dict(before=old['metrics'], after=result['metrics'], comparison=comparison,
            nonregression=gate, changes=changes, frames=len(frames),
            neural_calls_before=sum(f['neural_calls'] for f in old['frames']),
            neural_calls_after=sum(f['neural_calls'] for f in frames))
        full[panel] = result
        print(panel, dict(metrics=result['metrics'], nonregression=gate,
            lost=len(comparison['lost_parent_targets']), gained=len(comparison['gained_targets'])), flush=True)
        if panel == 'training' and not gate:
            break
    write_json(RAW, full)
    write_json(OUTPUT, dict(reference_sha256=sha256(REFERENCE), protocol_sha256=sha256(PROTOCOL),
        script_sha256=sha256(__file__), helpers_sha256={p: sha256(p) for p in (
            'scripts/research_confident_containment.py', 'scripts/compare_balloon_containment.py',
            'scripts/research_balloon_confidence_selection.py', 'scripts/replay_balloon_suppression.py')},
        raw=dict(path=str(RAW), sha256=sha256(RAW)), panels=panels,
        eligible_for_pi_comparison=len(panels)==4 and all(p['nonregression'] for p in panels.values()),
        meets_94_each_evaluated_panel=all(all(p['after'][c][m] >= .94 for c in LABELS
            for m in ('precision', 'recall')) for p in panels.values()),
        scope='Existing fixed containment rule composed with newer pipeline. Saved TRAIN/development prediction replay, not new camera inference, independent accuracy or measured Pi speed.',
        test_evaluated=False, pi_tested=False, deployment_approved=False, qualification_achieved=False))


if __name__ == '__main__':
    evaluate()
