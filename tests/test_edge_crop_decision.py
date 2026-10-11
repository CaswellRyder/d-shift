from copy import deepcopy

import pytest

from scripts.audit_edge_crop_decision import audit_results
from scripts.audit_indoor5_outcomes import compare_frames
from scripts.evaluate_red_blue_development import LABELS, detection_counts, metrics


def result(miss=False):
    frames = []
    for i in range(64):
        truth = [dict(label=c, box=[j*30, 0, j*30+20, 20]) for j, c in enumerate(LABELS)]
        detections = [dict(t, score=.99, accepted=True) for t in truth]
        if miss and i == 0:
            detections.pop()
        frames.append(dict(id=i, source=f'train/{i}', frame_sha256=str(i), truth=truth,
                           detections=detections, counts=detection_counts(truth, detections)))
    total = {c: {k: sum(f['counts'][c][k] for f in frames) for k in ('tp', 'fp', 'fn')} for c in LABELS}
    return dict(frames=frames, metrics=metrics(total))


def fixture():
    parent, failed = result(), result(True)
    names = [f'{a}{s}' for a in ('control', 'edge_crop') for s in (47, 48)]
    raw = {'parent': parent, **{n: deepcopy(failed) for n in names}}
    factors = {f'{n}/{p}': deepcopy(failed) for n in names for p in ('both', 'heat_only', 'geometry_only')}
    flags = dict(development_evaluated=False, test_evaluated=False,
                 deployment_approved=False, qualification_achieved=False)
    def summary(rows):
        return {n: dict(metrics=r['metrics'], comparison=compare_frames(r['frames'], parent['frames']))
                for n, r in rows.items()}
    return (dict(flags, training_scenes=summary(raw), eligible_for_development=False), raw,
            dict(flags, results=summary(factors)), factors)


def test_recounts_complete_failed_pairs_and_factor_swaps():
    summaries = audit_results(*fixture())
    assert len(summaries) == 17
    assert summaries['parent']['lost'] == 0
    assert summaries['edge_crop47/heat_only']['lost'] == 1


@pytest.mark.parametrize('case', ['missing', 'count', 'identity', 'duplicate', 'scope', 'parity'])
def test_rejects_incomplete_tampered_or_overclaimed_receipts(case):
    trial, raw, attribution, factors = fixture()
    if case == 'missing':
        factors.pop('edge_crop47/both')
    elif case == 'count':
        raw['control47']['frames'][0]['counts']['blue_balloon']['tp'] += 1
    elif case == 'identity':
        factors['edge_crop47/heat_only']['frames'][0]['frame_sha256'] = 'changed'
    elif case == 'duplicate':
        raw['parent']['frames'][1]['source'] = 'train/0'
    elif case == 'scope':
        attribution['qualification_achieved'] = True
    else:
        factors['control47/both']['frames'][1]['detections'][0]['score'] = .98
    with pytest.raises(ValueError):
        audit_results(trial, raw, attribution, factors)
