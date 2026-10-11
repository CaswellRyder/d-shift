from copy import deepcopy

import pytest

from scripts.audit_background_convergence_scenes import recount
from scripts.audit_indoor5_outcomes import compare_frames
from scripts.evaluate_red_blue_development import detection_counts, metrics


def fixture():
    truth = [dict(label='red_balloon', box=[0, 0, 10, 10]),
             dict(label='blue_balloon', box=[20, 20, 30, 30])]
    detections = [dict(t, crop_box=t['box'], score=.99, accepted=True, raw_accepted=True) for t in truth]
    frame = dict(id=0, source='train/example.jpg', frame_sha256='fixture', truth=truth,
                 detections=detections, counts=detection_counts(truth, detections), neural_calls=2)
    result = dict(frames=[frame], metrics=metrics(frame['counts']))
    names = ['positive_only47', 'positive_only48']
    panels = ['training', 'development1', 'development2', 'development3']
    raw = {p: {n: deepcopy(result) for n in ['parent']+names} for p in panels}
    summary = dict(metrics=result['metrics'], comparison=compare_frames([frame], [frame]),
                   nonregression=True, meets_94=True, neural_calls=2)
    doc = dict(panels={p: {n: deepcopy(summary) for n in ['parent']+names} for p in panels},
        eligible_for_development=names, eligible_for_pi_comparison=names,
        development_evaluated=True, test_evaluated=False, pi_tested=False,
        deployment_approved=False, qualification_achieved=False)
    return doc, raw, {p: deepcopy(result) for p in panels}, names


def test_recount_accepts_complete_matching_fixture():
    doc, raw, baseline, names = fixture()
    summaries, viable = recount(doc, raw, baseline, names)
    assert len(summaries) == 4 and viable == names


@pytest.mark.parametrize('mutation', ['missing_panel', 'metric', 'scope', 'geometry', 'workload', 'seed'])
def test_recount_rejects_corrupted_evidence(mutation):
    doc, raw, baseline, names = fixture()
    if mutation == 'missing_panel':
        raw.pop('development1')
    elif mutation == 'metric':
        doc['panels']['training']['positive_only47']['metrics']['blue_balloon']['tp'] = 2
    elif mutation == 'scope':
        doc['qualification_achieved'] = True
    elif mutation == 'geometry':
        raw['training']['positive_only47']['frames'][0]['detections'][0]['crop_box'] = [0, 0, 11, 11]
    elif mutation == 'workload':
        raw['training']['positive_only47']['frames'][0]['neural_calls'] = 3
    else:
        raw['training'].pop('positive_only48')
    with pytest.raises(ValueError):
        recount(doc, raw, baseline, names)
