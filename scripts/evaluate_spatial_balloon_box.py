"""Matched scene replay for frozen box heads; no threshold or search changes."""
import argparse
from pathlib import Path

from dtr.data import read_json,sha256,write_json
from scripts.evaluate_balloon_box_refinement import BoxPredictor,observations
from scripts.evaluate_reviewed_balloon_scenes import reviewed_frames
from scripts.compare_box_selection import locked_selection
from scripts.evaluate_red_blue_development import LABELS,detection_counts,metrics
from scripts.pi_balloon_search_bench import select_detections
from scripts.summarize_public_balloon_trial import checked_metrics


def checked_source(source, references, review, review_path, metadata):
    if (source['model_sha256'] != metadata['source_tflite_sha256']
            or source['queue_sha256'] != review['queue_sha256']
            or source['review_sha256'] != sha256(review_path)
            or source['threshold'] != .8 or source['matching_iou'] != .5
            or source['selection'] != 'split_merged' or source['split_max_aspect'] != 2.42913421076193
            or source['test_evaluated'] is not False or source['deployment_approved'] is not False):
        raise ValueError('Frozen source identity, geometry policy or scope changed')
    baseline = source['variants']['baseline']
    checked_metrics(baseline)
    if len(baseline['frames']) != len(references) or {f['id'] for f in baseline['frames']} != set(references):
        raise ValueError('Source does not cover all reviewed frames')
    for frame in baseline['frames']:
        ref = references[frame['id']][0]
        if (frame['truth'] != ref['provisional_truth'] or frame['source'] != ref['source']
                or frame['frame_sha256'] != ref['sha256']):
            raise ValueError('Source frame identity or reviewed truth changed')
    return baseline


def evaluate(model,source_path,queue,review_path):
    predictor = BoxPredictor(model)
    frames,review = reviewed_frames(queue,review_path)
    references = {r['id']:(r,pixels) for r,pixels in frames}
    source = read_json(source_path)
    baseline = checked_source(source,references,review,review_path,predictor.metadata)
    outputs = {k:[] for k in ('parent','locked_selection','fresh_selection')}
    maximum = 0.
    for frame in baseline['frames']:
        original,refined,delta = observations(predictor,references[frame['id']][1],frame['detections'])
        maximum = max(maximum,delta)
        parent = select_detections(original,'split_merged',2.42913421076193)
        fresh = select_detections(refined,'split_merged',2.42913421076193)
        if any(a['accepted'] != b['accepted'] for a,b in zip(parent,frame['detections'],strict=True)):
            raise ValueError('Parent instance selection changed')
        fixed = locked_selection(parent,fresh)
        for name,found in (('parent',parent),('locked_selection',fixed),('fresh_selection',fresh)):
            outputs[name].append(dict(id=frame['id'],source=frame['source'],frame_sha256=frame['frame_sha256'],
                                      truth=frame['truth'],detections=found,
                                      counts=detection_counts(frame['truth'],found)))
    results = {}
    for name,rows in outputs.items():
        total = {label:{key:sum(r['counts'][label][key] for r in rows) for key in ('tp','fp','fn')}
                 for label in LABELS}
        results[name] = dict(metrics=metrics(total),frames=rows)
    if results['parent']['metrics'] != baseline['metrics']:
        raise ValueError('Parent counts not reproduced')
    return dict(results=results,model_sha256=sha256(model),metadata_sha256=sha256(Path(model).with_suffix('.json')),
                source_report_sha256=sha256(source_path),queue_sha256=review['queue_sha256'],
                review_sha256=sha256(review_path),max_class_probability_delta=maximum,
                script_sha256=sha256(__file__),pool_grid=predictor.metadata['pool_grid'],
                selection='split_merged',split_max_aspect=2.42913421076193,threshold=.8,matching_iou=.5,
                helpers_sha256={p:sha256(p) for p in ('scripts/evaluate_balloon_box_refinement.py',
                    'scripts/compare_box_selection.py','scripts/pi_balloon_search_bench.py',
                    'scripts/train_balloon_box_refinement.py','scripts/evaluate_red_blue_development.py')},
                scope='Repeatedly inspected development; same proposals and class probabilities. Both fixed and fresh selection predeclared.',
                test_evaluated=False,deployment_approved=False,pi_tested=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model','source','queue','review','output'):
        p.add_argument('--'+name,required=True)
    a = p.parse_args()
    if Path(a.output).exists():
        raise FileExistsError(a.output)
    result = evaluate(a.model,a.source,a.queue,a.review)
    write_json(a.output,result)
    print({k:v['metrics'] for k,v in result['results'].items()})
