"""Fixed TRAIN-only decoder intervention with exact parent reproduction."""
from pathlib import Path
from time import perf_counter

from dtr.data import read_json, sha256, write_json
from dtr.native_balloon_regions import NativeBalloonRegions
from dtr.runtime import Predictor
from scripts.audit_balloon_dense_proposals import supplement
from scripts.audit_indoor5_outcomes import compare_frames
from scripts.balloon_dense_runtime import DenseProposals
from scripts.evaluate_red_blue_development import LABELS, detection_counts, metrics
from scripts.prepare_blue_false_alarm_review import source_frames
from scripts.research_blue_box_nms import decode
from scripts.research_blue_bright_search import search
from scripts.research_blue_peak_blend import decode as parent_decode
from scripts.research_confident_containment import select
from scripts.research_predictor import classify_candidates
from scripts.summarize_dense_background_trial import check_baseline
from scripts.summarize_public_balloon_trial import checked_metrics

PROTOCOL = Path('configs/blue-box-nms-20261011.json')
OUTPUT = Path('research/balloon-red-blue-20261008/blue-box-nms-training-20261011.json')
RAW = Path('runs/blue-box-nms-training-20261011.json')


def evaluate():
    if OUTPUT.exists() or RAW.exists():
        raise FileExistsError(OUTPUT)
    p = read_json(PROTOCOL)
    for name in ('parent_receipt', 'localizer', 'classifier'):
        if sha256(p[name]) != p[name+'_sha256']:
            raise ValueError('Protocol identity changed')
    reference = read_json(p['parent_receipt'])
    if sha256(reference['raw']['path']) != reference['raw']['sha256']:
        raise ValueError('Parent raw outputs changed')
    prior = read_json(reference['raw']['path'])['parent']
    checked_metrics(prior)
    helpers = {path: sha256(path) for path in (
        str(PROTOCOL), __file__, 'scripts/research_blue_box_nms.py',
        'scripts/research_blue_peak_blend.py', 'scripts/research_blue_bright_search.py',
        'scripts/research_confident_containment.py', 'scripts/audit_balloon_dense_proposals.py',
        'scripts/research_predictor.py', 'scripts/prepare_blue_false_alarm_review.py')}
    localizer = DenseProposals(p['localizer'])
    classifier = Predictor(p['classifier'], allow_unvalidated=True)
    native = NativeBalloonRegions('artifacts/balloon-regions-mac-20261008/regions.so', direct=True)
    if classifier.metadata['threshold'] != .8:
        raise ValueError('Classifier acceptance threshold changed')
    pixels = {s['source']: (s, rgb) for s, rgb in source_frames() if s['origin'] == 'complete_fit'}
    if len(pixels) != 64 or len(prior['frames']) != 64:
        raise ValueError('Wrong TRAIN inventory')
    rows = {'parent': [], 'box_nms': []}
    times = {name: [] for name in rows}
    for index, frame in enumerate(prior['frames']):
        source, rgb = pixels[frame['source']]
        if source['source_file_sha256'] != frame['frame_sha256'] or source['truth'] != frame['truth']:
            raise ValueError('Different TRAIN image or labels')
        base = search(rgb, native, True)
        localizer.predict(rgb)
        grid = localizer.interpreter.get_tensor(localizer.output['index'])[0]
        for name in rows:
            start = perf_counter()
            learned = parent_decode(grid, True) if name == 'parent' else decode(grid)
            times[name].append((perf_counter()-start)*1000)
            proposed = supplement(base, learned)
            if len(learned) > 12 or len(proposed) > max(len(base), min(12, len(base)+2)):
                raise ValueError('Inference budget changed')
            detections = select(classify_candidates(classifier, rgb, proposed))
            row = dict({k: frame[k] for k in ('id', 'source', 'frame_sha256', 'truth')},
                detections=detections, counts=detection_counts(frame['truth'], detections),
                neural_calls=len(proposed))
            if name == 'parent':
                check_baseline(row, frame)
            rows[name].append(row)
        if (index+1) % 16 == 0:
            print(dict(complete=index+1, total=64), flush=True)
    full, summaries = {}, {}
    for name, frames in rows.items():
        total = {c: {k: sum(f['counts'][c][k] for f in frames) for k in ('tp', 'fp', 'fn')} for c in LABELS}
        full[name] = dict(frames=frames, metrics=metrics(total))
        checked_metrics(full[name])
        comparison = compare_frames(frames, rows['parent'])
        summaries[name] = dict(metrics=full[name]['metrics'], comparison=comparison,
            neural_calls=sum(f['neural_calls'] for f in frames),
            desktop_decoder_mean_ms=sum(times[name])/len(times[name]))
    candidate, parent = summaries['box_nms'], summaries['parent']
    eligible = (not candidate['comparison']['lost_parent_targets']
        and bool(candidate['comparison']['gained_targets'])
        and all(candidate['metrics'][c]['tp'] >= parent['metrics'][c]['tp']
                and candidate['metrics'][c]['fp'] <= parent['metrics'][c]['fp'] for c in LABELS))
    if any(sha256(path) != digest for path, digest in helpers.items()):
        raise ValueError('Experiment helpers changed while running')
    write_json(RAW, full)
    write_json(OUTPUT, dict(protocol=p, helpers_sha256=helpers, results=summaries,
        raw=dict(path=str(RAW), sha256=sha256(RAW)), eligible_for_development=eligible,
        native_sha256=sha256(native.path),
        scope='64 complete TRAIN scenes only. Parent reproduced exactly except timing. Desktop decoder timing is diagnostic, not Pi FPS or camera latency. Single fixed intervention; no threshold sweep.',
        development_evaluated=False, test_evaluated=False, pi_tested=False,
        deployment_approved=False, qualification_achieved=False))
    print(dict(results=summaries, eligible_for_development=eligible), flush=True)


if __name__ == '__main__':
    evaluate()
