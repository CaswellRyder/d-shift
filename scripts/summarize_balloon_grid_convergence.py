"""Consolidate verified continuation, exports and full-scene development; never qualify flight."""
import argparse
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from scripts.audit_balloon_grid_convergence import checked_run
from scripts.summarize_balloon_fragments import PANELS
from scripts.summarize_public_balloon_trial import checked_metrics


def summarize(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    training_path = 'runs/balloon-grid-convergence-training-20261011.json'
    errors_path = 'runs/balloon-grid-convergence-false-positives-20261011.json'
    training, errors = read_json(training_path), read_json(errors_path)
    exports, fits = {}, {}
    for arm in training['eligible_arms']:
        for seed in (47, 48):
            run, doc, _, _, _, p, report, _ = checked_run(arm, seed)
            recorded = training['trials'][arm][str(seed)]
            meta_path = run / 'proposal.fp32.json'
            m = read_json(meta_path)
            if (not recorded['training_gate_pass'] or recorded['model_sha256'] != report['model_sha256']
                    or recorded['provenance_sha256'] != report['provenance_sha256']
                    or recorded['predictions_sha256'] != report['predictions_sha256']
                    or m['source_checkpoint_sha256'] != report['model_sha256']
                    or m['provenance_sha256'] != report['provenance_sha256']
                    or m['sha256'] != sha256(run / 'proposal.fp32.tflite')
                    or m['trial_sha256'] != sha256(training_path) or m['arm'] != arm or m['seed'] != seed
                    or m['decoded_parity_frames'] != len(doc['frames']) or not 0 <= m['max_tensor_delta'] <= .002
                    or m['decoder_sha256'] != sha256(m['decoder_path'])
                    or m['export_script_sha256'] != sha256('scripts/export_balloon_grid_convergence.py')):
                raise ValueError('Training gate or export evidence changed')
            name = arm + str(seed)
            exports[name] = dict(path=str(run / 'proposal.fp32.tflite'), sha256=m['sha256'],
                metadata_sha256=sha256(meta_path), bytes=m['bytes'], parameters=m['parameters'],
                parity_frames=m['decoded_parity_frames'], max_tensor_delta=m['max_tensor_delta'])
            fits[name] = dict(coverage=recorded['continued'], optimizer_start=p['optimizer_restored'],
                             optimizer_end=report['optimizer_final'])
    panels = {}
    for panel in PANELS:
        path = Path(f'runs/balloon-grid-convergence-scenes-{panel}-20261011.json')
        doc = read_json(path)
        e = errors['panels'][panel]
        if (e['sha256'] != sha256(path) or doc['trial_sha256'] != sha256(training_path)
                or doc['classifier_threshold'] != .8 or doc['matching_iou'] != .5
                or doc['proposal_threshold'] != .1 or doc['selection'] != 'split_merged'
                or doc['split_max_aspect'] != 2.42913421076193
                or set(doc['models']) != set(exports)):
            raise ValueError('Scene operating point or attribution changed')
        for name, identity in doc['models'].items():
            if any(identity[k] != exports[name][k] for k in ('path', 'sha256', 'metadata_sha256')):
                raise ValueError('Evaluated export identity changed')
        results = {name: checked_metrics(result) for name, result in doc['results'].items()}
        if any(e['results'][name]['metrics'] != metrics for name, metrics in results.items()):
            raise ValueError('Attribution and full-scene metrics disagree')
        panels[panel] = dict(path=str(path), sha256=sha256(path), results=results,
            error_counts={name: r['error_counts'] for name, r in e['results'].items()})
    meets = {name: all((metric['precision'] or 0) >= .94 and (metric['recall'] or 0) >= .94
        for panel in panels.values() for metric in panel['results'][name].values()) for name in exports}
    result = dict(training_report_sha256=sha256(training_path), false_positive_report_sha256=sha256(errors_path),
        fits=fits, exports=exports, panels=panels, all_development_panels_meet_target=meets,
        decision='No runtime promotion: improved training and recall do not offset false positives.',
        next_step='Review new localizer-generated hard negatives only from already approved complete TRAIN scenes; exclude every reserved IMG source and any crop touching a known balloon. Do not mine these development errors into training.',
        script_sha256=sha256(__file__),
        scope='Paired TRAIN continuation and repeated development only. Related panels are not pooled as independent observations. Camera-inclusive representative qualification remains missing.',
        test_evaluated=False, pi_tested=False, deployment_approved=False, qualification_achieved=False)
    write_json(output, result)
    print(dict(development_target=meets, exports=exports))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    summarize(parser.parse_args().output)
