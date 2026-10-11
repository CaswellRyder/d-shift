"""Independent TRAIN recount for source-priority and direct-localizer experiments."""
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from scripts.audit_source_aware_balloon import OUTPUT as SOURCE
from scripts.audit_direct_localizer_balloon import OUTPUT as DIRECT
from scripts.audit_balanced_localizer import screen
from scripts.audit_joint_balanced_balloon import PANELS, paired_gate
from scripts.diagnose_clutter_stage_attribution import difference

OUTPUT = Path('research/balloon-red-blue-20261008/selection-trials-decision-20261011.json')


def checked(path):
    report = read_json(path)
    for filename, digest in report['inputs_sha256'].items():
        if sha256(filename) != digest:
            raise ValueError('Experiment input changed')
    if sha256(report['raw']['path']) != report['raw']['sha256']:
        raise ValueError('Predictions changed')
    return report, read_json(report['raw']['path'])


def summarize():
    if OUTPUT.exists():
        raise FileExistsError('Preserve decisions')
    rows = {}
    for name, path, protocol in (
            ('source_priority', SOURCE, 'configs/source-aware-balloon-20261011.json'),
            ('direct_localizer', DIRECT, 'configs/direct-localizer-balloon-20261011.json')):
        report, raw = checked(path)
        p = read_json(protocol)
        previous = read_json(p['source_audit'] if name == 'source_priority' else p['localizer_audit'])
        if sha256(previous['raw']['path']) != previous['raw']['sha256']:
            raise ValueError('Reference predictions changed')
        baseline = read_json(previous['raw']['path'])
        reference_name = 'parent_parent47_current' if name == 'source_priority' else 'parent_current'
        if set(raw) != set(PANELS):
            raise ValueError('Panel coverage changed')
        recounted = {}
        for panel, variants in raw.items():
            expected_names = ({f'{loc}_{cla}{seed}_{policy}' for loc in ('parent', 'control', 'balanced')
                for cla in ('parent', 'control', 'admit') for seed in (47, 48)
                for policy in ('current', 'reserved')} if name == 'source_priority' else
                {'parent', 'control47', 'control48', 'balanced47', 'balanced48'})
            if set(variants) != expected_names:
                raise ValueError('Checkpoint or control coverage changed')
            recounted[panel] = {}
            for variant, result in variants.items():
                if len(result['frames']) != PANELS[panel]:
                    raise ValueError('Frame count changed')
                for frame in result['frames']:
                    if len(frame['detections']) > 12:
                        raise ValueError('Detection budget exceeded')
                    if name == 'source_priority':
                        if frame['neural_calls'] != len(frame['detections']):
                            raise ValueError('Classifier-call accounting changed')
                    elif frame['neural_calls'] != 0 or frame['full_frame_neural_calls'] != 1:
                        raise ValueError('Direct-detector call accounting changed')
                fresh = screen(result, baseline[panel][reference_name])
                if name == 'source_priority':
                    fresh['selection_effect'] = difference(result, baseline[panel][variant])
                if fresh != report['panels'][panel][variant]:
                    raise ValueError('Independent counts or target identities disagree')
                recounted[panel][variant] = fresh
        gate = ({policy: paired_gate(recounted, policy) for policy in ('current', 'reserved')}
                if name == 'source_priority' else
                all(all(panel[f'balanced{seed}']['no_regression'] for panel in recounted.values()) and
                    any(panel[f'balanced{seed}']['effect']['gained_targets'] for panel in recounted.values())
                    for seed in (47, 48)))
        if gate != report['eligible_for_separate_export_development']:
            raise ValueError('Eligibility decision changed')
        rows[name] = dict(receipt=str(path), receipt_sha256=sha256(path),
            raw_sha256=report['raw']['sha256'], eligible_for_separate_export_development=gate,
            decision=report['decision'], variants=len(expected_names), frames_per_variant=82)
    write_json(OUTPUT, dict(results=rows, script_sha256=sha256(__file__),
        finding='Source priority alone trades errors without passing retention. Direct balanced localizers find10of13window2red targets in both seeds with0or1FP, but lose too many original-scene detections. Neither replaces the hybrid.',
        next_experiment='Freeze a hybrid that retains the full MSER candidate set and supplements accepted crop detections with direct neural detections. This tests complementary coverage without reclassifying learned boxes. Compare source-priority and pooled-score merging with matched controls; do not assume heat and classifier confidences are calibrated. Keep classifier-call count explicit, and measure Pi latency separately only after accuracy gates.',
        scope='Fresh recount of41variants across82TRAIN scenes each. No neural execution by this summarizer. Assistant-reviewed TRAIN labels and repeated fitting scenes are not independent qualification.',
        training_performed=False, development_evaluated=False, test_evaluated=False,
        pi_tested=False, deployment_approved=False, qualification_achieved=False))
    print(rows, flush=True)


if __name__ == '__main__':
    summarize()
