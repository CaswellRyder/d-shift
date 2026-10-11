"""Independently recount direct-hybrid outcomes and attribute remaining TRAIN errors."""
from collections import Counter
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.audit_direct_hybrid_balloon import OUTPUT as AUDIT, checked_inputs, paired_gate
from scripts.audit_joint_balanced_balloon import PANELS
from scripts.audit_balanced_localizer import screen
from scripts.audit_balloon_pipeline_errors import analyze_frame
from scripts.diagnose_clutter_stage_attribution import difference

OUTPUT = Path('research/balloon-red-blue-20261008/direct-hybrid-balloon-decision-20261011.json')


def key(row):
    return tuple(row['box']), tuple(row['crop_box']), row['color_group']


def summarize():
    if OUTPUT.exists():
        raise FileExistsError('Preserve decision evidence')
    _, refs, _, _ = checked_inputs()
    prior = read_json(refs['joint']['raw']['path'])
    report = read_json(AUDIT)
    for path, digest in {**report['inputs_sha256'], **report['models_sha256']}.items():
        if sha256(path) != digest:
            raise ValueError('Audit dependency changed')
    if sha256(report['raw']['path']) != report['raw']['sha256']:
        raise ValueError('Audited predictions changed')
    raw = read_json(report['raw']['path'])
    if set(raw) != set(PANELS):
        raise ValueError('Panel coverage changed')
    recounted, details, common = {}, {}, {}
    for panel, variants in raw.items():
        expected = {f'{loc}_{cla}{s}_{policy}' for loc in ('parent', 'control', 'balanced')
                    for cla in ('parent', 'control', 'admit') for s in (47, 48)
                    for policy in ('pooled', 'base_first')}
        expected.update(f'base_{name}' for name in ('parent', 'control47', 'control48', 'admit47', 'admit48'))
        if set(variants) != expected:
            raise ValueError('Factorial controls incomplete')
        recounted[panel], details[panel], common[panel] = {}, {}, {}
        for name, result in variants.items():
            if len(result['frames']) != PANELS[panel]:
                raise ValueError('Scene coverage changed')
            for frame in result['frames']:
                base = name.startswith('base_')
                if (not 0 <= frame['neural_calls'] <= 12 or len(frame['detections']) > (12 if base else 24)
                        or frame['full_frame_neural_calls'] != (0 if base else 1)):
                    raise ValueError('Call or candidate accounting changed')
                if not base and sum(r['proposal_origin'] == 'mser' for r in frame['detections']) != frame['neural_calls']:
                    raise ValueError('Crop-call provenance mismatch')
            fresh = screen(result, prior[panel]['parent_parent47_current'])
            expected_row = {k: v for k, v in report['panels'][panel][name].items() if k != 'comparisons'}
            if fresh != expected_row:
                raise ValueError('Recounted counts or retention disagree')
            recounted[panel][name] = fresh
            if name.startswith('base_'):
                classifier = name.removeprefix('base_')
                classifier = 'parent47' if classifier == 'parent' else classifier
                count = 0
                for actual, old in zip(result['frames'], prior[panel][f'parent_{classifier}_current']['frames'], strict=True):
                    if any(actual[k] != old[k] for k in ('id', 'source', 'truth', 'frame_sha256')):
                        raise ValueError('Classifier reference frame differs')
                    reference = {key(r): r for r in old['detections']}
                    for row in actual['detections']:
                        if key(row) in reference:
                            saved = reference[key(row)]
                            if row['label'] != saved['label'] or row['raw_accepted'] != saved['raw_accepted']:
                                raise ValueError('Shared-crop decision changed')
                            np.testing.assert_allclose(row['scores'], saved['scores'], atol=1e-5, rtol=0)
                            count += 1
                if not count:
                    raise ValueError('No shared crop control reproduced')
                common[panel][name] = count
            if name.startswith('balanced_admit'):
                suffix = name.removeprefix('balanced_admit')
                seed = suffix[:2]
                comparisons = {
                    'matched_control': difference(result, variants[f'control_control{suffix}']),
                    'classifier_only': difference(result, variants[f'base_admit{seed}']),
                    'localizer_only': difference(result, variants[f'balanced_parent{suffix}'])}
                if comparisons != report['panels'][panel][name]['comparisons']:
                    raise ValueError('Component comparison differs')
                errors = Counter()
                for frame in result['frames']:
                    for error in analyze_frame(frame)['false_positives']:
                        origin = frame['detections'][error['index']]['proposal_origin']
                        errors[f"{origin}/{error['label']}/{error['reason']}"] += 1
                details[panel][name] = dict(metrics=fresh['metrics'], false_positives_by_source=dict(errors),
                    false_negatives=fresh['errors']['false_negatives'], lost_targets=fresh['effect']['lost_targets'],
                    gained_targets=fresh['effect']['gained_targets'], crop_calls=fresh['neural_calls'])
    gates = {policy: paired_gate(recounted, policy) for policy in ('pooled', 'base_first')}
    if gates != report['eligible_for_separate_export_development']:
        raise ValueError('Eligibility gate changed')
    write_json(OUTPUT, dict(audit_sha256=sha256(AUDIT), raw_sha256=report['raw']['sha256'],
        script_sha256=sha256(__file__), eligible_for_separate_export_development=gates,
        panels=details, reproduced_shared_crop_decisions=common,
        decision=report['decision'],
        next_experiment='Prioritize crop-classifier hard negatives: all7window2false positives in seed47 and6of7in seed48 originate in MSER crop classification; the other seed48false positive is a learned box localization error. Mine only TRAIN crops outside all reviewed target boxes plus16pixel uncertainty guards, then visually review before admitting backgrounds. Old-scene learned duplicates/localization errors require separate geometry work, not automatic background labels. Do not use IMG_ or reserved scenes.',
        scope='Independent recount of41variants over82TRAIN scenes and shared-crop classification parity. This is not held-out, live-camera or flight qualification.',
        training_performed=False, development_evaluated=False, test_evaluated=False,
        pi_tested=False, deployment_approved=False, qualification_achieved=False))
    print(dict(eligible=gates, shared_crop_checks=common), flush=True)


if __name__ == '__main__':
    summarize()
