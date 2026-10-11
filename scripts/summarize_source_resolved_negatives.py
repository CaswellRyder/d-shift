"""Recount source-resolved-negative trial decisions, paired effects and artifact integrity."""
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from scripts.audit_source_resolved_negatives import OUTPUT as AUDIT, eligible_policies
from scripts.audit_balanced_localizer import screen
from scripts.diagnose_clutter_stage_attribution import difference

OUTPUT = Path('research/balloon-red-blue-20261008/source-resolved-negative-decision-20261011.json')


def summarize():
    if OUTPUT.exists():
        raise FileExistsError('Preserve correction decision evidence')
    report = read_json(AUDIT)
    for path, digest in report['inputs_sha256'].items():
        if sha256(path) != digest:
            raise ValueError('Evaluation dependency changed')
    if sha256(report['raw']['path']) != report['raw']['sha256']:
        raise ValueError('Scene predictions changed')
    for record in report['training'].values():
        root = Path(record['run'])
        for name, digest in record['artifacts_sha256'].items():
            if sha256(root/name) != digest:
                raise ValueError('Verified checkpoint changed')
        for path, digest in read_json(root/'provenance.json')['sources_sha256'].items():
            if sha256(path) != digest:
                raise ValueError('Training source changed')
    config = read_json('configs/direct-hybrid-balloon-20261011.json')
    if sha256(config['joint_audit']) != config['joint_audit_sha256']:
        raise ValueError('Incumbent reference changed')
    parent = read_json(config['joint_audit'])
    if sha256(parent['raw']['path']) != parent['raw']['sha256']:
        raise ValueError('Incumbent scene decisions changed')
    baseline, raw = read_json(parent['raw']['path']), read_json(report['raw']['path'])
    recounted, effects = {}, {}
    for panel, expected_frames in dict(original=64, window1=6, window2=12).items():
        variants = raw[panel]
        names = {f'{arm}{s}_{policy}' for arm in ('parent', 'control', 'corrected')
                 for s in (47, 48) for policy in ('current', 'pooled', 'base_first')}
        if set(variants) != names:
            raise ValueError('Incomplete matched controls or policies')
        recounted[panel], effects[panel] = {}, {}
        for name, result in variants.items():
            if len(result['frames']) != expected_frames:
                raise ValueError('Frame coverage changed')
            for frame in result['frames']:
                direct = not name.endswith('_current')
                if (not 0 <= frame['neural_calls'] <= 12 or len(frame['detections']) > (24 if direct else 12)
                        or (direct and frame['full_frame_neural_calls'] != 1)):
                    raise ValueError('Inference budget changed')
            fresh = screen(result, baseline[panel]['parent_parent47_current'])
            saved = {k: v for k, v in report['panels'][panel][name].items() if k != 'matched_control_effect'}
            if fresh != saved:
                raise ValueError('Independent metrics/retention recount differs')
            recounted[panel][name] = fresh
            if name.startswith('corrected'):
                change = difference(result, variants[name.replace('corrected', 'control', 1)])
                if change != report['panels'][panel][name]['matched_control_effect']:
                    raise ValueError('Paired correction effect differs')
                effects[panel][name] = dict(metrics=fresh['metrics'], against_control=change,
                    against_incumbent=fresh['effect'], errors=fresh['errors'])
    gates = eligible_policies(recounted)
    if gates != report['eligible_for_separate_export_development']:
        raise ValueError('Eligibility decision changed')
    write_json(OUTPUT, dict(audit_sha256=sha256(AUDIT), raw_sha256=report['raw']['sha256'],
        script_sha256=sha256(__file__), eligible_for_separate_export_development=gates,
        panels=effects, training=report['training'], decision=report['decision'],
        scope='Independent recount of18variants on82TRAIN scenes; four verified matched checkpoint artifacts. Three source-resolution reviewed background labels repeated4times are isolated from matched teacher retention on identical inputs. All238annotation-derived positive proposal labels remain in both arms. No independent accuracy, export, live-camera or flight-readiness claim.',
        development_evaluated=False, test_evaluated=False, pi_tested=False,
        deployment_approved=False, qualification_achieved=False))
    print(dict(eligible=gates), flush=True)


if __name__ == '__main__':
    summarize()
