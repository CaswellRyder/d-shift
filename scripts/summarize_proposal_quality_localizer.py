"""Independent recount, exact gate reconstruction and residual error origins."""
from collections import Counter
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from scripts.audit_proposal_quality_localizer import OUTPUT as AUDIT, DECODERS, eligible
from scripts.proposal_quality_features import gate
from scripts.audit_balanced_localizer import screen
from scripts.audit_direct_hybrid_balloon import checked_inputs
from scripts.diagnose_clutter_stage_attribution import difference
from scripts.audit_balloon_pipeline_errors import analyze_frame

OUTPUT = Path('research/balloon-red-blue-20261008/proposal-quality-localizer-decision-20261011.json')


def verify_gate(parent, quality):
    if any(parent[k] != quality[k] for k in ('id', 'source', 'truth', 'frame_sha256')):
        raise ValueError('Gate changed scene identity')
    rows = quality['detections']
    if gate(parent['detections'], [r.get('box_quality') for r in rows]) != rows:
        raise ValueError('Saved outputs do not reproduce the fixed quality gate')
    if (quality['neural_calls'] != parent['neural_calls'] or quality['full_frame_neural_calls'] != 1
            or not 0 <= quality['neural_calls'] <= 12 or len(rows) > 24
            or quality['quality_rows'] != sum(r['proposal_origin'] == 'learned' for r in rows)
            or quality['quality_rows'] > 12
            or quality['quality_head_calls'] != int(quality['quality_rows'] > 0)):
        raise ValueError('Inference work accounting changed')


def origins(result):
    counts = {c: {origin: Counter() for origin in ('mser', 'learned')}
              for c in ('red_balloon', 'blue_balloon')}
    for frame in result['frames']:
        for fp in analyze_frame(frame)['false_positives']:
            origin = frame['detections'][fp['index']]['proposal_origin']
            counts[fp['label']][origin][fp['reason']] += 1
    return {c: {o: dict(v) for o, v in groups.items()} for c, groups in counts.items()}


def summarize():
    if OUTPUT.exists():
        raise FileExistsError('Preserve independent decision')
    report = read_json(AUDIT)
    for path, digest in report['inputs_sha256'].items():
        if sha256(path) != digest:
            raise ValueError('Audit source changed')
    if sha256(report['raw']['path']) != report['raw']['sha256']:
        raise ValueError('Raw scene predictions changed')
    for record in report['training'].values():
        root = Path(record['run'])
        if sha256(root/'report.json') != record['report_sha256'] or read_json(root/'status.json')['state'] != 'complete':
            raise ValueError('Training report changed')
        for name, digest in record['artifacts_sha256'].items():
            if sha256(root/name) != digest:
                raise ValueError('Training artifact changed')
    raw = read_json(report['raw']['path'])
    _, refs, _, _ = checked_inputs()
    incumbent = read_json(refs['joint']['raw']['path'])
    panels, residuals = {}, {}
    names = {f'{arm}{seed}_{decoder}' for arm in ('parent', 'quality') for seed in (47, 48) for decoder in DECODERS}
    if set(raw) != {'original', 'window1', 'window2'}:
        raise ValueError('Incomplete panel coverage')
    for panel, count in dict(original=64, window1=6, window2=12).items():
        if set(raw[panel]) != names:
            raise ValueError('Missing comparison arms')
        panels[panel], residuals[panel] = {}, {}
        for name, result in raw[panel].items():
            if len(result['frames']) != count:
                raise ValueError('Scene coverage changed')
            fresh = screen(result, incumbent[panel]['parent_parent47_current'])
            if name.startswith('quality'):
                parent = raw[panel][name.replace('quality', 'parent')]
                for old, new in zip(parent['frames'], result['frames'], strict=True):
                    verify_gate(old, new)
                fresh['quality_effect'] = difference(result, parent)
                residuals[panel][name] = origins(result)
            if fresh != report['panels'][panel][name]:
                raise ValueError('Metrics, target identities or error counts do not recount')
            panels[panel][name] = fresh
    gates = {decoder: eligible(panels, decoder) for decoder in DECODERS}
    if gates != report['matched_head_gates']:
        raise ValueError('Promotion gate does not recount')
    write_json(OUTPUT, dict(audit_sha256=sha256(AUDIT), raw=report['raw'],
        inputs_sha256={str(p): sha256(p) for p in (__file__, 'scripts/proposal_quality_features.py',
            'scripts/audit_balloon_pipeline_errors.py', 'scripts/audit_balanced_localizer.py',
            'scripts/diagnose_clutter_stage_attribution.py')},
        panels=panels, residual_false_positives_by_origin=residuals, matched_head_gates=gates,
        decision='Reject promotion if either seed loses a parent target or increases FP; no threshold adjustment.',
        original_holdout_evaluated=False, development_evaluated=False, test_evaluated=False,
        pi_tested=False, deployment_approved=False, qualification_achieved=False))
    print(dict(gates=gates, residuals=residuals), flush=True)


if __name__ == '__main__':
    summarize()
