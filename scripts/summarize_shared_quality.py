"""Recount shared-source gating and freeze the next research evaluation candidates."""
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from scripts.audit_shared_quality import OUTPUT as AUDIT, DECODERS, eligible
from scripts.shared_proposal_quality import gate
from scripts.audit_source_aware_balloon import aggregate
from scripts.audit_balanced_localizer import screen
from scripts.audit_direct_hybrid_balloon import checked_inputs
from scripts.diagnose_clutter_stage_attribution import difference
from scripts.summarize_proposal_quality_localizer import origins

OUTPUT = Path('research/balloon-red-blue-20261008/shared-proposal-quality-decision-20261011.json')


def verify_gate(parent, result):
    if any(parent[k] != result[k] for k in ('id', 'source', 'frame_sha256', 'truth')):
        raise ValueError('Gate changed source identity')
    qualities = [r.get('box_quality') for r in result['detections']]
    if gate(parent['detections'], qualities) != result['detections']:
        raise ValueError('Stored detections do not reproduce the gate')
    accepted = sum(r['accepted'] for r in parent['detections'])
    if (result['neural_calls'] != parent['neural_calls'] or not 0 <= result['neural_calls'] <= 12
            or result['full_frame_neural_calls'] != 1 or len(result['detections']) > 24
            or result['quality_head_calls'] != int(accepted > 0) or result['quality_rows'] != accepted):
        raise ValueError('Work accounting changed')


def meets94(metrics):
    # Compare unrounded fractions, not displayed percentages.
    return all(m['precision'] >= .94 and m['recall'] >= .94 for m in metrics.values())


def summarize():
    if OUTPUT.exists():
        raise FileExistsError('Preserve independent recount')
    report = read_json(AUDIT)
    for path, digest in report['inputs_sha256'].items():
        if sha256(path) != digest:
            raise ValueError('Audited source changed')
    if sha256(report['raw']['path']) != report['raw']['sha256']:
        raise ValueError('Audited scene output changed')
    for record in report['training'].values():
        root = Path(record['run'])
        if read_json(root/'status.json')['state'] != 'complete' or sha256(root/'report.json') != record['report_sha256']:
            raise ValueError('Training receipt changed')
        for name, digest in record['artifacts_sha256'].items():
            if sha256(root/name) != digest:
                raise ValueError('Model artifact changed')
    raw = read_json(report['raw']['path'])
    _, refs, _, _ = checked_inputs()
    incumbent = read_json(refs['joint']['raw']['path'])
    names = {f'{arm}{seed}_{decoder}' for arm in ('parent', 'iou', 'valid') for seed in (47, 48) for decoder in DECODERS}
    if set(raw) != {'original', 'window1', 'window2'}:
        raise ValueError('Incomplete TRAIN coverage')
    panels = {}
    for panel, count in dict(original=64, window1=6, window2=12).items():
        if set(raw[panel]) != names:
            raise ValueError('Incomplete arms')
        panels[panel] = {}
        for name, result in raw[panel].items():
            if len(result['frames']) != count:
                raise ValueError('Frame coverage changed')
            fresh = screen(result, incumbent[panel]['parent_parent47_current'])
            if not name.startswith('parent'):
                arm = 'iou' if name.startswith('iou') else 'valid'
                parent = raw[panel][name.replace(arm, 'parent', 1)]
                for a, b in zip(parent['frames'], result['frames'], strict=True):
                    verify_gate(a, b)
                fresh['quality_effect'] = difference(result, parent)
                fresh['residual_origins'] = origins(result)
                if arm == 'valid':
                    fresh['objective_effect'] = difference(result, raw[panel][name.replace('valid', 'iou', 1)])
            if fresh != report['panels'][panel][name]:
                raise ValueError('Counts, target retention or error origins do not recount')
            panels[panel][name] = fresh
    gates = {f'{arm}_{decoder}': eligible(panels, arm, decoder) for arm in ('iou', 'valid') for decoder in DECODERS}
    if gates != report['matched_head_gates']:
        raise ValueError('Advancement gates do not recount')
    pooled = {name: aggregate([f for panel in raw.values() for f in panel[name]['frames']])['metrics'] for name in sorted(names)}
    # Selection is explicit, TRAIN-only, and made before any new evaluation data.
    selected = ['iou47_frozen', 'iou48_frozen']
    if not gates['iou_frozen'] or not all(meets94(pooled[name]) for name in selected):
        raise ValueError('Candidate rationale is not supported by unrounded TRAIN metrics')
    write_json(OUTPUT, dict(audit_sha256=sha256(AUDIT), raw=report['raw'],
        inputs_sha256={str(path): sha256(path) for path in (__file__, 'scripts/shared_proposal_quality.py',
            'scripts/audit_source_aware_balloon.py', 'scripts/audit_balanced_localizer.py',
            'scripts/summarize_proposal_quality_localizer.py', 'scripts/diagnose_clutter_stage_attribution.py')},
        panels=panels, pooled_train_metrics=pooled, matched_head_gates=gates,
        per_panel_train_meets94={panel: {name: meets94(r['metrics']) for name, r in rows.items()} for panel, rows in panels.items()},
        selected_for_next_evaluation=selected,
        selection_reason='Shared IoU heads with frozen peak decoding meet pooled TRAIN94percent in both seeds and preserve every corresponding parent target. Public-panel recall still fails; this is a research evaluation selection, not deployment or final qualification. Freeze both seeds for separate evaluation before further fitting.',
        scope='Independent recount of82TRAIN scenes. Pooled metrics do not erase weaker panel results. No independent heldout accuracy evidence.',
        original_holdout_evaluated=False, development_evaluated=False, test_evaluated=False,
        pi_tested=False, deployment_approved=False, qualification_achieved=False))
    print(dict(selected=selected, pooled={name: pooled[name] for name in selected}, gates=gates), flush=True)


if __name__ == '__main__':
    summarize()
