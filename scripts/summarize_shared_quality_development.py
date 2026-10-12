"""Independently recount repeated-development failures without fitting or promotion."""
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from scripts.evaluate_shared_quality_development import OUTPUT as AUDIT, PROTOCOL
from scripts.shared_proposal_quality import gate
from scripts.audit_source_aware_balloon import aggregate
from scripts.audit_balanced_localizer import screen
from scripts.diagnose_clutter_stage_attribution import difference
from scripts.summarize_proposal_quality_localizer import origins
from scripts.summarize_shared_quality import meets94

OUTPUT = Path('research/balloon-red-blue-20261008/shared-quality-development-decision-20261011.json')


def verify_frame(parent, quality):
    if any(parent[k] != quality[k] for k in ('id', 'source', 'frame_sha256', 'truth', 'vertical_flip')):
        raise ValueError('Development frame identity changed')
    q = [r.get('box_quality') for r in quality['detections']]
    if gate(parent['detections'], q) != quality['detections']:
        raise ValueError('Recorded quality decisions do not reproduce')
    n = sum(r['accepted'] for r in parent['detections'])
    if (quality['quality_rows'] != n or quality['quality_head_calls'] != int(n > 0)
            or quality['neural_calls'] != parent['neural_calls'] or quality['full_frame_neural_calls'] != 1
            or not 0 <= quality['neural_calls'] <= 12 or len(quality['detections']) > 24):
        raise ValueError('Inference accounting changed')


def summarize():
    if OUTPUT.exists():
        raise FileExistsError('Preserve evaluation decision')
    report, p = read_json(AUDIT), read_json(PROTOCOL)
    if report['protocol_sha256'] != sha256(PROTOCOL) or sha256(report['raw']['path']) != report['raw']['sha256']:
        raise ValueError('Evaluation identity changed')
    for path, digest in report['inputs_sha256'].items():
        if sha256(path) != digest:
            raise ValueError('Evaluation source changed')
    for candidate in report['candidates'].values():
        for artifact in candidate.values():
            if sha256(artifact['path']) != artifact['sha256']:
                raise ValueError('Frozen candidate changed')
    raw = read_json(report['raw']['path'])
    names = {'parent47', 'parent48', 'quality47', 'quality48', 'incumbent'}
    if set(raw) != set(p['panels']) or report['train_parity'] != dict(original=64, window1=6, window2=12):
        raise ValueError('Evaluation or TRAIN parity coverage changed')
    panels, attribution = {}, {}
    for panel, count in p['panels'].items():
        if set(raw[panel]) != names:
            raise ValueError('Missing frozen comparison arms')
        panels[panel], attribution[panel] = {}, {}
        for name, result in raw[panel].items():
            if len(result['frames']) != count or any('IMG_' not in f['source'] for f in result['frames']):
                raise ValueError('Reviewed development coverage changed')
            fresh = screen(result, raw[panel]['incumbent'])
            if name.startswith('quality'):
                parent = raw[panel][name.replace('quality', 'parent')]
                for a, b in zip(parent['frames'], result['frames'], strict=True):
                    verify_frame(a, b)
                fresh['quality_effect'] = difference(result, parent)
            if fresh != report['panels'][panel][name]:
                raise ValueError('Metrics or matched target identities do not recount')
            panels[panel][name] = fresh
            attribution[panel][name] = origins(result) if name != 'incumbent' else None
    pooled = {name: aggregate([f for panel in raw.values() for f in panel[name]['frames']]) for name in names}
    effects = {f'quality{s}': difference(pooled[f'quality{s}'], pooled[f'parent{s}']) for s in (47, 48)}
    every_panel = {f'quality{s}': all(meets94(panel[f'quality{s}']['metrics']) for panel in panels.values()) for s in (47, 48)}
    if every_panel != report['meets94_every_development_panel']:
        raise ValueError('Accuracy gate does not recount')
    write_json(OUTPUT, dict(audit_sha256=sha256(AUDIT), raw=report['raw'], protocol_sha256=sha256(PROTOCOL),
        inputs_sha256={path: sha256(path) for path in (__file__, 'scripts/shared_proposal_quality.py',
            'scripts/audit_source_aware_balloon.py', 'scripts/audit_balanced_localizer.py',
            'scripts/diagnose_clutter_stage_attribution.py', 'scripts/summarize_proposal_quality_localizer.py')},
        panels=panels, pooled_metrics={n: r['metrics'] for n, r in pooled.items()},
        pooled_quality_effects=effects, residual_false_positives_by_origin=attribution,
        meets94_every_development_panel=every_panel,
        pooled_meets94={name: meets94(result['metrics']) for name, result in pooled.items()},
        decision='Do not promote either candidate. TRAIN gains fail to generalize; preserve the existing deployment and do not tune thresholds against these evaluation-only labels.',
        next_research='Improve TRAIN-domain diversity and source-grouped validation before further frozen development evaluation. Isolate red-path regressions from blue precision work; do not add IMG development crops or labels to training, calibration or negative mining.',
        scope='Repeated related development, not independent competition qualification. No training or runtime changes.',
        training_performed=False, development_evaluated=True, test_evaluated=False,
        independent_recording_sessions_verified=False, pi_tested=False, deployment_approved=False, qualification_achieved=False))
    print(dict(pooled_metrics={n: r['metrics'] for n, r in pooled.items()},
        quality_target_losses={n: len(e['lost_targets']) for n, e in effects.items()}, qualifies=every_panel), flush=True)


if __name__ == '__main__':
    summarize()
