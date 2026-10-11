"""Independently recount fixed center-presence trial decisions and paired effects."""
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from scripts.audit_center_presence_localizer import OUTPUT as AUDIT, POLICIES, DECODERS, eligible_policies
from scripts.audit_direct_hybrid_balloon import checked_inputs
from scripts.audit_balanced_localizer import screen
from scripts.diagnose_clutter_stage_attribution import difference

OUTPUT = Path('research/balloon-red-blue-20261008/center-presence-localizer-decision-20261011.json')


def summarize():
    if OUTPUT.exists():
        raise FileExistsError('Preserve independent recount')
    r = read_json(AUDIT)
    if any(sha256(path) != h for path, h in r['inputs_sha256'].items()):
        raise ValueError('Audited input changed')
    if sha256(r['raw']['path']) != r['raw']['sha256']:
        raise ValueError('Audited predictions changed')
    if sha256(r['cells']['path']) != r['cells']['sha256']:
        raise ValueError('Cell diagnostics changed')
    cells = read_json(r['cells']['path'])
    for t in r['training'].values():
        root = Path(t['run'])
        if read_json(root/'status.json')['state'] != 'complete':
            raise ValueError('Run no longer complete')
        if any(sha256(root/name) != h for name, h in t['artifacts_sha256'].items()):
            raise ValueError('Audited checkpoint changed')
        if any(sha256(path) != h for path, h in read_json(root/'provenance.json')['sources_sha256'].items()):
            raise ValueError('Training source changed')
    for c in r['classifiers'].values():
        if sha256(c['path']) != c['sha256']:
            raise ValueError('Frozen classifier changed')
    _, refs, _, _ = checked_inputs()
    baseline, raw = read_json(refs['joint']['raw']['path']), read_json(r['raw']['path'])
    panels, effects = {}, {}
    names = {f'{arm}{seed}_{decoder}_{policy}' for arm in ('parent', 'control', 'presence')
             for seed in (47, 48) for decoder in DECODERS for policy in POLICIES}
    if set(raw) != {'original', 'window1', 'window2'}:
        raise ValueError('Panel coverage changed')
    for panel, count in dict(original=64, window1=6, window2=12).items():
        if set(raw[panel]) != names:
            raise ValueError('Matched variants missing')
        panels[panel], effects[panel] = {}, {}
        for name, result in raw[panel].items():
            if len(result['frames']) != count:
                raise ValueError('Frame coverage changed')
            for f in result['frames']:
                direct, current = name.endswith('_direct'), name.endswith('_current')
                if (f['full_frame_neural_calls'] != 1 or not 0 <= f['neural_calls'] <= 12
                        or (direct and f['neural_calls'] != 0)
                        or len(f['detections']) > (12 if direct or current else 24)):
                    raise ValueError('Inference budget changed')
            fresh = screen(result, baseline[panel]['parent_parent47_current'])
            saved = {k: v for k, v in r['panels'][panel][name].items()
                     if k not in ('matched_control_effect', 'localizer_parent_effect', 'decoder_effect')}
            if fresh != saved:
                raise ValueError('Metrics, errors or retention do not recount')
            panels[panel][name] = fresh
            if '_independent_neighbor_' in name:
                changed = difference(result, raw[panel][name.replace('_independent_neighbor_', '_frozen_')])
                if changed != r['panels'][panel][name]['decoder_effect']:
                    raise ValueError('Decoder-only effect does not recount')
            if name.startswith('presence'):
                paired = difference(result, raw[panel][name.replace('presence', 'control', 1)])
                parent = difference(result, raw[panel][name.replace('presence', 'parent', 1)])
                if (paired != r['panels'][panel][name]['matched_control_effect'] or
                        parent != r['panels'][panel][name]['localizer_parent_effect']):
                    raise ValueError('Paired effects do not recount')
                effects[panel][name] = dict(metrics=fresh['metrics'], against_control=paired,
                    against_localizer_parent=parent, against_incumbent=fresh['effect'], errors=fresh['errors'])
        for name, frames in cells[panel].items():
            if [f['source'] for f in frames] != [f['source'] for f in raw[panel][name+'_frozen_current']['frames']]:
                raise ValueError('Cell diagnostic frame identities differ')
    gates = eligible_policies(panels)
    if gates != r['eligible_for_separate_export_development']:
        raise ValueError('Promotion decision changed')
    write_json(OUTPUT, dict(audit_sha256=sha256(AUDIT), raw_sha256=r['raw']['sha256'],
        script_sha256=sha256(__file__), eligible_for_separate_export_development=gates,
        panels=effects, training=r['training'], decision=r['decision'],
        scope='Independent metrics and paired-effect recount of48variants on82TRAIN frames. Four fixed-final checkpoints and frozen classifier hashes checked. No held-out or live qualification.',
        original_holdout_evaluated=False, development_evaluated=False, test_evaluated=False,
        pi_tested=False, deployment_approved=False, qualification_achieved=False))
    print(dict(eligible=gates), flush=True)


if __name__ == '__main__':
    summarize()
