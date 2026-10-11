"""Record TRAIN-only center/geometry mismatch without changing selection rules."""
from collections import Counter
from pathlib import Path

from dtr.data import sha256, write_json
from scripts.audit_geometry_adapter import checked_run, TRIAL
from scripts.audit_balloon_dense_proposals import decode
from scripts.diagnose_grid_blue_training import center_probe
from scripts.audit_balloon_pipeline_errors import maximum_matching

OUTPUT = Path('research/balloon-red-blue-20261008/geometry-adapter-diagnosis-20261011.json')


def diagnose():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    results = {}
    for seed in (47, 48):
        _, _, probe, _, report, arrays = checked_run('admit', seed)
        for point, values in arrays.items():
            count, rows = Counter(), []
            for index, row in enumerate(probe['records']):
                if index < 129:
                    continue
                matched = maximum_matching(row['truth'], decode(values[index]), lambda t, d: d['color_group'] == 1)
                for j, target in enumerate(row['truth']):
                    center = center_probe(values[index], target['box'])
                    count['targets'] += 1
                    count['center_geometry_ok'] += center['iou'] >= .5
                    count['center_peak'] += center['local_peak']
                    count['decoded_matched'] += j in matched
                    count['good_center_not_peak_and_unmatched'] += center['iou'] >= .5 and not center['local_peak'] and j not in matched
                    count['bad_center_geometry_and_unmatched'] += center['iou'] < .5 and j not in matched
                    rows.append(dict(source=row['source'], target=target, center=center, matched=j in matched))
            results[f'admit{seed}/{point}'] = dict(counts=dict(count), rows=rows, report=report)
    write_json(OUTPUT, dict(results=results, trial_sha256=sha256(TRIAL), script_sha256=sha256(__file__),
        helpers_sha256={p: sha256(p) for p in ('scripts/audit_geometry_adapter.py',
            'scripts/diagnose_grid_blue_training.py',
            'scripts/audit_balloon_pipeline_errors.py')},
        scope='Known TRAIN centers and decoded matching only. Diagnostic association, not a causal ablation. No development pixels or qualification.',
        test_evaluated=False, deployment_approved=False, qualification_achieved=False))
    print({name: row['counts'] for name, row in results.items()})


if __name__ == '__main__':
    diagnose()
