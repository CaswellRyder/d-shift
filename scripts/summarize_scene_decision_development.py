"""Independently recount decision-retention development results; never qualify flight."""
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from scripts.evaluate_scene_decision_retention import OUTPUT as SOURCE, BASELINE
from scripts.summarize_public_balloon_trial import checked_metrics
from scripts.diagnose_clutter_stage_attribution import difference

OUTPUT = Path('research/balloon-red-blue-20261008/scene-decision-retention-development-decision-20261011.json')


def summarize():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    doc = read_json(SOURCE)
    if (sha256(doc['raw']['path']) != doc['raw']['sha256']
            or sha256('scripts/evaluate_scene_decision_retention.py') != doc['script_sha256']
            or sha256(BASELINE) != doc['baseline_sha256']
            or any(sha256(p) != h for p, h in {**doc['helpers_sha256'], **doc['inputs']}.items())
            or any(doc[k] is not False for k in ('test_evaluated', 'pi_tested', 'deployment_approved', 'qualification_achieved'))):
        raise ValueError('Development evidence or scope changed')
    raw = read_json(doc['raw']['path'])
    expected = {'parent', 'control47', 'control48', 'decision47', 'decision48'}
    if len(raw) != 4 or 'training' not in raw or any(set(v) != expected for v in raw.values()):
        raise ValueError('Missing complete paired evaluation')
    for row in doc['exports'].values():
        path = Path(row['path'])
        if sha256(path) != row['sha256'] or sha256(path.with_suffix('.json')) != row['metadata_sha256']:
            raise ValueError('Export changed')
    panels = {}
    survivors = {'decision47', 'decision48'}
    for panel, variants in raw.items():
        panels[panel] = {}
        for name, result in variants.items():
            m, effect = checked_metrics(result), difference(result, variants['parent'])
            gate = not effect['lost_targets'] and all(r['fp'] <= 0 and r['tp'] >= 0 for r in effect['count_delta'].values())
            meets = all(r[k] >= .94 for r in m.values() for k in ('precision', 'recall'))
            original = doc['panels'][panel][name]
            if (m != original['metrics'] or gate != original['nonregression'] or meets != original['meets_94']
                    or effect['lost_targets'] != original['comparison']['lost_parent_targets']
                    or effect['gained_targets'] != original['comparison']['gained_targets']):
                raise ValueError('Scene outcome does not reproduce')
            if not gate:
                survivors.discard(name)
            panels[panel][name] = dict(metrics=m, lost=len(effect['lost_targets']), gained=len(effect['gained_targets']),
                                       nonregression=gate, meets_94=meets)
            if name.startswith('decision'):
                panels[panel][name]['against_matched_control'] = difference(result, variants['control'+name[-2:]])['count_delta']
    eligible = ['decision47', 'decision48'] if len(survivors) == 2 else []
    if eligible != doc['eligible_for_pi_comparison']:
        raise ValueError('Paired progression decision changed')
    write_json(OUTPUT, dict(source_sha256=sha256(SOURCE), script_sha256=sha256(__file__), panels=panels,
        eligible_for_pi_comparison=eligible,
        decision='Eligible for separate isolated Pi comparison, not qualification.' if eligible else 'Reject deployment: training improvements did not pass paired development nonregression.',
        scope='64TRAIN scenes and three related repeatedly viewed development panels. Repeated development is not independent final qualification. No reserved test access, Pi execution or model replacement.',
        test_evaluated=False, pi_tested=False, deployment_approved=False, qualification_achieved=False))
    print(dict(eligible_for_pi_comparison=eligible, panels=panels), flush=True)


if __name__ == '__main__':
    summarize()
