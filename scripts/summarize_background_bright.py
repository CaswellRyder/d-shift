"""Recount paired localizer integration and require retained targets before Pi work."""
from dtr.data import read_json, sha256, write_json
from scripts.audit_indoor5_outcomes import compare_frames
from scripts.evaluate_background_bright import BASE, PROTOCOL
from scripts.evaluate_blue_scene_localizer import nonregression
from scripts.summarize_public_balloon_trial import checked_metrics


def summarize():
    out = BASE/'background-bright-decision-20261011.json'
    if out.exists():
        raise FileExistsError(out)
    stages,inputs = {},{}
    for stage in ('training','development'):
        path = BASE/f'background-bright-{stage}-20261011.json'
        doc = read_json(path)
        if (sha256(doc['raw']['path']) != doc['raw']['sha256'] or doc['protocol_sha256'] != sha256(PROTOCOL)
                or doc['script_sha256'] != sha256('scripts/evaluate_background_bright.py')
                or any(sha256(p) != h for p,h in doc['helpers_sha256'].items())
                or any(sha256(p) != h for p,h in doc['inputs'].items())):
            raise ValueError('Changed experiment inputs')
        for panel,variants in read_json(doc['raw']['path']).items():
            for name,result in variants.items():
                saved = doc['panels'][panel][name]
                compared = compare_frames(result['frames'],variants['parent']['frames'])
                if (checked_metrics(result) != saved['metrics'] or any(saved[k] != v for k,v in compared.items())
                        or nonregression(variants['parent']['metrics'],result['metrics']) != saved['nonregression']):
                    raise ValueError('Metrics or target identities do not recount')
        stages[stage] = doc['panels']
        inputs[str(path)] = sha256(path)
    candidates = {name:dict(eligible_for_pi_comparison=all(
        panel[name]['nonregression'] and not panel[name]['lost_parent_targets']
        for stage in stages.values() for panel in stage.values())) for name in ('background47','background48')}
    write_json(out,dict(stages=stages,candidates=candidates,inputs=inputs,script_sha256=sha256(__file__),
        decision='Only eligible candidates may receive isolated Pi timing; no deployment approval',
        interpretation='Earlier full-search development improvement did not survive lower-cost search unchanged. Complete TRAIN retention is also required; crop/replay evidence is not independent qualification.',
        test_evaluated=False,pi_tested=False,deployment_approved=False,qualification_achieved=False))
    print(candidates)


if __name__ == '__main__':
    summarize()
