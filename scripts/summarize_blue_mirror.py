"""Recount the frozen mirror ablation; never interpret repeated panels as qualification."""
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from scripts.audit_indoor5_outcomes import compare_frames
from scripts.evaluate_blue_mirror import BASE, PROTOCOL
from scripts.summarize_public_balloon_trial import checked_metrics


def summarize():
    out = BASE/'blue-mirror-decision-20261011.json'
    if out.exists():
        raise FileExistsError(out)
    inputs, stages = {}, {}
    for stage in ('training','development'):
        path = BASE/f'blue-mirror-{stage}-20261011.json'
        report = read_json(path)
        raw = Path(report['raw']['path'])
        if (report['raw']['sha256'] != sha256(raw)
                or report['protocol_sha256'] != sha256(PROTOCOL)
                or report['script_sha256'] != sha256('scripts/evaluate_blue_mirror.py')
                or any(sha256(p) != h for p,h in report['helpers_sha256'].items())
                or any(sha256(p) != h for p,h in report['inputs'].items())):
            raise ValueError('Changed experiment input')
        panels = read_json(raw)
        for panel, arms in panels.items():
            saved = report['panels'][panel]
            for name in ('parent','mirror'):
                if checked_metrics(arms[name]) != saved[name]:
                    raise ValueError('Counts do not reproduce')
            compared = compare_frames(arms['mirror']['frames'],arms['parent']['frames'])
            if any(saved[k] != v for k,v in compared.items()):
                raise ValueError('Target identity comparison changed')
            for new,old in zip(arms['mirror']['frames'],arms['parent']['frames'],strict=True):
                calls = 0
                for d,b in zip(new['detections'],old['detections'],strict=True):
                    if b['accepted'] and b['label'] == 'blue_balloon':
                        calls += 1
                        probe = d['blue_mirror']
                        keep = probe['accepted'] and probe['label'] == 'blue_balloon'
                        if d['accepted'] != keep or probe['retained'] != keep:
                            raise ValueError('Gate does not reproduce')
                        if {k:v for k,v in d.items() if k not in ('accepted','blue_mirror')} != {k:v for k,v in b.items() if k != 'accepted'}:
                            raise ValueError('Gate changed original scores or geometry')
                    elif d != b:
                        raise ValueError('Gate changed nonselected-blue detection')
                if new['extra_mirror_calls'] != calls or new['neural_calls'] != old['neural_calls']+calls:
                    raise ValueError('Workload does not recount')
        stages[stage] = report['panels']
        inputs[str(path)] = sha256(path)
    passed = all(p['nonregression'] and not p['lost_parent_targets'] for s in stages.values() for p in s.values())
    write_json(out,dict(stages=stages,inputs=inputs,script_sha256=sha256(__file__),
        eligible_for_pi_timing=passed,
        decision='Candidate for timing only' if passed else 'Reject runtime mirror gate: loses real targets while adding neural calls.',
        interpretation='TRAIN crop sensitivity did not generalize to full-scene recall. Retain current lower-cost pipeline; do not deploy this gate or infer Pi FPS from call count.',
        training_performed=False,pi_tested=False,deployment_approved=False,test_evaluated=False,qualification_achieved=False))
    print(dict(eligible_for_pi_timing=passed,output=str(out)))


if __name__ == '__main__':
    summarize()
