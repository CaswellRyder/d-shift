"""Attribute paired blue-object scene errors without changing decisions or training."""
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from scripts.audit_indoor5_outcomes import compare_frames
from scripts.evaluate_blue_objects_round2_trial import OUTPUT as DEVELOPMENT, RAW
from scripts.summarize_blue_objects_round2_trial import OUTPUT as DECISION
from scripts.summarize_public_balloon_trial import checked_metrics

OUTPUT = Path('research/balloon-red-blue-20261008/blue-objects-round2-errors-20261011.json')


def audit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    doc, raw, decision = map(read_json, (DEVELOPMENT, RAW, DECISION))
    if (doc['raw']['sha256'] != sha256(RAW)
            or decision['development_receipt_sha256'] != sha256(DEVELOPMENT)
            or doc['qualification_achieved'] is not False):
        raise ValueError('Frozen evaluation changed')
    panels = {}
    for panel, variants in raw.items():
        panels[panel] = {}
        for name, result in variants.items():
            if checked_metrics(result) != doc['panels'][panel]['results'][name]:
                raise ValueError('Stored metrics do not reproduce')
            parent = variants['parent/' + name.split('/')[1]]['frames']
            panels[panel][name] = dict(metrics=result['metrics'], **compare_frames(result['frames'], parent))
    write_json(OUTPUT, dict(panels=panels, development_sha256=sha256(DEVELOPMENT),
        decision_sha256=sha256(DECISION), raw_sha256=sha256(RAW), script_sha256=sha256(__file__),
        helpers_sha256={p: sha256(p) for p in ('scripts/audit_indoor5_outcomes.py',
            'scripts/audit_balloon_pipeline_errors.py', 'scripts/audit_current_blue_search.py',
            'scripts/summarize_public_balloon_trial.py')},
        scope='Repeated development error attribution. Truth-assisted geometric oracle is not deployable accuracy. No changes to training or runtime.',
        training_performed=False, test_evaluated=False, pi_tested=False,
        deployment_approved=False, qualification_achieved=False))
    for panel, variants in panels.items():
        print(panel, {k: v['colors']['blue_balloon'] for k, v in variants.items()
                      if k in ('parent/coarse47', 'admit47/coarse47', 'admit48/coarse47')})


if __name__ == '__main__':
    audit()
