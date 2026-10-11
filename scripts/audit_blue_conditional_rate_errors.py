"""Attribute new correction failures on frozen development predictions, not training."""
from collections import Counter
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from scripts.audit_balloon_pipeline_errors import analyze_frame, maximum_matching
from scripts.audit_current_blue_search import matched_truth
from scripts.evaluate_blue_conditional_rate import OUTPUT as DEVELOPMENT, RAW
from scripts.summarize_blue_conditional_rate import OUTPUT as DECISION
from scripts.summarize_public_balloon_trial import checked_metrics

OUTPUT = Path('research/balloon-red-blue-20261008/blue-conditional-rate-errors-20261011.json')


def accepted_red(frame):
    return sorted(tuple(d['box']) for d in frame['detections']
                  if d['accepted'] and d['label'] == 'red_balloon')


def audit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    doc, raw, decision = map(read_json, (DEVELOPMENT, RAW, DECISION))
    if (doc['raw']['sha256'] != sha256(RAW)
            or decision['development_receipt_sha256'] != sha256(DEVELOPMENT)):
        raise ValueError('Changed frozen evaluation')
    panels = {}
    for panel, variants in raw.items():
        results = {}
        for name, result in variants.items():
            if checked_metrics(result) != doc['panels'][panel]['results'][name]:
                raise ValueError('Metrics do not reproduce')
            frames = result['frames']
            parent = variants['parent/' + name.split('/')[1]]['frames']
            rows = [analyze_frame(f) for f in frames]
            colors = {}
            for label in ('red_balloon', 'blue_balloon'):
                total = sum(t['label'] == label for f in frames for t in f['truth'])
                covered = sum(len(maximum_matching([t for t in f['truth'] if t['label'] == label],
                                                  f['detections'], lambda t, d: True)) for f in frames)
                colors[label] = dict(targets=total, geometric_oracle_covered=covered,
                    geometric_oracle_recall=covered / total if total else None,
                    errors={kind: dict(Counter(e['reason'] for r in rows for e in r[kind]
                                              if e['label'] == label))
                            for kind in ('false_positives', 'false_negatives')})
            red_unchanged, lost, gained = True, [], []
            for row, base in zip(frames, parent, strict=True):
                if any(row[k] != base[k] for k in ('id', 'source', 'frame_sha256', 'truth')):
                    raise ValueError('Comparison identity changed')
                red_unchanged &= accepted_red(row) == accepted_red(base)
                for destination, ids in ((lost, matched_truth(base)-matched_truth(row)),
                                         (gained, matched_truth(row)-matched_truth(base))):
                    destination.extend(dict(source=row['source'], target=row['truth'][i]) for i in sorted(ids))
            results[name] = dict(metrics=result['metrics'], colors=colors,
                accepted_red_boxes_identical_to_parent=red_unchanged,
                lost_parent_targets=lost, gained_targets=gained)
        panels[panel] = results
    write_json(OUTPUT, dict(panels=panels, development_sha256=sha256(DEVELOPMENT),
        decision_sha256=sha256(DECISION), raw_sha256=sha256(RAW), script_sha256=sha256(__file__),
        helpers_sha256={p: sha256(p) for p in ('scripts/audit_balloon_pipeline_errors.py',
            'scripts/audit_current_blue_search.py', 'scripts/summarize_public_balloon_trial.py')},
        scope='Repeated related development. Truth-assisted geometric oracle is not attainable model accuracy. No changes to training or deployment.',
        test_evaluated=False, pi_tested=False, training_performed=False,
        deployment_approved=False, qualification_achieved=False))
    for panel, variants in panels.items():
        print(panel, {name: dict(blue=r['colors']['blue_balloon'], red_boxes_unchanged=r['accepted_red_boxes_identical_to_parent'])
                      for name, r in variants.items() if name in ('parent/coarse47', 'admit47/coarse47', 'admit48/coarse47')})


if __name__ == '__main__':
    audit()
