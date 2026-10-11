"""Recount annotation categories and relate them to frozen TRAIN recall losses."""
from collections import Counter
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from dtr.tracking import iou
from scripts.balloon_scene_label_audit import OUTPUT as AUDIT, classify_geometry
from scripts.audit_balloon_gradient_projection import OUTPUT as PROJECTION
from scripts.diagnose_clutter_stage_attribution import difference
from scripts.summarize_public_balloon_trial import checked_metrics

OUTPUT = Path('research/balloon-red-blue-20261008/balloon-scene-label-decision-20261011.json')


def loss_context(lost, rows):
    supports = [r for r in rows if r['source'] == lost['source'] and r['selected']
                and r['teacher_decision'] == lost['target']['label']
                and iou(r['box'], lost['target']['box']) >= .5]
    if not supports:
        raise ValueError('Lost target has no original supporting detection')
    changed = [r['original']['category'].startswith('matched_')
               and r['wider']['category'] == 'partial_or_mixed' for r in supports]
    return dict(original_supports=len(supports), new_mixed_supports=sum(changed),
                all_supports_newly_mixed=all(changed))


def summarize():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    doc, projection = read_json(AUDIT), read_json(PROJECTION)
    if (any(sha256(p) != h for p, h in doc['sources_sha256'].items())
            or sha256(doc['detail']['path']) != doc['detail']['sha256']
            or sha256(projection['raw']['path']) != projection['raw']['sha256']):
        raise ValueError('Audit evidence changed')
    rows, scenes = read_json(doc['detail']['path']), read_json(projection['raw']['path'])
    counts, transitions = Counter(), Counter()
    for row in rows:
        old = classify_geometry(row['box'], row['crop_box'], row['truth'])
        new = classify_geometry(row['box'], row['wider_crop'], row['truth'])
        if old != row['original'] or new != row['wider']:
            raise ValueError('Geometry categories do not reproduce')
        for mode, geo in (('original', old), ('wider', new)):
            counts[f'{mode}/{geo["category"]}/{row["teacher_decision"]}'] += 1
        transitions[old['category'] + ' -> ' + new['category']] += 1
    if dict(counts) != doc['counts'] or dict(transitions) != doc['geometry_transitions']:
        raise ValueError('Annotation summary does not reproduce')
    checked_metrics(scenes['parent'])
    results = {}
    for name, result in projection['results'].items():
        effect = difference(scenes[name], scenes['parent'])
        if effect != result['effect']:
            raise ValueError('Frozen losses changed')
        contexts = [dict(**lost, **loss_context(lost, rows)) for lost in effect['lost_targets']]
        results[name] = dict(lost=len(contexts),
            losses_with_all_original_supports_newly_mixed=sum(r['all_supports_newly_mixed'] for r in contexts),
            losses_with_any_original_support_newly_mixed=sum(r['new_mixed_supports'] > 0 for r in contexts))
    write_json(OUTPUT, dict(audit_sha256=sha256(AUDIT), projection_sha256=sha256(PROJECTION),
        script_sha256=sha256(__file__), context_loss_association=results,
        counts=dict(counts), geometry_transitions=dict(transitions),
        interpretation='Wider context introduces competing annotated colors in39previously unambiguous proposals. Original teacher accepts7red proposals with no annotated balloon in their crops. These are separate diagnostic issues, not proof of causal attribution or approved new labels.',
        next_step='Review annotation-clean original TRAIN proposals for explicit scene supervision; retain partial/mixed crops as ambiguous rather than manufacturing background labels. Do not repeat unchanged wider-context fitting or promote failed models.',
        scope='TRAIN diagnosis only. Geometric overlap is not visual confirmation; association is not causation. Held-out94%per-color precision/recall and separate livePi remain unmet.',
        labels_created=False, development_evaluated=False, test_evaluated=False,
        pi_tested=False, deployment_approved=False, qualification_achieved=False))
    print(results, flush=True)


if __name__ == '__main__':
    summarize()
