"""Recompute saved TRAIN/development evidence and attribute new search errors."""
from collections import Counter
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from scripts.develop_censored_ellipse import OUTPUT as DEVELOPMENT, RAW, PANELS
from scripts.evaluate_censored_ellipse import OUTPUT as TRAIN, RAW as TRAIN_RAW
from scripts.evaluate_reviewed_balloon_scenes import reviewed_frames
from scripts.audit_balloon_false_positives import classify_errors
from scripts.audit_current_blue_search import matched_truth
from scripts.summarize_public_balloon_trial import checked_metrics

OUTPUT = Path('research/balloon-red-blue-20261008/censored-ellipse-audit-20261011.json')


def audit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    train, development, raw = read_json(TRAIN), read_json(DEVELOPMENT), read_json(RAW)
    if (sha256(TRAIN_RAW) != train['raw']['sha256'] or sha256(RAW) != development['raw']['sha256']
            or sha256(TRAIN) != development['training_receipt_sha256']
            or sha256('scripts/develop_censored_ellipse.py') != development['script_sha256']
            or any(sha256(p) != h for p,h in {**train['scripts_sha256'],**train['helpers_sha256']}.items())
            or checked_metrics(read_json(TRAIN_RAW)) != train['metrics']):
        raise ValueError('Evidence changed')
    panels = {}
    for panel in PANELS:
        sources, review = reviewed_frames('data/'+panel,'configs/'+panel+'.json')
        refs = {r['source']:r for r,_ in sources}
        saved = development['panels'][panel]
        if review['queue_sha256'] != saved['queue_sha256'] or sha256('configs/'+panel+'.json') != saved['review_sha256']:
            raise ValueError('Development review changed')
        for mode in ('baseline','candidate'):
            result = raw[panel][mode]
            if checked_metrics(result) != saved['results'][mode] or len(result['frames']) != len(refs):
                raise ValueError('Metrics/frame count changed')
            if {r['source'] for r in result['frames']} != set(refs):
                raise ValueError('Source coverage changed')
            for r in result['frames']:
                ref = refs[r['source']]
                if r['truth'] != ref['provisional_truth'] or r['frame_sha256'] != ref['sha256']:
                    raise ValueError('Scene identity changed')
        errors, losses, gains = [], [], []
        for r,base in zip(raw[panel]['candidate']['frames'],raw[panel]['baseline']['frames'],strict=True):
            if r['source'] != base['source']:
                raise ValueError('Frame order differs')
            errors.extend(dict(e,source=r['source']) for e in classify_errors(r,base))
            old,new = matched_truth(base),matched_truth(r)
            for dest,indices in ((losses,old-new),(gains,new-old)):
                dest.extend(dict(source=r['source'],target=r['truth'][i]) for i in sorted(indices))
        if losses != saved['lost_targets'] or gains != saved['gained_targets']:
            raise ValueError('Target identity accounting changed')
        counts = Counter('/'.join((e['label'],e['origin'],e['reason'])) for e in errors)
        panels[panel] = dict(error_counts=dict(counts),errors=errors,gained=len(gains),lost=len(losses))
    write_json(OUTPUT,dict(panels=panels,training_receipt_sha256=sha256(TRAIN),
        development_receipt_sha256=sha256(DEVELOPMENT),script_sha256=sha256(__file__),
        helper_sha256=sha256('scripts/audit_balloon_false_positives.py'),
        decision='Reject candidate. Training gain did not generalize; Pi left unchanged.',
        test_evaluated=False,deployment_approved=False,qualification_achieved=False))
    print({p:dict(error_counts=r['error_counts'],gained=r['gained'],lost=r['lost']) for p,r in panels.items()})


if __name__ == '__main__':
    audit()
