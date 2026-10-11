"""Verify explicit visual labels and identify conflicting old teacher retention rows."""
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from scripts.prepare_direct_hybrid_negatives import ROOT, AUDIT, AUDIT_SHA, clear_of_targets
from scripts.prepare_openimages_balloon_review import checked_path
from scripts.balloon_scene_constraints import RAW as SCENES, RAW_SHA

REVIEW = Path('configs/direct-hybrid-negative-review-20261011.json')
OUTPUT = Path('research/balloon-red-blue-20261008/direct-hybrid-negative-admission-20261011.json')


def checked_extra():
    review = read_json(REVIEW)
    if (review['queue'] != str(ROOT) or sha256(ROOT/'review.json') != review['queue_sha256']
            or review['training_approved'] is not True or any(review[k] is not False
            for k in ('development_evaluated', 'test_evaluated', 'deployment_approved', 'qualification_achieved'))):
        raise ValueError('Visual review identity or TRAIN scope changed')
    queue = read_json(ROOT/'review.json')
    if (queue['audit_sha256'] != AUDIT_SHA or sha256(AUDIT) != AUDIT_SHA
            or sha256(ROOT/'sheet.png') != queue['sheet_sha256']
            or sha256('scripts/prepare_direct_hybrid_negatives.py') != queue['script_sha256']):
        raise ValueError('Review source or contact sheet changed')
    audit = read_json(AUDIT)
    if sha256(audit['raw']['path']) != queue['raw_sha256']:
        raise ValueError('Mined predictions changed')
    ids = {str(s['id']) for s in queue['samples']}
    accepted, excluded = set(review['labels']), set(review['excluded'])
    if (len(ids) != 14 or len(accepted) != 8 or accepted & excluded or accepted | excluded != ids
            or set(review['review_notes']) != accepted or set(review['labels'].values()) != {'background'}
            or any(not v for v in [*review['review_notes'].values(), *review['excluded'].values()])):
        raise ValueError('Require an explicit visual decision for every candidate')
    rows = []
    for sample in queue['samples']:
        if (sample['split'] != 'train' or 'IMG_' in sample['source'] or not
                sample['source'].startswith(('train/', 'openimages/train/'))
                or not clear_of_targets(sample['crop_box'], sample['truth'])):
            raise ValueError('Heldout, unknown or target-overlapping negative')
        checked_path(ROOT, sample['frame_path'], sample['rendered_frame_sha256'])
        path = checked_path(ROOT, sample['crop_path'], sample['crop_sha256'])
        if str(sample['id']) in accepted:
            rows.append(dict(sample, label='background', path=str(path)))
    if len({r['source'] for r in rows}) != 8:
        raise ValueError('Expected eight distinct reviewed scenes')
    return rows


def admit():
    if OUTPUT.exists():
        raise FileExistsError('Preserve admission receipt')
    rows = checked_extra()
    if sha256(SCENES) != RAW_SHA:
        raise ValueError('Old teacher scene evidence changed')
    frames = read_json(SCENES)['training']['parent']['frames']
    lookup = {(r['source'], tuple(r['crop_box'])): r for r in rows}
    conflicts, index = [], 0
    for f in frames:
        for proposal_index, d in enumerate(f['detections']):
            reviewed = lookup.get((f['source'], tuple(d['crop_box'])))
            if reviewed is not None:
                if f['frame_sha256'] != reviewed['frame_sha256']:
                    raise ValueError('Reviewed frame identity differs from teacher scene')
                conflicts.append(dict(scene_row=index, proposal_index=proposal_index,
                    reviewed_id=reviewed['id'], source=f['source'], crop_box=d['crop_box'],
                    teacher_label=d['label'], teacher_raw_accepted=d['raw_accepted'],
                    teacher_scores=d['scores'], reviewed_label='background'))
            index += 1
    write_json(OUTPUT, dict(rows=rows, teacher_retention_conflicts=conflicts,
        review_sha256=sha256(REVIEW), queue_sha256=sha256(ROOT/'review.json'),
        script_sha256=sha256(__file__), scene_raw_sha256=RAW_SHA,
        scope='Eight visually reviewed background crops from already-admitted TRAIN scenes. No fitting yet. Exact matching old scene rows identify where blanket teacher-logit preservation conflicts with approved background supervision; other teacher rows remain unchanged.',
        decision='Admit reviewed crops for a prospective matched negative-correction trial. Explicitly freeze the data and supervision contrast before fitting. Candidate corrections must replace matching scene-retention targets rather than coexist with contradictory duplicate targets; preserve unrelated teacher rows.',
        training_performed=False, development_evaluated=False, test_evaluated=False,
        human_review_verified=False, pi_tested=False, deployment_approved=False, qualification_achieved=False))
    print(dict(approved=len(rows), excluded=6, teacher_retention_conflicts=conflicts), flush=True)


if __name__ == '__main__':
    admit()
