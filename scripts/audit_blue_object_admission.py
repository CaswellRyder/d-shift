"""Compact provenance and attribution for reviewed external TRAIN negatives."""
from collections import Counter
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from scripts.blue_object_data import checked_extra, ROOT, QUEUE, REVIEW

OUTPUT=Path('research/balloon-red-blue-20261008/blue-object-admission-20261011.json')


def audit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    rows=checked_extra()
    acquisition,queue,licenses=read_json(ROOT/'acquisition.json'),read_json(QUEUE/'review.json'),read_json(ROOT/'license-verification.json')
    selection=read_json(ROOT/'selection.json')
    sources={r['source_image']:r['attribution'] for r in rows}
    write_json(OUTPUT,dict(download_candidates=len(selection['selected']),
        acquired_blue_screen_photos=len(acquisition['images']),
        license_verified_photos=sum(r['verified'] is True for r in licenses['results']),
        review_frames=len(queue['frames']),review_crops=len(queue['samples']),
        admitted_crops=len(rows),admitted_photos=len(sources),admitted_labels=dict(Counter(r['label'] for r in rows)),
        exact_crops=[{k:r[k] for k in ('id','frame_id','source_image','sha256','label','proposal')} for r in rows],
        attribution=sources,manual_balloon_guards=read_json(REVIEW)['manual_balloon_guards'],
        hashes={str(p):sha256(p) for p in (ROOT/'acquisition.json',ROOT/'selection.json',
            ROOT/'license-verification.json',QUEUE/'review.json',REVIEW,Path('scripts/blue_object_data.py'),Path(__file__))},
        source_documentation='https://storage.googleapis.com/openimages/web/download_v7.html',
        scope='Exact reviewed TRAIN crop admission only. Known balloons outside one crop are explicitly guarded. Whole photos are not negative-labeled. Source license link checks are not ownership warranties.',
        training_approved=True,test_evaluated=False,pi_tested=False,deployment_approved=False,qualification_achieved=False))
    print(dict(crops=len(rows),photographs=len(sources),receipt_sha256=sha256(OUTPUT)))


if __name__=='__main__':
    audit()
