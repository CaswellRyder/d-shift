"""Verify edge-search receipts and preserve a non-promotion decision."""
import argparse
from pathlib import Path

from dtr.data import read_json,sha256,write_json
from scripts.evaluate_reviewed_balloon_scenes import reviewed_frames
from scripts.summarize_public_balloon_trial import checked_metrics


def checked_panel(path,queue,review_path,model_hash,search_hash):
    doc=read_json(path)
    frames,review=reviewed_frames(queue,review_path)
    refs={row['id']:row for row,_ in frames}
    if (doc['model_sha256']!=model_hash or doc['search_sha256']!=search_hash
            or doc['queue_sha256']!=review['queue_sha256']
            or doc['review_sha256']!=sha256(review_path)
            or doc['threshold']!=.8 or doc['matching_iou']!=.5
            or doc['selection']!='split_merged' or doc['split_max_aspect']!=2.42913421076193
            or doc['test_evaluated'] is not False or doc['deployment_approved'] is not False
            or set(doc['results'])!={'baseline','edge'}):
        raise ValueError('Edge trial identity, scope or operating point changed')
    result={}
    for name,variant in doc['results'].items():
        if len(variant['frames'])!=len(refs) or {r['id'] for r in variant['frames']}!=set(refs):
            raise ValueError('Incomplete reviewed scene coverage')
        for row in variant['frames']:
            ref=refs[row['id']]
            if (row['source']!=ref['source'] or row['frame_sha256']!=ref['sha256']
                    or row['truth']!=ref['provisional_truth']):
                raise ValueError('Scene geometry or pixels changed')
        result[name]=checked_metrics(variant)
    return dict(report_sha256=sha256(path),queue_sha256=doc['queue_sha256'],
                review_sha256=doc['review_sha256'],frames=len(refs),results=result)


def summarize():
    from scripts.evaluate_balloon_edge_search import MODEL
    train_path='runs/balloon-edge-training-20261011.json'
    train=read_json(train_path)
    search_hash=sha256('scripts/research_balloon_edge_ellipse.py')
    if train['search_sha256']!=search_hash or train['test_evaluated'] is not False:
        raise ValueError('Training search changed')
    training={}
    for name,panel in train['panels'].items():
        counts={label:dict(targets=sum(f['baseline'][label]['targets'] for f in panel['frames']),
                    baseline=sum(f['baseline'][label]['localized'] for f in panel['frames']),
                    edge=sum(f['edge'][label]['localized'] for f in panel['frames']))
                for label in ('red_balloon','blue_balloon')}
        if counts!=panel['totals']:
            raise ValueError('Training aggregate differs from source rows')
        training[name]=dict(frames=len(panel['frames']),coverage=counts,
                            added_calls=sum(f['extra_calls'] for f in panel['frames']))
    panels={name:checked_panel('runs/balloon-edge-'+name+'-20261011.json','data/'+name,
                              'configs/'+name+'.json',sha256(MODEL),search_hash)
            for name in ('engdes2-lockbox-panel3-20261010','engdes2-development-panel2-20261010',
                         'engdes2-development-scenes-20261008')}
    return dict(training=training,training_report_sha256=sha256(train_path),
                training_scope=train['scope'],panels=panels,search_sha256=search_hash,
                model_sha256=sha256(MODEL),script_sha256=sha256(__file__),
                decision='Do not promote: blue recall gains are accompanied by lower precision on all three development panels',
                scope='Previously inspected, overlapping development imagery; no independent qualification',
                test_evaluated=False,pi_tested=False,deployment_approved=False,qualification_achieved=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    if Path(a.output).exists():
        raise FileExistsError(a.output)
    write_json(a.output,summarize())
