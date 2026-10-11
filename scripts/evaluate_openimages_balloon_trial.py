"""Matched final-checkpoint development; separate source and accuracy evidence."""
import argparse
from collections import Counter
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from scripts.openimages_balloon_training import data, QUEUE, REVIEW, ROOT, LICENSE, MANIFEST
from scripts.evaluate_reviewed_balloon_scenes import evaluate
from scripts.summarize_balloon_fragments import PANELS
from scripts.summarize_dense_background_trial import check_baseline
from scripts.summarize_public_balloon_trial import checked_metrics

PROTOCOL = 'configs/balloon-openimages-trial-20261011.json'
PARENT = Path('runs/balloon-fragment-control-seed47-20261011/color.fp32.tflite')


def validate_run(provenance, metadata, run, mode, seed, identity, protocol, count):
    if (provenance['mode'] != mode or provenance['seed'] != seed
            or provenance['experiment'] != 'openimages'
            or any(provenance[k] != protocol[k] for k in ('epochs','learning_rate','batch_size','hard_weight'))
            or provenance['manifest_sha256'] != sha256(MANIFEST)
            or provenance['review_sha256'] != sha256(REVIEW)
            or provenance['queue_sha256'] != sha256(Path(QUEUE)/'review.json')
            or provenance['data_helper_sha256'] != sha256('scripts/openimages_balloon_training.py')
            or provenance['script_sha256'] != sha256('scripts/train_balloon_reviewed_clutter.py')
            or provenance['trial_protocol_sha256'] != sha256(PROTOCOL)
            or provenance['license_verification_sha256'] != sha256(LICENSE)
            or provenance['parent_model_sha256'] != sha256(PARENT)
            or provenance['parent_checkpoint_sha256'] != sha256(PARENT.parent/'color.keras')
            or provenance['sample_identity'] != identity
            or provenance['ordinary_rows'] != 1487 or provenance['hard_label_rows'] != count*2
            or provenance['reviewed_unique_crops'] != count or provenance['fixed_final_epoch'] is not True
            or read_json(run/'status.json')['state'] != 'complete'
            or metadata['provenance_sha256'] != sha256(run/'provenance.json')
            or metadata['sha256'] != sha256(run/'student.fp32.tflite')
            or metadata['source_student_sha256'] != sha256(run/'student.keras')
            or metadata['deployment_approved'] is not False
            or not 0 <= metadata['max_keras_tflite_delta'] <= .002):
        raise ValueError('Matched public-data trial identity changed')


def models():
    protocol = read_json(PROTOCOL)
    if protocol['parent_sha256'] != sha256(PARENT):
        raise ValueError('Frozen parent changed')
    sources,evidence = {'parent':PARENT},{}
    for mode in ('control','admit'):
        rows,extra = data(mode)
        identity = [{k:r[k] for k in ('source_image','sha256','label','preserve')} for r in rows]
        for seed in (47,48):
            name = f'{mode}{seed}'
            run = Path(f'runs/balloon-openimages-{mode}-seed{seed}-20261011')
            p,meta = read_json(run/'provenance.json'),read_json(run/'student.fp32.json')
            validate_run(p,meta,run,mode,seed,identity,protocol,len(extra))
            sources[name] = run/'student.fp32.tflite'
            evidence[name] = dict(path=str(sources[name]),model_sha256=sha256(sources[name]),
                provenance_sha256=sha256(run/'provenance.json'),fit=read_json(run/'report.json'))
    return sources,evidence,extra


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    sources,evidence,extra = models()
    panels = {}
    for panel in PANELS:
        path = Path(f'runs/balloon-openimages-scenes-{panel}-20261011.json')
        if path.exists():
            raise FileExistsError(path)
        result = evaluate('data/'+panel,'configs/'+panel+'.json',sources,
                          ('mser_confirmed_parts',),'split_merged',2.42913421076193)
        write_json(path,result)
        prior = read_json(f'runs/balloon-expansion4-scenes-{panel}-20261011.json')
        metrics,identities = {},None
        for name,record in result['results'].items():
            scenes = record['full_frame']['mser_confirmed_parts']
            identity = [(r['source'],r['frame_sha256'],r['truth']) for r in scenes['frames']]
            if identities is not None and identities != identity:
                raise ValueError('Compared different scenes')
            identities = identity
            metrics[name] = checked_metrics(scenes)
            if name=='parent':
                old = prior['results']['parent']['full_frame']['mser_confirmed_parts']['frames']
                for a,b in zip(scenes['frames'],old,strict=True):
                    check_baseline(a,b)
        panels[panel] = dict(path=str(path),sha256=sha256(path),frames=len(identities),results=metrics)
        print(panel,{n:{c:(m['tp'],m['fp'],m['fn']) for c,m in s.items()} for n,s in metrics.items()},flush=True)
    queue = read_json(Path(QUEUE)/'review.json')
    licenses = read_json(LICENSE)
    selected_frames = {r['frame_id'] for r in extra}
    write_json(output,dict(models=evidence,panels=panels,protocol_sha256=sha256(PROTOCOL),
        queue_sha256=sha256(Path(QUEUE)/'review.json'),review_sha256=sha256(REVIEW),
        acquisition_sha256=sha256(ROOT/'acquisition.json'),license_verification_sha256=sha256(LICENSE),
        script_sha256=sha256(__file__),selected_crop_ids=[r['id'] for r in extra],
        selected_counts=dict(Counter(r['label'] for r in extra)),selected_frames=len(selected_frames),
        selected_source_attribution=[f['attribution'] for f in queue['frames'] if f['id'] in selected_frames],
        source_page_verified_images=sum(r['verified'] for r in licenses['results']),
        reviewed_frames=len(queue['frames']),reviewed_crops=len(queue['samples']),
        scope='Assistant-reviewed licensed public TRAIN crops; existing repeated development panels only',
        source_urls=['https://storage.googleapis.com/openimages/web/download_v7.html',
                     'https://storage.googleapis.com/openimages/web/factsfigures_v7.html#licenses'],
        caveats=['No independent held-out qualification or speed measurement.',
                 'CVDF images were resized, then scaled to 320x240 and cropped; source attribution is retained.',
                 'Photo-page CC BY verification is not a copyright ownership warranty.',
                 'Perceptual duplicate screening is heuristic; capture-session independence is not established.'],
        test_evaluated=False,pi_tested=False,deployment_approved=False,qualification_achieved=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    a = p.parse_args()
    run(a.output)
