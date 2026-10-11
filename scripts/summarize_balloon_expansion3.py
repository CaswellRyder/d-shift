"""Verify reviewed training-view expansion against same-count controls."""
import argparse
from collections import Counter
from pathlib import Path

from dtr.data import read_json,sha256,write_json
from scripts.balloon_expansion3 import QUEUE,REVIEW,MANIFEST,data,checked_extra
from scripts.train_balloon_reviewed_clutter import PARENT
from scripts.summarize_balloon_fragments import PANELS
from scripts.summarize_dense_background_trial import check_baseline
from scripts.summarize_public_balloon_trial import checked_metrics


def summarize():
    extra = checked_extra()
    sources = dict(parent=PARENT/'color.fp32.tflite')
    models = {}
    for mode in ('control','admit'):
        rows,_ = data(mode)
        identity = [{k:r[k] for k in ('source_image','sha256','label','preserve')} for r in rows]
        for seed in (47,48):
            name = f'{mode}{seed}'
            run = Path(f'runs/balloon-expansion3-{mode}-seed{seed}-20261011')
            p = read_json(run/'provenance.json')
            model = run/'student.fp32.tflite'
            meta = read_json(model.with_suffix('.json'))
            if (p['mode']!=mode or p['seed']!=seed or p['experiment']!='expansion3'
                    or p['epochs']!=8 or p['learning_rate']!=.0001 or p['batch_size']!=16
                    or p['manifest_sha256']!=sha256(MANIFEST) or p['review_sha256']!=sha256(REVIEW)
                    or p['queue_sha256']!=sha256(Path(QUEUE)/'review.json')
                    or p['parent_model_sha256']!=sha256(sources['parent'])
                    or p['parent_checkpoint_sha256']!=sha256(PARENT/'color.keras')
                    or p['sample_identity']!=identity or p['ordinary_rows']!=1487
                    or p['hard_label_rows']!=158 or p['reviewed_unique_crops']!=79
                    or p['fixed_final_epoch'] is not True
                    or read_json(run/'status.json')['state']!='complete'
                    or meta['provenance_sha256']!=sha256(run/'provenance.json')
                    or meta['sha256']!=sha256(model) or meta['deployment_approved'] is not False
                    or meta['source_student_sha256']!=sha256(run/'student.keras')
                    or not 0<=meta['max_keras_tflite_delta']<=.002):
                raise ValueError('Matched training or export identity changed')
            sources[name] = model
            models[name] = dict(path=str(model),sha256=sha256(model),bytes=model.stat().st_size,
                provenance_sha256=sha256(run/'provenance.json'),
                design={k:v for k,v in p.items() if k!='sample_identity'},
                fit_diagnostics=read_json(run/'report.json'))
    models['parent'] = dict(path=str(sources['parent']),sha256=sha256(sources['parent']))
    panels = {}
    for panel in PANELS:
        path = Path(f'runs/balloon-expansion3-scenes-{panel}-20261011.json')
        doc = read_json(path)
        if (set(doc['results'])!=set(sources) or doc['selection']!='split_merged'
                or doc['split_max_aspect']!=2.42913421076193 or doc['matching_script_sha256']!=sha256('scripts/evaluate_red_blue_development.py')
                or doc['test_evaluated'] is not False or doc['deployment_approved'] is not False):
            raise ValueError('Development scope changed')
        previous = read_json(f'runs/balloon-dense-bg-scenes-{panel}-20261011.json')
        results,identities = {},None
        for name,r in doc['results'].items():
            if r['model_sha256']!=models[name]['sha256'] or r['threshold']!=.8:
                raise ValueError('Model or frozen threshold changed')
            v = r['full_frame']['mser_confirmed_parts']
            ids = [(f['source'],f['frame_sha256'],f['truth']) for f in v['frames']]
            if identities is not None and ids!=identities:
                raise ValueError('Compared different scene identities')
            identities = ids
            results[name] = checked_metrics(v)
            if name=='parent':
                for a,b in zip(v['frames'],previous['results']['baseline']['frames'],strict=True):
                    check_baseline(a,b)
        panels[panel] = dict(path=str(path),sha256=sha256(path),frames=len(identities),results=results)
    queue = read_json(Path(QUEUE)/'review.json')
    return dict(models=models,panels=panels,queue_sha256=sha256(Path(QUEUE)/'review.json'),
        review_sha256=sha256(REVIEW),frames=len(queue['frames']),rejected_candidates=len(queue['rejected']),
        admitted_counts=dict(Counter(r['label'] for r in extra)),script_sha256=sha256(__file__),
        decision='No promotion: expansion effects are small and inconsistent across seeds/panels; admit48 also loses a red target on the small panel. Keep the earlier color-control integration candidate.',
        next_direction='Use approved new full frames for localization or mine reviewed training-only failure crops. More easy color crops alone did not resolve full-scene errors.',
        caveats=['Assistant-reviewed public training data, not independent human certification.',
                 'Same training filename families; additional frames are not new independent environments.',
                 'Development panels overlap and have been repeatedly inspected, not final qualification.',
                 'Matched class-count controls show training-step effects separately from new-data effects.',
                 'Pi model, camera service and ESP32 untouched.'],
        test_evaluated=False,deployment_approved=False,pi_tested=False,qualification_achieved=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    write_json(a.output,summarize())
