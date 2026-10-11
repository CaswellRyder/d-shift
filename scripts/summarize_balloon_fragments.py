"""Recompute fragment/control evidence; repeated development is not qualification."""
import argparse
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from scripts.summarize_public_balloon_trial import checked_metrics
from scripts.train_balloon_fragments import CLASSES, MANIFEST, PARENT, QUEUE, REVIEW

PANELS = ('engdes2-lockbox-panel3-20261010', 'engdes2-development-panel2-20261010',
          'engdes2-development-scenes-20261008')


def summarize():
    models = {'incumbent': dict(path=str(Path(PARENT)/'student.fp32.tflite'))}
    models['incumbent']['sha256'] = sha256(models['incumbent']['path'])
    common = None
    for mode in ('control', 'fragment'):
        for seed in (47, 48):
            run = Path(f'runs/balloon-fragment-{mode}-seed{seed}-20261011')
            provenance = read_json(run/'provenance.json')
            expected = dict(mode=mode, seed=seed, epochs=8, batch_size=16, learning_rate=.0001,
                manifest_sha256=sha256(MANIFEST), queue_sha256=sha256(Path(QUEUE)/'review.json'),
                review_sha256=sha256(REVIEW), classes=CLASSES, ordinary_training_rows=1411,
                fragment_unique_crops=19, fragment_repetitions=4, fragment_training_rows=76,
                fixed_final_epoch=True, test_evaluated=False, deployment_approved=False,
                source_tflite_sha256=models['incumbent']['sha256'])
            if any(provenance.get(k) != v for k, v in expected.items()):
                raise ValueError('Training design or source identity changed')
            shared = {k: v for k, v in provenance.items() if k not in ('mode', 'seed')}
            if common is not None and common != shared:
                raise ValueError('Matched arms differ beyond mode and seed')
            common = shared
            model = run/'fragment.fp32.tflite'
            meta = read_json(model.with_suffix('.json'))
            report = read_json(run/'report.json')
            if (read_json(run/'status.json')['state'] != 'complete'
                    or meta['sha256'] != sha256(model)
                    or meta['provenance_sha256'] != sha256(run/'provenance.json')
                    or meta['classes'] != CLASSES or meta['parameters'] != 7892
                    or meta['deployment_approved'] is not False
                    or not 0 <= report['max_keras_tflite_delta'] <= .002):
                raise ValueError('Incomplete or changed research export')
            models[f'{mode}{seed}'] = dict(path=str(model), sha256=sha256(model),
                bytes=model.stat().st_size, provenance_sha256=sha256(run/'provenance.json'),
                fit_diagnostics=report)
            if mode == 'control':
                color = run/'color.fp32.tflite'
                cm = read_json(color.with_suffix('.json'))
                if (cm['sha256'] != sha256(color) or cm['classes'] != CLASSES[:3]
                        or cm['source_student_sha256'] != sha256(run/'color.keras')
                        or cm['source_four_output_checkpoint_sha256'] != sha256(run/'fragment.keras')
                        or cm['fragment_control_provenance_sha256'] != sha256(run/'provenance.json')
                        or cm['deployment_approved'] is not False or cm['pi_zero_verified'] is not False):
                    raise ValueError('Color export identity changed')
                models[f'color{seed}'] = dict(path=str(color), sha256=sha256(color),
                    metadata_sha256=sha256(color.with_suffix('.json')), bytes=color.stat().st_size)
    panels = {}
    for panel in PANELS:
        results = {}
        identities = None
        receipts = {}
        for prefix, names, field in (
                ('balloon-fragments', {'incumbent','control47','control48','fragment47','fragment48'}, 'variants'),
                ('balloon-fragment-color', {'incumbent','color47','color48'}, 'full_frame')):
            path = Path(f'runs/{prefix}-{panel}-20261011.json')
            doc = read_json(path)
            if (set(doc['results']) != names or doc['selection'] != 'split_merged'
                    or doc['split_max_aspect'] != 2.42913421076193
                    or doc['deployment_approved'] is not False or doc['test_evaluated'] is not False):
                raise ValueError('Panel scope changed')
            receipts[prefix] = dict(path=str(path), sha256=sha256(path),
                queue_sha256=doc['queue_sha256'], review_sha256=doc['review_sha256'])
            for name, result in doc['results'].items():
                if result['model_sha256'] != models[name]['sha256']:
                    raise ValueError('Panel model changed')
                variants = result[field]
                if set(variants) != ({'mser', 'lab'} if field == 'variants' else {'mser_confirmed_parts'}):
                    raise ValueError('Search variant changed')
                for search, variant in variants.items():
                    frame_ids = [(r['source'], r['frame_sha256'], r['truth']) for r in variant['frames']]
                    if identities is not None and frame_ids != identities:
                        raise ValueError('Compared frame identities differ')
                    identities = frame_ids
                    key = f'{name}/{"mser" if search == "mser_confirmed_parts" else search}'
                    metrics = checked_metrics(variant)
                    if key in results and results[key] != metrics:
                        raise ValueError('Incumbent changed between evaluations')
                    results[key] = metrics
        panels[panel] = dict(frames=len(identities), receipts=receipts, results=results)
    return dict(models=models, panels=panels, summary_script_sha256=sha256(__file__),
        training_design={k: v for k, v in common.items() if k != 'sample_identity'},
        decision='Do not promote fragment class: recall loss. Stage color47 for research Pi replay; color48 reproduces development count changes.',
        caveats=['Panels are repeatedly inspected and overlapping, not independent qualification.',
                 'Fragment training fits only 2-3 of 19 unique fragment crops; fit diagnostics are not validation.',
                 'Three-class export changes normalization; separately evaluated on every panel.',
                 'No live Pi accuracy or speed inferred from host evaluation or model size.'],
        qualification_achieved=False, deployment_approved=False, pi_tested=False,
        test_evaluated=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    write_json(args.output, summarize())
