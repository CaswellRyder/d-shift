"""Verify matched clutter evidence without promoting development results."""
import argparse
from collections import Counter
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from scripts.summarize_balloon_fragments import PANELS
from scripts.summarize_public_balloon_trial import checked_metrics
from scripts.train_balloon_reviewed_clutter import PARENT, MANIFEST, REVIEW
from scripts.train_balloon_fragments import PARENT as INCUMBENT


def summarize():
    sources = dict(incumbent=Path(INCUMBENT)/'student.fp32.tflite', parent=PARENT/'color.fp32.tflite')
    models, arms = {}, {}
    for mode in ('control', 'admit'):
        for seed in (47, 48):
            run = Path(f'runs/balloon-reviewed-clutter-{mode}-seed{seed}-20261011')
            name = f'{mode}{seed}'
            p = read_json(run/'provenance.json')
            model = run/'student.fp32.tflite'
            meta = read_json(model.with_suffix('.json'))
            if (p['mode'] != mode or p['seed'] != seed or p['epochs'] != 8
                    or p['manifest_sha256'] != sha256(MANIFEST) or p['review_sha256'] != sha256(REVIEW)
                    or p['parent_model_sha256'] != sha256(sources['parent'])
                    or p['parent_checkpoint_sha256'] != sha256(PARENT/'color.keras')
                    or p['ordinary_rows'] != 1487 or p['hard_label_rows'] != 36
                    or p['learning_rate'] != .0001 or p['fixed_final_epoch'] is not True
                    or read_json(run/'status.json')['state'] != 'complete'
                    or meta['provenance_sha256'] != sha256(run/'provenance.json')
                    or meta['source_student_sha256'] != sha256(run/'student.keras')
                    or meta['sha256'] != sha256(model) or meta['deployment_approved'] is not False
                    or not 0 <= meta['max_keras_tflite_delta'] <= .002):
                raise ValueError('Training design or export identity changed')
            arms[name] = p
            sources[name] = model
            models[name] = dict(provenance_sha256=sha256(run/'provenance.json'),
                design={k: v for k, v in p.items() if k != 'sample_identity'},
                fit_diagnostics=read_json(run/'report.json'))
    common = arms['control47']['sample_identity'][:1487]
    for name, p in arms.items():
        if (len(p['sample_identity']) != 1523 or p['sample_identity'][:1487] != common
                or Counter(r['label'] for r in p['sample_identity']) !=
                Counter(r['label'] for r in arms['control47']['sample_identity'])
                or any(r['label'] != 'background' or r['preserve'] is not False
                       for r in p['sample_identity'][1487:])):
            raise ValueError('Matched row order, counts or supervision changed')
    for mode in ('control', 'admit'):
        if arms[f'{mode}47']['sample_identity'] != arms[f'{mode}48']['sample_identity']:
            raise ValueError('Seeds received different data')
    for name, model in sources.items():
        models.setdefault(name, {}).update(path=str(model), sha256=sha256(model), bytes=model.stat().st_size)
    panels = {}
    for panel in PANELS:
        path = Path(f'runs/balloon-reviewed-clutter-{panel}-20261011.json')
        doc = read_json(path)
        if (set(doc['results']) != set(sources) or doc['selection'] != 'split_merged'
                or doc['split_max_aspect'] != 2.42913421076193
                or doc['test_evaluated'] is not False or doc['deployment_approved'] is not False):
            raise ValueError('Development comparison scope changed')
        results, frame_ids = {}, None
        for name, result in doc['results'].items():
            if result['model_sha256'] != models[name]['sha256'] or result['threshold'] != .8:
                raise ValueError('Model or operating point changed')
            variant = result['full_frame']['mser_confirmed_parts']
            identities = [(r['source'], r['frame_sha256'], r['truth']) for r in variant['frames']]
            if frame_ids is not None and frame_ids != identities:
                raise ValueError('Models saw different frames')
            frame_ids = identities
            results[name] = checked_metrics(variant)
        panels[panel] = dict(path=str(path), sha256=sha256(path), frames=len(frame_ids), results=results)
    return dict(models=models, panels=panels, summary_script_sha256=sha256(__file__),
        decision='Retain earlier color-control candidate; do not promote clutter retraining. Both admit seeds lose red targets.',
        scope='Repeatedly inspected development panels; no independent sessions or final qualification',
        training_note='Nine reviewed negatives repeated four times; one is an existing background crop. This is an emphasis trial, not nine new independent images.',
        test_evaluated=False, deployment_approved=False, pi_tested=False, qualification_achieved=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    write_json(args.output, summarize())
