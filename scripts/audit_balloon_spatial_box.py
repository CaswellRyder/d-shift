"""Training-source grouped box-head validation; not full-model qualification."""
import argparse
from pathlib import Path

import numpy as np

from dtr.data import load_rgb, read_json, sha256, write_json
from scripts.train_balloon_box_refinement import box_features, box_metrics, partition
from scripts.train_balloon_quality import fit_ridge, fold_ids, training_rows


def grouped_predictions(features, rows, config):
    x = np.asarray(features)
    if x.ndim != 2 or len(x) != len(rows) or not np.isfinite(x).all():
        raise ValueError('Invalid box feature matrix')
    folds = fold_ids(rows)
    if len(np.unique(folds)) < 2:
        raise ValueError('Require distinct source folds')
    targets = np.asarray([r['normalized_target'] for r in rows])
    predictions = np.empty_like(targets)
    groups = []
    for fold in np.unique(folds):
        take = folds != fold
        if take.sum() < 2:
            raise ValueError('Insufficient grouped fitting examples')
        for key in ('source_image','source_group','source_sha256'):
            train = {r[key] for r,yes in zip(rows,take) if yes}
            held = {r[key] for r,yes in zip(rows,take) if not yes}
            if train & held:
                raise ValueError('Cross-fold source leakage')
        for coordinate in range(4):
            w,b = fit_ridge(x[take],targets[take,coordinate],config['ridge_penalty'])
            predictions[~take,coordinate] = x[~take]@w+b
        groups.append(dict(fold=int(fold),fit_count=int(take.sum()),hold_count=int((~take).sum())))
    return predictions, groups


def audit(config_path, student_run):
    config, run = read_json(config_path), Path(student_run)
    rows, receipts = training_rows(read_json(config['quality_config']))
    fitted, held, excluded = partition(rows,config)
    rows = fitted+held
    metadata = read_json(run/'student.fp32.json')
    if (metadata['source_student_sha256'] != sha256(run/'student.keras')
            or metadata['sha256'] != sha256(run/'student.fp32.tflite')):
        raise ValueError('Backbone identity changed')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    model = keras.models.load_model(run/'student.keras',compile=False)
    model.trainable = False
    pixels = np.stack([load_rgb(r['absolute_path'],64) for r in rows])
    results = {}
    for grid in (2,4):
        feature_model = keras.Model(model.input,box_features(model,grid))
        features = feature_model(pixels,training=False).numpy()
        predictions,folds = grouped_predictions(features,rows,config)
        results[str(grid)] = dict(metrics=box_metrics(rows,predictions,config),folds=folds,
                                  feature_dimensions=features.shape[1])
    return dict(results=results,excluded=excluded,source_reviews=receipts,
                parent_model_sha256=metadata['sha256'],parent_checkpoint_sha256=sha256(run/'student.keras'),
                config_sha256=sha256(config_path),quality_config_sha256=sha256(config['quality_config']),
                script_sha256=sha256(__file__),
                training_script_sha256=sha256('scripts/train_balloon_box_refinement.py'),
                source_partitions=[dict(source_image=r['source_image'],source_group=r['source_group'],
                    source_sha256=r['source_sha256'],crop_sha256=r['sha256'],fold=int(f))
                    for r,f in zip(rows,fold_ids(rows))],
                scope='Only regression labels held out by source group. Frozen backbone saw related classification data. Not independent model accuracy.',
                test_evaluated=False,deployment_approved=False,pi_tested=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config','student-run','output'):
        p.add_argument('--'+name,required=True)
    a = p.parse_args()
    if Path(a.output).exists():
        raise FileExistsError(a.output)
    report = audit(a.config,a.student_run)
    write_json(a.output,report)
    print({k:{key:value for key,value in v['metrics'].items() if key!='examples'}
           for k,v in report['results'].items()})
