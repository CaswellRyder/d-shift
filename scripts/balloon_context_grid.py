"""A paired context-restoration experiment, not a runtime default."""
import argparse
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.balloon_fine_grid import checked_grid_data, loss
from scripts.audit_balloon_fine_grid import checked_predictions

PROTOCOL = Path('configs/balloon-context-grid-trial-20261011.json')


def network():
    import keras
    layers = keras.layers
    inputs = layers.Input((120, 160, 3))
    x = layers.Rescaling(1 / 255.)(inputs)
    x = layers.Conv2D(8, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.SeparableConv2D(16, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.SeparableConv2D(24, 3, strides=1, padding='same', activation='relu')(x)
    x = layers.SeparableConv2D(24, 5, dilation_rate=2, padding='same', activation='relu')(x)
    heat = layers.Conv2D(2, 1, bias_initializer=keras.initializers.Constant(-4.), name='center_logits')(x)
    box = layers.Conv2D(4, 1, activation='sigmoid', name='center_geometry')(x)
    return keras.Model(inputs, layers.Concatenate(name='proposal_output')([heat, box]))


def spatial_cost(model):
    """Analytic MACs and receptive field; excludes bias, activations and runtime overhead."""
    import keras
    jump, receptive, macs = 1, 1, 0
    for layer in model.layers:
        if isinstance(layer, (keras.layers.Conv2D, keras.layers.SeparableConv2D)):
            k = layer.kernel_size[0]
            height, width, channels = layer.output.shape[1:]
            incoming = layer.input.shape[-1]
            if isinstance(layer, keras.layers.SeparableConv2D):
                macs += height * width * incoming * layer.depth_multiplier * (k * k + channels)
            else:
                macs += height * width * incoming * channels * k * k
            # The two parallel 1x1 heads do not alter receptive field or jump.
            receptive += (k - 1) * layer.dilation_rate[0] * jump
            jump *= layer.strides[0]
    return dict(parameters=model.count_params(), multiply_accumulates=int(macs),
                theoretical_receptive_field_input_pixels=int(receptive), output_stride=int(jump),
                scope='Analytic convolution work only, not wall-clock time or Pi throughput')


def train(seed, output):
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    protocol = read_json(PROTOCOL)
    if seed not in protocol['seeds'] or protocol['final_spatial_dilation'] != 2:
        raise ValueError('Fixed paired protocol required')
    doc, x, targets, folds, groups = checked_grid_data()
    if sha256(Path(protocol['data']) / 'manifest.json') != protocol['manifest_sha256']:
        raise ValueError('Data identity changed')
    control = Path(protocol['fine_controls'][str(seed)])
    cp, _, _ = checked_predictions(control, targets, folds)
    if (cp['seed'] != seed or any(cp[k] != protocol[k] for k in ('epochs', 'batch_size', 'learning_rate', 'box_objective'))):
        raise ValueError('Fine-grid control schedule changed')
    fit, hold = folds != 0, folds == 0
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    keras.utils.set_random_seed(seed)
    model = network()
    out.mkdir(parents=True)
    np.save(out / 'targets.npy', targets, allow_pickle=False)
    if sha256(out / 'targets.npy') != cp['targets_sha256']:
        raise ValueError('Paired supervision is not byte-identical')
    provenance = dict(seed=seed, epochs=protocol['epochs'], batch_size=protocol['batch_size'],
        learning_rate=protocol['learning_rate'], box_objective=protocol['box_objective'],
        manifest_sha256=protocol['manifest_sha256'], folds=folds.tolist(), groups=groups,
        targets_sha256=cp['targets_sha256'], fine_control_provenance_sha256=sha256(control / 'provenance.json'),
        script_sha256=sha256(__file__), protocol_sha256=sha256(PROTOCOL),
        helpers_sha256={p: sha256(p) for p in ('scripts/balloon_fine_grid.py',
            'scripts/balloon_dense_iou_loss.py', 'scripts/train_balloon_dense_proposals.py')},
        omitted_negatives=doc['fine_grid_omitted_negatives'], fixed_final_epoch=True,
        architecture=spatial_cost(model), test_evaluated=False, deployment_approved=False, pi_tested=False)
    write_json(out / 'provenance.json', provenance)
    write_json(out / 'status.json', dict(state='running'))
    try:
        ds = tf.data.Dataset.from_tensor_slices((x[fit], targets[fit])).shuffle(int(fit.sum()), seed=seed).batch(protocol['batch_size'])
        options = tf.data.Options()
        options.threading.private_threadpool_size = 2
        ds = ds.with_options(options).prefetch(1)
        model.compile(optimizer=keras.optimizers.Adam(protocol['learning_rate']), loss=loss)

        class Progress(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if not logs or not all(np.isfinite(v) for v in logs.values()):
                    raise ValueError('Nonfinite training')
                write_json(out / 'status.json', dict(state='running', epoch=epoch + 1, metrics=logs))

        model.fit(ds, epochs=protocol['epochs'], verbose=2,
                  callbacks=[Progress(), keras.callbacks.CSVLogger(str(out / 'epochs.csv'))])
        model.save(out / 'proposal.keras')
        values = model(x, training=False).numpy()
        if values.shape != (len(x), 30, 40, 6) or not np.isfinite(values).all():
            raise ValueError('Invalid outputs')
        np.save(out / 'predictions.npy', values, allow_pickle=False)
        report = dict(training_loss=float(loss(tf.constant(targets[fit]), tf.constant(values[fit]))),
            grouped_holdout_loss=float(loss(tf.constant(targets[hold]), tf.constant(values[hold]))),
            model_sha256=sha256(out / 'proposal.keras'), predictions_sha256=sha256(out / 'predictions.npy'),
            provenance_sha256=sha256(out / 'provenance.json'), fit_frames=int(fit.sum()), holdout_frames=int(hold.sum()),
            scope='Loss diagnostics only; no detection quality or Pi verification',
            test_evaluated=False, deployment_approved=False, pi_tested=False)
        write_json(out / 'report.json', report)
        write_json(out / 'status.json', dict(state='complete'))
        print(report)
    except BaseException as exc:
        write_json(out / 'status.json', dict(state='failed', error=str(exc)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=(47, 48), required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    train(args.seed, args.output)
