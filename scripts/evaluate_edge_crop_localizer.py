"""Reproduce paired training, export four checkpoints, screen complete TRAIN scenes."""
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from dtr.native_balloon_regions import NativeBalloonRegions
from dtr.runtime import Predictor
from scripts.audit_balloon_dense_proposals import supplement
from scripts.audit_indoor5_outcomes import compare_frames
from scripts.balloon_dense_runtime import DenseProposals
from scripts.compare_current_blue_search import MODEL, MODEL_SHA
from scripts.edge_crop_localizer_data import PROTOCOL, inputs
from scripts.evaluate_current_confident_containment import OUTPUT as BASELINE
from scripts.evaluate_red_blue_development import LABELS, detection_counts, metrics
from scripts.export_balloon_dense_proposals import proposal_parity
from scripts.prepare_blue_false_alarm_review import source_frames
from scripts.research_confident_containment import select
from scripts.research_blue_bright_search import search
from scripts.research_blue_peak_blend import decode
from scripts.research_predictor import classify_candidates
from scripts.summarize_dense_background_trial import check_baseline
from scripts.summarize_public_balloon_trial import checked_metrics
from scripts.train_edge_crop_localizer import run_path

OUTPUT = Path('research/balloon-red-blue-20261008/edge-crop-localizer-training-20261011.json')
RAW = Path('runs/edge-crop-localizer-scenes-20261011.json')


def export_all():
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    p = read_json(PROTOCOL)
    results, paired = {}, {}
    for arm in p['arms']:
        data = inputs(arm)
        paired[arm] = data
        for seed in p['seeds']:
            root = run_path(arm, seed)
            path = root/'proposal.fp32.tflite'
            if path.exists() or path.with_suffix('.json').exists():
                raise FileExistsError(path)
            prov, report = read_json(root/'provenance.json'), read_json(root/'report.json')
            if (read_json(root/'status.json')['state'] != 'complete' or prov['settings'] != p
                    or prov['arm'] != arm or prov['seed'] != seed or prov['parameters'] != 2342
                    or prov['fit_frames'] != 441 or prov['heldout_frames'] != 16
                    or prov['optimizer_steps'] != 4480 or report['optimizer_steps'] != 4480
                    or prov['tensor_sha256'] != data['hashes'] or prov['records'] != data['records']
                    or prov['folds'] != data['folds'].tolist()
                    or any(sha256(s) != h for s, h in prov['sources_sha256'].items())
                    or report['provenance_sha256'] != sha256(root/'provenance.json')
                    or report['model_sha256'] != sha256(root/'proposal.keras')
                    or report['predictions_sha256'] != sha256(root/'predictions.npz')):
                raise ValueError('Training inputs, scope or checkpoint changed')
            model = keras.models.load_model(root/'proposal.keras', compile=False)
            if model.count_params() != 2342 or model.output_shape != (None, 15, 20, 6):
                raise ValueError('Architecture changed')
            x = data['original_images']
            expected = model(x, training=False).numpy()
            with np.load(root/'predictions.npz') as stored:
                if not np.allclose(expected, stored['after'], atol=1e-5, rtol=0):
                    raise ValueError('Checkpoint outputs differ')

            @tf.function(input_signature=[tf.TensorSpec([1, 120, 160, 3], tf.float32)])
            def serve(image):
                return model(image, training=False)

            frozen = convert_variables_to_constants_v2(serve.get_concrete_function())
            converter = tf.lite.TFLiteConverter.from_concrete_functions([frozen])
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
            binary = converter.convert()
            runtime = tf.lite.Interpreter(model_content=binary, num_threads=1)
            runtime.allocate_tensors()
            inp, out = runtime.get_input_details()[0], runtime.get_output_details()[0]
            maximum = 0.
            for image, reference in zip(x, expected, strict=True):
                runtime.set_tensor(inp['index'], image[None])
                runtime.invoke()
                actual = runtime.get_tensor(out['index'])[0]
                if not np.isfinite(actual).all():
                    raise ValueError('Nonfinite TFLite output')
                maximum = max(maximum, float(np.max(np.abs(actual-reference))))
                proposal_parity(decode(actual, True), decode(reference, True))
            if maximum > .002:
                raise ValueError('Export tensor parity failed')
            path.write_bytes(binary)
            metadata = dict(kind='balloon_dense_proposal_research', experiment='edge_crop_localizer', arm=arm, seed=seed,
                sha256=sha256(path), bytes=len(binary), parameters=2342, proposal_threshold=.1,
                max_proposals=12, max_added_blue_proposals=2, decoder_sha256=sha256('scripts/audit_balloon_dense_proposals.py'),
                blended_decoder_sha256=sha256('scripts/research_blue_peak_blend.py'),
                source_checkpoint_sha256=report['model_sha256'], provenance_sha256=report['provenance_sha256'],
                max_tensor_delta=maximum, decoded_parity_frames=len(x),
                input_shape=[1, 120, 160, 3], output_shape=[1, 15, 20, 6], tensor_dtype='float32',
                preprocessing='320x240RGB to160x120INTER_AREA float32 range0..255',
                test_evaluated=False, deployment_approved=False, pi_tested=False)
            write_json(path.with_suffix('.json'), metadata)
            results[f'{arm}{seed}'] = dict(path=str(path), metadata=metadata, training_report=report)
            print(dict(export=f'{arm}{seed}', max_delta=maximum, frames=len(x)), flush=True)
    if (not np.array_equal(paired['control']['images'][:137], paired['edge_crop']['images'][:137])
            or not np.array_equal(paired['control']['targets'][:137], paired['edge_crop']['targets'][:137])
            or not np.array_equal(paired['control']['folds'], paired['edge_crop']['folds'])):
        raise ValueError('Original paired data changed')
    return results


def evaluate():
    if OUTPUT.exists() or RAW.exists():
        raise FileExistsError(OUTPUT)
    exports = export_all()
    reference = read_json(BASELINE)
    if (sha256(reference['raw']['path']) != reference['raw']['sha256'] or sha256(MODEL) != MODEL_SHA
            or any(sha256(p) != h for p, h in reference['helpers_sha256'].items())
            or reference['script_sha256'] != sha256('scripts/evaluate_current_confident_containment.py')):
        raise ValueError('Current comparison baseline changed')
    old = read_json(reference['raw']['path'])['training']
    if checked_metrics(old) != reference['panels']['training']['after']:
        raise ValueError('Baseline counts changed')
    parent = Path('runs/blue-head-only-background-20261011/proposal.fp32.tflite')
    networks = {'parent': DenseProposals(parent), **{n: DenseProposals(r['path']) for n, r in exports.items()}}
    classifier = Predictor(MODEL, allow_unvalidated=True)
    native = NativeBalloonRegions('artifacts/balloon-regions-mac-20261008/regions.so', direct=True)
    if (classifier.metadata['threshold'] != .8
            or read_json(native.path.with_suffix('.json'))['source_sha256'] != sha256('src/dtr/native_balloon_regions.c')):
        raise ValueError('Classifier threshold or native source changed')
    pixels = {s['source']: (s, rgb) for s, rgb in source_frames() if s['origin']=='complete_fit'}
    if len(pixels) != 64 or len(old['frames']) != 64:
        raise ValueError('Incomplete TRAIN coverage')
    rows = {name: [] for name in networks}
    for frame in old['frames']:
        source, rgb = pixels[frame['source']]
        if source['source_file_sha256'] != frame['frame_sha256'] or source['truth'] != frame['truth']:
            raise ValueError('TRAIN identity changed')
        base = search(rgb, native, True)
        for name, network in networks.items():
            network.predict(rgb)
            proposed = supplement(base, decode(network.interpreter.get_tensor(network.output['index'])[0], True))
            detections = select(classify_candidates(classifier, rgb, proposed))
            row = dict({k: frame[k] for k in ('id', 'source', 'frame_sha256', 'truth')},
                detections=detections, counts=detection_counts(frame['truth'], detections), neural_calls=len(proposed))
            if name == 'parent':
                check_baseline(row, frame)
            rows[name].append(row)
    full, summary = {}, {}
    for name, frames in rows.items():
        total = {c: {k: sum(f['counts'][c][k] for f in frames) for k in ('tp', 'fp', 'fn')} for c in LABELS}
        full[name] = dict(frames=frames, metrics=metrics(total))
        checked_metrics(full[name])
        comparison = compare_frames(frames, rows['parent'])
        eligible = not comparison['lost_parent_targets'] and all(total[c]['tp'] >= old['metrics'][c]['tp']
            and total[c]['fp'] <= old['metrics'][c]['fp'] for c in LABELS)
        summary[name] = dict(metrics=full[name]['metrics'], comparison=comparison, nonregression=eligible,
            neural_calls=sum(f['neural_calls'] for f in frames))
    eligible = all(summary[f'edge_crop{s}']['nonregression'] for s in (47, 48)) and any(
        summary[f'edge_crop{s}']['comparison']['gained_targets'] for s in (47, 48))
    write_json(RAW, full)
    write_json(OUTPUT, dict(exports=exports, training_scenes=summary, eligible_for_development=eligible,
        raw=dict(path=str(RAW), sha256=sha256(RAW)), script_sha256=sha256(__file__),
        baseline_sha256=sha256(BASELINE), protocol_sha256=sha256(PROTOCOL),
        scope='Paired augmentation retraining, export parity and original64completeTRAIN scene screening. No new development evaluation or independent accuracy claim.',
        development_evaluated=False, test_evaluated=False, pi_tested=False, deployment_approved=False, qualification_achieved=False))
    print(dict(results={n: dict(metrics=r['metrics'], lost=len(r['comparison']['lost_parent_targets']),
        gained=len(r['comparison']['gained_targets'])) for n, r in summary.items()}, eligible_for_development=eligible), flush=True)


if __name__ == '__main__':
    evaluate()
