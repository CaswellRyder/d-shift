"""Compare visible-box correction with identical-classification parent proposals."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.runtime import softmax
from dtr.vision import suppress_duplicates
from scripts.evaluate_balloon_quality import panel_frames
from scripts.evaluate_red_blue_development import LABELS, detection_counts, metrics
from scripts.research_balloon_search import suppress_confirmed_balloon_parts
from scripts.train_balloon_box_refinement import check_config, decoded_box


class BoxPredictor:
    def __init__(self, path):
        self.metadata = read_json(Path(path).with_suffix('.json'))
        m = self.metadata
        if (m.get('kind') != 'balloon_box_research' or m['sha256'] != sha256(path)
                or m['classes'] != ['background', 'red_balloon', 'blue_balloon']
                or m['size'] != 64 or m['threshold'] != .8 or m.get('deployment_approved') is not False):
            raise ValueError('Invalid research box-model identity')
        check_config(m['config'])
        import tensorflow as tf
        self.interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=1)
        self.interpreter.allocate_tensors()
        self.input, self.output = self.interpreter.get_input_details()[0], self.interpreter.get_output_details()[0]
        if (self.input['shape'].tolist() != [1,64,64,3] or self.output['shape'].tolist() != [1,7]
                or self.input['dtype'] != np.float32 or self.output['dtype'] != np.float32):
            raise ValueError('Box model requires explicit seven-output FP32 contract')

    def predict(self, rgb):
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
            raise ValueError('Require nonempty uint8 RGB crop')
        crop = np.asarray(Image.fromarray(rgb).resize((64,64), Image.Resampling.BILINEAR), np.float32)
        self.interpreter.set_tensor(self.input['index'], crop[None])
        self.interpreter.invoke()
        result = self.interpreter.get_tensor(self.output['index'])[0]
        if not np.isfinite(result).all():
            raise ValueError('Non-finite box model output')
        return softmax(result[:3]), result[3:]


def observations(predictor, rgb, old_rows):
    original, refined, maximum = [], [], 0.
    for old in old_rows:
        a,b,c,d = old['crop_box']
        if (any(type(v) is not int for v in (a,b,c,d))
                or not 0 <= a < c <= 320 or not 0 <= b < d <= 240):
            raise ValueError('Invalid saved crop box')
        scores, normalized = predictor.predict(rgb[b:d,a:c])
        previous = np.asarray(old['scores'])
        if previous.shape != (3,) or not np.isfinite(previous).all():
            raise ValueError('Invalid saved class scores')
        delta = float(np.abs(scores-previous).max())
        maximum = max(maximum, delta)
        best = int(scores.argmax())
        label = predictor.metadata['classes'][best]
        accepted = bool(best != 0 and scores[best] >= .8)
        if delta > 1e-5 or label != old['label'] or accepted != old['raw_accepted']:
            raise ValueError('Frozen classifier predictions changed')
        row = {k:old[k] for k in ('box','crop_box','area','color_group','proposal_score') if k in old}
        row.update(label=label,score=float(scores[best]),scores=scores.tolist(),accepted=accepted)
        original.append(row)
        box, applied = decoded_box(normalized, old, predictor.metadata['config']) if accepted else (old['box'],False)
        refined.append(dict(row,box=box,proposal_box=old['box'],box_correction_applied=applied,
                            predicted_crop_xyxy=normalized.tolist()))
    return original, refined, maximum


def evaluate(model_path, source_path, name, panel):
    source = read_json(source_path)
    if source.get('test_evaluated') is not False or source.get('deployment_approved') is not False:
        raise ValueError('Require explicit development-only source report')
    predictor = BoxPredictor(model_path)
    baseline = source['results'][name]
    if baseline['model_sha256'] != predictor.metadata['source_tflite_sha256'] or baseline['threshold'] != .8:
        raise ValueError('Parent classifier identity changed')
    pixels = panel_frames(panel, source)
    old = baseline['full_frame']['mser_confirmed_parts']
    if len(old['frames']) != len(pixels) or {f['source'] for f in old['frames']} != set(pixels):
        raise ValueError('Scene coverage changed')
    outputs = {k:[] for k in ('parent','refined')}
    max_delta = 0.
    for frame in old['frames']:
        rgb, truth = pixels[frame['source']]
        if truth != frame['truth']:
            raise ValueError('Reviewed truth changed')
        original, refined, delta = observations(predictor,rgb,frame['detections'])
        max_delta = max(max_delta,delta)
        for name, rows in (('parent',original),('refined',refined)):
            found = suppress_confirmed_balloon_parts(suppress_duplicates(rows))
            outputs[name].append(dict(source=frame['source'],truth=truth,detections=found,
                                      counts=detection_counts(truth,found)))
    results = {}
    for name, frames in outputs.items():
        total = {label:{key:sum(f['counts'][label][key] for f in frames) for key in ('tp','fp','fn')}
                 for label in LABELS}
        results[name] = dict(metrics=metrics(total),frames=frames)
    if results['parent']['metrics'] != old['metrics']:
        raise ValueError('Uncorrected control does not reproduce parent metrics')
    return dict(results=results,panel=panel,model_sha256=sha256(model_path),
        metadata_sha256=sha256(Path(model_path).with_suffix('.json')), source_report_sha256=sha256(source_path),
        max_class_probability_delta=max_delta,script_sha256=sha256(__file__),
        helpers_sha256={p:sha256(p) for p in ('scripts/train_balloon_box_refinement.py',
            'scripts/evaluate_balloon_quality.py','scripts/evaluate_red_blue_development.py',
            'scripts/research_balloon_search.py','src/dtr/vision.py')},
        scope='Development-only fixed-proposal replay; frozen class scores, corrected geometry, fresh NMS',
        test_evaluated=False,deployment_approved=False,pi_timing_measured=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model','source-report','source-model','output'):
        parser.add_argument(f'--{name}',required=True)
    parser.add_argument('--panel',choices=('indoor','original'),required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    report = evaluate(args.model,args.source_report,args.source_model,args.panel)
    write_json(args.output,report)
    print({k:v['metrics'] for k,v in report['results'].items()})
