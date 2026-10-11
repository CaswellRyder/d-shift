"""Freeze every gated development input for native Pi tensor parity, no training."""
import argparse
from pathlib import Path
import shutil

import numpy as np

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor
from scripts.balloon_dense_runtime import DenseProposals, prepare_input
from scripts.evaluate_reviewed_balloon_scenes import reviewed_frames
from scripts.research_balloon_search import experimental
from scripts.research_predictor import classify_candidates
from scripts.research_dense_supported_selection import candidate_pairs, supported_selection
from scripts.summarize_dense_supported_selection import same_output

PANELS = ('engdes2-lockbox-panel3-20261010', 'engdes2-development-panel2-20261010',
          'engdes2-development-scenes-20261008')


def prepare(output):
    output = Path(output)
    output.mkdir(exist_ok=False)
    classifier = Path('runs/balloon-fragment-control-seed47-20261011/color.fp32.tflite')
    predictor = Predictor(classifier, allow_unvalidated=True)
    networks, models = {}, {}
    for seed in ('47', '48'):
        path = Path(f'runs/balloon-dense-overlap-l1_giou-seed{seed}-20261011/proposal.fp32.tflite')
        networks[seed] = DenseProposals(path)
        models[seed] = f'proposal-{seed}.tflite'
        shutil.copy2(path, output / models[seed])
        shutil.copy2(path.with_suffix('.json'), output / f'proposal-{seed}.json')
    inputs, rows, source_reports = [], [], {}
    tensors = {s:[] for s in networks}
    for panel in PANELS:
        report_path = Path(f'runs/balloon-dense-supported-gated-{panel}-20261011.json')
        report = read_json(report_path)
        source_reports[str(report_path)] = sha256(report_path)
        if (report['classifier_sha256'] != sha256(classifier)
                or report['policy_sha256'] != sha256('scripts/research_dense_supported_selection.py')
                or report['queue_sha256'] != sha256(f'data/{panel}/review.json')
                or report['review_sha256'] != sha256(f'configs/{panel}.json')):
            raise ValueError('Frozen report identity changed')
        frames, _ = reviewed_frames('data/'+panel, 'configs/'+panel+'.json')
        expected = {r['id']:r for r in report['results']['supported47']['frames']}
        selected = {s:{r['id']:r for r in report['results']['supported'+s]['frames']} for s in networks}
        for row, rgb in frames:
            if not expected[row['id']]['dense_calls']:
                continue
            raw = classify_candidates(predictor, rgb, experimental(rgb, 'mser'))
            if not candidate_pairs(raw):
                raise ValueError('Previously gated frame no longer calls localizer')
            inputs.append(prepare_input(rgb)[0])
            predictions = {}
            for seed, net in networks.items():
                learned = net.predict(rgb)
                tensors[seed].append(net.interpreter.get_tensor(net.output['index'])[0])
                found = supported_selection(raw, learned)
                ref = selected[seed][row['id']]
                same_output(ref, dict(ref, detections=found))
                predictions[seed] = found
            rows.append(dict(panel=panel, id=row['id'], frame_sha256=row['sha256'],
                             raw=raw, selected=predictions))
    np.save(output / 'inputs.npy', np.asarray(inputs), allow_pickle=False)
    for seed, values in tensors.items():
        np.save(output / f'host-{seed}.npy', np.asarray(values), allow_pickle=False)
    write_json(output / 'golden.json', rows)
    shutil.copy2('src/dtr/native_tflite.py', output / 'native_tflite.py')
    shutil.copy2('scripts/pi_dense_compatibility_probe.py', output / 'probe.py')
    files = {p.name:sha256(p) for p in output.iterdir() if p.is_file()}
    write_json(output / 'manifest.json', dict(files=files, frames=len(rows), models=models,
        source_reports=source_reports, classifier_sha256=sha256(classifier),
        policy_sha256=sha256('scripts/research_dense_supported_selection.py'),
        decoder_sha256=sha256('scripts/audit_balloon_dense_proposals.py'),
        scope='Repeated development gated tensor compatibility only', deployment_approved=False))
    return len(rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    print(prepare(p.parse_args().output))
