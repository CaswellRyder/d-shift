"""Verify final checkpoints against pixels, frozen teacher outputs and gates."""
import hashlib
from pathlib import Path

import numpy as np

from dtr.data import read_json,sha256,write_json
from scripts.train_blue_photo_expansion import PROTOCOL,inputs,run_path,calculate

OUTPUT = Path('research/balloon-red-blue-20261008/blue-photo-expansion-training-20261011.json')


def audit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    p = read_json(PROTOCOL)
    if sha256(p['parent'])!=p['parent_sha256']:
        raise ValueError('Parent changed')
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    parent = keras.models.load_model(p['parent'],compile=False)
    names = ('','blue_','clutter_','real_','new_','runtime_','photo_')
    results = {}
    common = None
    probes = None
    for mode in p['arms']:
        rows,labels,ordinary,tensors = inputs(mode)
        teacher = [parent(x,training=False).numpy() for x in tensors]
        identity = [{k:r[k] for k in ('source_image','sha256','label','preserve','view')} for r in rows]
        base_identity = identity[:1750]
        probe_hashes = [hashlib.sha256(x.tobytes()).hexdigest() for x in tensors[1:]]
        if common is not None and (base_identity!=common or probe_hashes!=probes):
            raise ValueError('Matched arms differ outside added blue-positive training rows')
        common,probes = base_identity,probe_hashes
        for seed in p['seeds']:
            run = run_path(mode,seed)
            prov,report = read_json(run/'provenance.json'),read_json(run/'report.json')
            if (read_json(run/'status.json')['state']!='complete' or prov['mode']!=mode or prov['seed']!=seed
                    or prov['settings']!=p or prov['sample_identity']!=identity or prov['parameters']!=7763
                    or prov['optimizer_reset'] is not True or prov['final_epoch_only'] is not True
                    or any(sha256(s)!=h for s,h in prov['sources_sha256'].items())
                    or prov['tensor_sha256']!=[hashlib.sha256(x.tobytes()).hexdigest() for x in tensors]
                    or report['optimizer_steps']!=26640):
                raise ValueError('Training recipe/pixels/steps changed')
            for key,file in (('model_sha256','student.keras'),('predictions_sha256','predictions.npz'),('provenance_sha256','provenance.json')):
                if report[key]!=sha256(run/file):
                    raise ValueError('Output identity changed')
            with np.load(run/'predictions.npz') as saved:
                a = {k:saved[k] for k in saved.files}
            if not np.array_equal(a['labels'],labels) or not np.array_equal(a['ordinary'],ordinary):
                raise ValueError('Labels changed')
            model = keras.models.load_model(run/'student.keras',compile=False)
            if model.count_params()!=7763:
                raise ValueError('Architecture changed')
            for prefix,x,before in zip(names,tensors,teacher,strict=True):
                if (not np.allclose(before,a[prefix+'before'],rtol=0,atol=1e-5)
                        or not np.allclose(model(x,training=False).numpy(),a[prefix+'after'],rtol=0,atol=1e-5)):
                    raise ValueError('Checkpoint does not reproduce saved outputs')
            fresh = calculate(a)
            if any(report[k]!=v for k,v in fresh.items()):
                raise ValueError('Recomputed gate differs')
            results[f'{mode}{seed}'] = dict(path=str(run),**report)
    eligible = all(results[f'admit{s}']['training_gate_pass'] for s in p['seeds'])
    write_json(OUTPUT,dict(trials=results,eligible_for_development=eligible,
        protocol_sha256=sha256(PROTOCOL),script_sha256=sha256(__file__),
        decision='Export all paired models for fixed development evaluation' if eligible else 'No export: paired training gate failed',
        scope='Repeated TRAIN crops and exact model reproduction. Not independent accuracy.',
        test_evaluated=False,pi_tested=False,deployment_approved=False,qualification_achieved=False))
    print(dict(eligible_for_development=eligible,results={k:dict(photo_blue=r['photo_blue_accepted'],runtime_blue=r['runtime_blue_accepted'],new=r['new_background_correct'],
        blue=r['blue_accepted'],old=r['old_clutter_correct'],real=r['real_blue_correct'],gate=r['training_gate_pass']) for k,r in results.items()}))


if __name__=='__main__':
    audit()
