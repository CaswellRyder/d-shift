"""Research-only continuation inference with explicit arm, shape and decoder identity."""
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256
from scripts.balloon_dense_runtime import prepare_input
from scripts.audit_balloon_dense_proposals import decode as coarse_decode
from scripts.balloon_fine_grid import decode as fine_decode


class GridProposals:
    def __init__(self, path):
        self.metadata = read_json(Path(path).with_suffix('.json'))
        m = self.metadata
        self.arm = m.get('arm')
        if self.arm not in ('coarse', 'context'):
            raise ValueError('Unknown grid arm')
        grid = (15, 20) if self.arm == 'coarse' else (30, 40)
        decoder = 'scripts/audit_balloon_dense_proposals.py' if self.arm == 'coarse' else 'scripts/balloon_fine_grid.py'
        if (m.get('kind') != 'balloon_grid_convergence_research' or m['sha256'] != sha256(path)
                or m['deployment_approved'] is not False or m['proposal_threshold'] != .1
                or m['max_proposals'] != 12 or m['max_added_blue_proposals'] != 2
                or m['decoder_path'] != decoder or m['decoder_sha256'] != sha256(decoder)
                or m['input_shape'] != [1, 120, 160, 3] or m['output_shape'] != [1, *grid, 6]):
            raise ValueError('Grid export or decoder changed')
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            try:
                from tensorflow.lite.python.interpreter import Interpreter
            except ImportError:
                from dtr.native_tflite import Interpreter
        self.interpreter = Interpreter(model_path=str(path), num_threads=1)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]
        if (self.input['shape'].tolist() != m['input_shape'] or self.output['shape'].tolist() != m['output_shape']
                or self.input['dtype'] != np.float32 or self.output['dtype'] != np.float32):
            raise ValueError('Unexpected tensor contract')
        self.decoder = coarse_decode if self.arm == 'coarse' else fine_decode

    def predict(self, rgb):
        self.interpreter.set_tensor(self.input['index'], prepare_input(rgb))
        self.interpreter.invoke()
        return self.decoder(self.interpreter.get_tensor(self.output['index'])[0])
