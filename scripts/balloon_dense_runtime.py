"""Explicit research-only full-frame proposal inference, never object acceptance."""
from pathlib import Path

import cv2
import numpy as np

from dtr.data import read_json,sha256
from scripts.audit_balloon_dense_proposals import decode


def prepare_input(rgb):
    if rgb.shape!=(240,320,3) or rgb.dtype!=np.uint8:
        raise ValueError('Require 320x240 RGB uint8 scene')
    return cv2.resize(rgb,(160,120),interpolation=cv2.INTER_AREA).astype(np.float32)[None]


class DenseProposals:
    def __init__(self,path):
        self.metadata=read_json(Path(path).with_suffix('.json'))
        m=self.metadata
        if (m.get('kind')!='balloon_dense_proposal_research' or m['sha256']!=sha256(path)
                or m['deployment_approved'] is not False or m['proposal_threshold']!=.1
                or m['max_proposals']!=12 or m['max_added_blue_proposals']!=2
                or m['decoder_sha256']!=sha256('scripts/audit_balloon_dense_proposals.py')):
            raise ValueError('Dense proposal model or fixed decoder identity changed')
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            try:
                from tensorflow.lite.python.interpreter import Interpreter
            except ImportError:
                from dtr.native_tflite import Interpreter
        self.interpreter=Interpreter(model_path=str(path),num_threads=1)
        self.interpreter.allocate_tensors()
        self.input,self.output=self.interpreter.get_input_details()[0],self.interpreter.get_output_details()[0]
        if (self.input['shape'].tolist()!=[1,120,160,3] or self.output['shape'].tolist()!=[1,15,20,6]
                or self.input['dtype']!=np.float32 or self.output['dtype']!=np.float32):
            raise ValueError('Dense proposal tensor contract changed')

    def predict(self,rgb):
        self.interpreter.set_tensor(self.input['index'],prepare_input(rgb))
        self.interpreter.invoke()
        return decode(self.interpreter.get_tensor(self.output['index'])[0])
