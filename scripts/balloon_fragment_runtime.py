"""Explicit four-logit research adapter; fragments can never become targets."""
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json,sha256
from dtr.runtime import softmax
from scripts.train_balloon_fragments import CLASSES


def prediction(logits):
    logits=np.asarray(logits)
    if logits.shape!=(4,) or not np.isfinite(logits).all():
        raise ValueError('Require four finite fragment-model logits')
    scores=softmax(logits)
    best=int(scores.argmax())
    return dict(label=CLASSES[best],scores=scores.tolist(),score=float(scores[best]),
                accepted=bool(best in (1,2) and scores[best]>=.8),fragment_rejected=best==3)


class FragmentPredictor:
    def __init__(self,path):
        self.metadata=read_json(Path(path).with_suffix('.json'))
        m=self.metadata
        if (m.get('kind')!='balloon_fragment_research' or m.get('classes')!=CLASSES
                or m.get('size')!=64 or m.get('threshold')!=.8
                or m.get('deployment_approved') is not False or m.get('tensor_dtype')!='float32'
                or m.get('sha256')!=sha256(path)):
            raise ValueError('Invalid fragment research model identity')
        import tensorflow as tf
        self.interpreter=tf.lite.Interpreter(model_path=str(path),num_threads=1)
        self.interpreter.allocate_tensors()
        self.input,self.output=self.interpreter.get_input_details()[0],self.interpreter.get_output_details()[0]
        if (self.input['shape'].tolist()!=[1,64,64,3] or self.output['shape'].tolist()!=[1,4]
                or self.input['dtype']!=np.float32 or self.output['dtype']!=np.float32):
            raise ValueError('Fragment tensor contract changed')

    def predict(self,rgb):
        if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or min(rgb.shape[:2])<1:
            raise ValueError('Require nonempty RGB uint8')
        pixels=np.asarray(Image.fromarray(rgb).resize((64,64),Image.Resampling.BILINEAR),np.float32)
        self.interpreter.set_tensor(self.input['index'],pixels[None])
        self.interpreter.invoke()
        return prediction(self.interpreter.get_tensor(self.output['index'])[0])
