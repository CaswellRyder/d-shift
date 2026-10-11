"""Architecture ablation: both arms receive identical admitted TRAIN data."""
from scripts.blue_photo_expansion_data import inputs as photo_inputs
def inputs(mode):
    if mode not in ('control','admit'):
        raise ValueError('Require matched architecture arm')
    return photo_inputs('admit')
def parameters(mode):
    if mode not in ('control','admit'):
        raise ValueError('Unknown architecture arm')
    return 7763 if mode=='control' else 15667
