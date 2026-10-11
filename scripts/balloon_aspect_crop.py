"""One fixed aspect-preserving classifier input policy; no proposal geometry change."""
import numpy as np
from PIL import Image


def resized(pixels, aspect=True):
    pixels = np.asarray(pixels)
    if (pixels.ndim != 3 or pixels.shape[-1] != 3 or min(pixels.shape[:2]) < 1
            or pixels.dtype != np.uint8):
        raise ValueError('Require nonempty uint8 RGB crop')
    im = Image.fromarray(pixels)
    if not aspect:
        return np.asarray(im.resize((64,64),Image.Resampling.BILINEAR),np.float32)
    h,w = pixels.shape[:2]
    scale = 64/max(h,w)
    width,height = max(1,round(w*scale)),max(1,round(h*scale))
    out = np.full((64,64,3),127,np.uint8)
    x,y = (64-width)//2,(64-height)//2
    out[y:y+height,x:x+width] = np.asarray(im.resize((width,height),Image.Resampling.BILINEAR))
    return out.astype(np.float32)
