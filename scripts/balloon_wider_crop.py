"""One fixed context ablation; preserve localization and network call counts."""
import math


def wider_crop(box, shape):
    h,w=shape[:2]
    if (len(box)!=4 or any(type(v) is not int for v in box)
            or not 0<=box[0]<box[2]<=w or not 0<=box[1]<box[3]<=h):
        raise ValueError('Require valid integer crop in image')
    a,b,c,d=box
    cw=min(w,max(16,math.ceil((c-a)*1.5)))
    ch=min(h,max(16,math.ceil((d-b)*1.5)))
    x=min(max((a+c-cw)//2,0),w-cw)
    y=min(max((b+d-ch)//2,0),h-ch)
    return [x,y,x+cw,y+ch]
