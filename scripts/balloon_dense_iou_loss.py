"""Training-only overlap loss; does not change the inference graph.

GIoU definition: https://arxiv.org/abs/1902.09630 . This experiment adds
2 * mean(1-GIoU) to the unchanged focal + 5*L1 objective. Weight fixed before
training/evaluation; only explicitly reviewed, non-colliding centers contribute.
"""


def generalized_iou(a,b):
    """Elementwise axis-aligned XYXY GIoU, finite for degenerate empty cells."""
    import tensorflow as tf
    def area(lo,hi):
        return tf.reduce_prod(tf.maximum(hi-lo,0.),axis=-1)
    inter = area(tf.maximum(a[...,:2],b[...,:2]),tf.minimum(a[...,2:],b[...,2:]))
    union = area(a[...,:2],a[...,2:])+area(b[...,:2],b[...,2:])-inter
    enclosing = area(tf.minimum(a[...,:2],b[...,:2]),tf.maximum(a[...,2:],b[...,2:]))
    return tf.math.divide_no_nan(inter,union)-tf.math.divide_no_nan(enclosing-union,enclosing)


def geometry_boxes(geometry):
    """Decode 15x20 cell offsets and normalized sizes in unit image coordinates."""
    import tensorflow as tf
    xx,yy = tf.meshgrid(tf.range(20,dtype=tf.float32),tf.range(15,dtype=tf.float32))
    center = (geometry[...,:2]+tf.stack([xx,yy],axis=-1))/tf.constant([20.,15.])
    radius = geometry[...,2:]/2
    return tf.clip_by_value(tf.concat([center-radius,center+radius],axis=-1),0.,1.)


def overlap_penalty(target,prediction):
    import tensorflow as tf
    valid = target[...,8]>0
    # Select centers before evaluating GIoU. Unknown/background geometry has no
    # loss or gradient, and a batch with no positive centers has zero penalty.
    truth = tf.boolean_mask(geometry_boxes(target[...,4:8]),valid)
    boxes = tf.boolean_mask(geometry_boxes(prediction[...,2:]),valid)
    total = tf.reduce_sum(1-generalized_iou(truth,boxes))
    return total/tf.maximum(tf.cast(tf.shape(truth)[0],tf.float32),1.)


def giou_loss(target,prediction):
    from scripts.train_balloon_dense_proposals import sparse_loss
    return sparse_loss(target,prediction)+2*overlap_penalty(target,prediction)
