"""Drop contradictory teacher preservation only on explicitly reviewed runtime rows."""
from scripts.balloon_all_correct_protection import all_correct_loss


def reviewed_runtime_loss(weight=8.,cap=.9,threshold=.8):
    import tensorflow as tf
    original = all_correct_loss(weight,cap,threshold)

    def loss(target,logits):
        tf.debugging.assert_equal(tf.shape(target)[1],8)
        truth = tf.argmax(target[:,:3],axis=1)
        teacher = tf.argmax(target[:,3:6],axis=1)
        reviewed = target[:,7]>.5
        conflict = reviewed & tf.not_equal(truth,teacher)
        preserve = tf.where(conflict,tf.zeros_like(target[:,6]),target[:,6])
        adjusted = tf.concat((target[:,:6],preserve[:,None]),axis=1)
        return original(adjusted,logits)

    return loss
