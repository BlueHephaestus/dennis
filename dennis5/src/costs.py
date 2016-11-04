import tensorflow as tf

class cross_entropy(object):

    @staticmethod
    def evaluate(a, y, regularization_term):
        #We use tf.max(a, 1e-10) to make sure we only clip if we risk doing log(0), thus avoiding NaN problems
        return tf.reduce_mean(-tf.reduce_sum(y * tf.log(tf.maximum(a, 1e-10)), reduction_indices=[1])) + regularization_term
