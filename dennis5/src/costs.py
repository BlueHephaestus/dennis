import tensorflow as tf

class cross_entropy(object):

    @staticmethod
    def evaluate(a, y, regularization_term):
        #We use tf.clip_by_value(tensor, min_value, max_value) to make sure we never do log(0) and thus avoid NaN
        return tf.reduce_mean(-tf.reduce_sum(y * tf.log(tf.maximum(a, 1e-10)), reduction_indices=[1])) + regularization_term
