import tensorflow as tf

class cross_entropy(object):

    @staticmethod
    def evaluate(a, y):
        return tf.reduce_mean(-tf.reduce_sum(y * tf.log(a), reduction_indices=[1]))
