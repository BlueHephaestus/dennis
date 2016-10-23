import numpy as np
import tensorflow as tf

def glorot_bengio(shape, n_in):
    return tf.Variable(tf.truncated_normal(shape, stddev=tf.sqrt(1.0/n_in)))

def standard(shape, n_in):
    return tf.Variable(tf.truncated_normal(shape, stddev=1.0))


