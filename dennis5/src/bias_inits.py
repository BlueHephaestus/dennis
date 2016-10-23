import numpy as np
import tensorflow as tf

def standard(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=1.0))


