
import numpy as np
import tensorflow as tf

def init_regularizer(regularization_type, regularization_rate, params):
    if regularization_type == 'l1':
        return regularization_rate * tf.reduce_sum(tf.abs(params))

    if regularization_type == 'l2':
        return regularization_rate * tf.reduce_sum(params ** 2) / 2
        #return regularization_rate * tf.nn.l2_loss(params)
