
import numpy as np
import tensorflow as tf

def regularizer(regularization_type, regularization_rate, params):
    """
    Our params all have different dimensions.

    So, we loop through each w or b in our params, apply operation, sum, then use the resulting scalar
        for the next, final sum in either L1 or L2 regularization
    """
    
    if regularization_type == 'l1':
        return regularization_rate * tf.reduce_sum([tf.reduce_sum(tf.abs(param)) for param in params])

    if regularization_type == 'l2':
        return regularization_rate * tf.reduce_sum([tf.reduce_sum(tf.square(param)) for param in params])/2.0

