#In the order they describe them in the documentation - https://www.tensorflow.org/versions/r0.11/api_docs/python/train.html

import numpy as np
import tensorflow as tf

def init_optimizer(optimization_type, optimization_term1, optimization_term2, optimization_term3, defaults=True):

    if optimization_type == 'gd':
        return tf.train.GradientDescentOptimizer(learning_rate=optimization_term1)

    elif optimization_type == 'adadelta':
        if defaults:
            return tf.train.AdadeltaOptimizer()
        return tf.train.AdadeltaOptimizer(learning_rate=optimization_term1, rho=optimization_term2)

    elif optimization_type == 'adagrad':
        if defaults:
            return tf.train.AdagradOptimizer(optimization_term1)
        return tf.train.AdagradOptimizer(optimization_term1, initial_accumulator_value=optimization_term2)

    elif optimization_type == 'adagrad-da':
        if defaults:
            return tf.train.AdagradDAOptimizer(optimization_term1)
        return tf.train.AdagradOptimizer(optimization_term1, initial_gradient_squared_accumulator_value=optimization_term2)

    elif optimization_type == 'momentum':
        return tf.train.MomentumOptimizer(optimization_term1, optimization_term2)

    elif optimization_type == 'nesterov':
        return tf.train.MomentumOptimizer(optimization_term1, optimization_term2, use_nesterov=True)

    elif optimization_type == 'adam':
        if defaults:
            return tf.train.AdamOptimizer()
        return tf.train.AdamOptimizer(learning_rate=optimization_term1, beta1=optimization_term2, beta2=optimization_term)

    elif optimization_type == 'ftrl':
        if defaults:
            return tf.train.FtrlOptimizer(optimization_term1)
        return tf.train.FtrlOptimizer(optimization_term1, learning_rate_power=optimization_term2, initial_accumulator_value=optimization_term3)

    elif optimization_type == 'rmsprop':
        if defaults:
            return tf.train.RMSPropOptimizer(optimization_term1)
        return tf.train.FtrlOptimizer(optimization_term1, decay=optimization_term2, momentum=optimization_term3)

