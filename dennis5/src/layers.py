import sys
import numpy as np
import tensorflow as tf

import weight_inits
from weight_inits import *

import bias_inits
from bias_inits import *

"""
Convoluting 2D

Take the arguments:
    x, a 4d tensor from the previous layer
    w, our filter/shared weights to apply
    strides, a list of length 2, with the strides in the x, y directions
    padding, with the type of padding ('SAME' or 'VALID')

You can make these the 4d that tensorflow allows, however I don't see why you (or I) would ever want to skip over samples in a mini batch or inputs.
"""
def conv2d(x, w, strides, padding):
    return tf.nn.conv2d(x, w, strides=[1, strides[0], strides[1], 1], padding=padding)

"""
Pooling 2D

Take the arguments:
    x, a 4d tensor from the previous convolutional layer
    kernel, a list of length 2, with the size of the pooling 
    strides, a list of length 2, with the strides in the x, y directions
    padding, with the type of padding ('SAME' or 'VALID')

You can make these the 4d that tensorflow allows, however I don't see why you (or I) would ever want to skip over samples in a mini batch or inputs.
"""
def pool2d(x, kernel, strides, padding, mode):
    if mode == 'max':
        return tf.nn.max_pool(x, ksize=[1, kernel[0], kernel[1], 1], strides=[1, strides[0], strides[1], 1], padding=padding)

    elif mode == 'avg':
        return tf.nn.avg_pool(x, ksize=[1, kernel[0], kernel[1], 1], strides=[1, strides[0], strides[1], 1], padding=padding)

    else:
        sys.exit("Invalid pool mode")
        
"""
Convolutional + Pooling Layer

filter_shape is a list of length 4, whose entries are the number
of filters, the number of input feature maps, the filter height, and the
filter width

image_shape is a list of length 4, whose entries are the
mini-batch size, the number of input feature maps, the image
height, and the image width

conv_strides is a list of length 2, containing the x and y strides for our convolutional layer

conv_padding is either 'SAME' or 'VALID', the padding for our convolutional layer

pool_kernel is a list of length 2, containing the x and y strides for our pooling layer

pool_size is a list of length 2, containing the x and y pooling sizes
    (Just set poolsize = [1, 1] for no pooling)

pool_padding is either 'SAME' or 'VALID', the padding for our pooling layer

pool_mode is our choice of pooling, an object such as MaxPool, AvgPool, etc.

activation_fn is our activation function

"""
class ConvPoolLayer(object):

    def __init__(self, filter_shape, image_shape, conv_strides=[1,1], conv_padding='SAME', pool_kernel = [2, 2], pool_size=[2, 2], pool_padding='SAME', pool_mode='max', activation_fn=tf.nn.sigmoid, weight_init=glorot_bengio, bias_init=standard):

        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.conv_strides = conv_strides
        self.conv_padding = conv_padding
        self.pool_kernel = pool_kernel
        self.pool_size = pool_size
        self.pool_padding = pool_padding
        self.pool_mode = pool_mode
        self.activation_fn = activation_fn

    def evaluate(self, x, keep_prob, weight_init, bias_init):
        #Reshape our x
        self.inpt = tf.reshape(x, self.image_shape)

        """
        Set our weights with our weight_init object, given weight filter and input shape
            We do [1:] because we are getting the prod of all the elements in it for our
            Glorot Bengio Init, which needs the number of inputs (we don't include mb_n)
        Note: these need to stay in evaluate so that we can get new weights every time we want to do a sample run,
            and not accidentally get sampling bias as a result of low sample count
        """
        self.w = weight_init(self.filter_shape, np.prod(self.image_shape[1:]))

        #Set our weights using our bias init object, given the depth for convpool layer
        self.b = bias_init([self.filter_shape[0]])
        #self.b = tf.Variable(tf.truncated_normal(self.filter_shape[0], stddev=1.0))

        #Get output of convolutional layer
        conv_out = self.activation_fn(conv2d(self.inpt, self.w, self.conv_strides, self.conv_padding))

        #Get output of pooling layer
        #No dropout in the convolutional layers, return output
        self.output = pool2d(conv_out, self.pool_kernel, self.pool_size, self.pool_padding, self.pool_mode)

class FullyConnectedLayer(object):

    def __init__(self, n_in, n_out, weight_init=glorot_bengio, bias_init=standard, activation_fn=tf.nn.sigmoid):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn

    def evaluate(self, x, keep_prob, weight_init, bias_init):
        #Reshape our x
        self.inpt = tf.reshape(x, [-1, self.n_in])

        self.w = weight_init([self.n_in, self.n_out], self.n_in)
        self.b = bias_init([self.n_out])

        #Apply operations
        fc_out = self.activation_fn(tf.matmul(self.inpt, self.w) + self.b)

        #Apply dropout, obtain our output
        self.output = tf.nn.dropout(fc_out, keep_prob)

class SoftmaxLayer(object):

    def __init__(self, n_in, n_out, weight_init=glorot_bengio, bias_init=standard):
        self.n_in = n_in
        self.n_out = n_out

    def evaluate(self, x, keep_prob, weight_init, bias_init):
        #Reshape our x
        self.inpt = tf.reshape(x, [-1, self.n_in])

        self.w = weight_init([self.n_in, self.n_out], self.n_in)
        self.b = bias_init([self.n_out])

        #Apply operations
        #No dropout in softmax layers, return output
        self.output = tf.nn.softmax(tf.matmul(self.inpt, self.w) + self.b)


