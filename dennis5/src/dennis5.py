
#import mnist dataset, and assign mnist object
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#import tensorflow, and start an interactive session so we can use our computational graph
import tensorflow as tf

import layers
from layers import *

import weight_inits
from weight_inits import *

import bias_inits
from bias_inits import *

#import costs
#from costs import *

class Network(self, layers, cost=log_likelihood, weight_init=glorot_bengio_init, bias_init=standard)
self.sess = tf.InteractiveSession()

#Initialize symbolic variables,
#x as a float input of shape 784,
#And y as a float output of shape 10
x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])#Our desired output

#Dropout percentage
keep_prob = tf.placeholder(tf.float32)

#Initialize layers
layers = [
        ConvPoolLayer(
            [5, 5, 1, 32], 
            [-1, 28, 28, 1], 
            conv_padding='VALID', 
            pool_padding='VALID', 
            activation_fn=tf.nn.relu
        ), 
        ConvPoolLayer(
            [5, 5, 32, 64], 
            [-1, 12, 12, 32], 
            conv_padding='VALID', 
            pool_padding='VALID', 
            activation_fn=tf.nn.relu
        ),
        FullyConnectedLayer(4*4*64, 10, keep_prob, activation_fn=tf.nn.sigmoid),
        SoftmaxLayer(10, 10)
        ]

#Propogate our input forwards throughout the network layers
init_layer = layers[0]
init_layer.evaluate(x)
for i in xrange(1, len(layers)):
    prev_layer, layer = layers[i-1], layers[i]
    layer.evaluate(prev_layer.output)
output = layers[-1].output


cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(output), reduction_indices=[1]))

#train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(y,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Initialize all the variables into our computational graph
self.sess.run(tf.initialize_all_variables())

for i in range(1000):
    batch = mnist.train.next_batch(50)
    #if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={ x:batch[0], y: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))

    train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})

    print("test accuracy %g" % accuracy.eval(feed_dict={ x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))

