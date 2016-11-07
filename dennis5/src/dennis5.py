import gzip, pickle

import tensorflow as tf

import layers
from layers import *

import costs
from costs import *

import optimizers
from optimizers import init_optimizer

import regularizers
from regularizers import regularizer

import weight_inits
from weight_inits import *

import bias_inits
from bias_inits import *

class Network(object):

    def __init__(self, layers, input_dims, output_dims, dataset, cost_type=cross_entropy, weight_init=glorot_bengio, bias_init=standard):

        self.layers = layers
        self.dataset = dataset
        self.cost_type = cost_type
        self.weight_init = weight_init
        self.bias_init = bias_init

        #Initialize interactive session
        self.sess = tf.InteractiveSession()

        #Initialize symbolic variables,
        #x as a float input of shape input_dims,
        #And y as a float output of shape output_dims
        self.x = tf.placeholder(tf.float32, shape=[None, input_dims])
        self.y = tf.placeholder(tf.float32, shape=[None, output_dims])#Our desired output, not actual

        #Dropout percentage
        self.keep_prob = tf.placeholder(tf.float32)

        #Propogate our input forwards throughout the network layers
        init_layer = layers[0]
        init_layer.evaluate(self.x, self.keep_prob, weight_init, bias_init)
        for i in xrange(1, len(layers)):
            prev_layer, layer = layers[i-1], layers[i]
            layer.evaluate(prev_layer.output, self.keep_prob, weight_init, bias_init)
        self.output = layers[-1].output

        #Get a collection of our parameters for various things such as regularization later on
        self.params = []
        for layer in layers:
            self.params.append(layer.w)
            self.params.append(layer.b)

        #Initialize our saver now that we have  all the parameters in our model, so we can save later if we choose to.
        self.saver = tf.train.Saver(self.params)

    """
    Optimize our cost function.

    epochs is the number of epochs, or training steps
    mb_n is the mini batch size
    optimization_type is the type of optimization to use,
        'sgd' = normal stochastic gradient descent
        'momentum' = stochastic gradient descent + momentum
        'adam' = adam optimizer
    optimization_term is the term to be passed to our optimization, e.g.
        would be learning rate for 'sgd',
        or momentum coefficient for 'momentum'
    optimization_term_decay_rate is the rate we exponentially decay our optimization term
    regularization_rate is our regularization rate
    keep_prob is the probability we keep a neuron, the (1 - dropout percentage.)
    """
    def optimize(self, output_config, epochs, mb_n, optimization_type='gd', initial_optimization_term1=0.0, optimization_term1_decay_rate=1.0, optimization_term2=0.0, optimization_term3=0.0, optimization_defaults=True, regularization_type='l2', regularization_rate=0.0, keep_prob=1.0):

        #Initialize our final output dict for this run
        output_dict = {}
        
        #Initialize current step number to 0, so we can increment it and use it for our exponential decay
        current_step = tf.Variable(0, trainable=False)

        #Set up our exponential decay of our optimization term
        #staircase=False because it doesn't decay correctly if True
        optimization_term1 = tf.train.exponential_decay(initial_optimization_term1, current_step, epochs, optimization_term1_decay_rate, staircase=False)

        #Initialize our optimizer if it's a string keyword argument,
        #   otherwise we were given the tensorflow object
        if type(optimization_type) == str:
            optimizer = init_optimizer(optimization_type, optimization_term1, optimization_term2, optimization_term3, defaults=optimization_defaults)
        else:
            optimizer = optimization_type

        #Initialize our regularization term
        regularization_term = regularizer(regularization_type, regularization_rate, self.params)
        
        #Set the value of our cost function passed, now that we have our regularization term
        self.cost = self.cost_type.evaluate(self.output, self.y, regularization_term)

        #Initialize our training function. We pass in global step so as to increment our current step each time this is called.
        train_step = optimizer.minimize(self.cost, global_step=current_step)

        #Initialize our accuracy function
        #Note: We keep our keep_prob at 1 when computing accuracy, since we want the entire network.
        correct_prediction = tf.equal(tf.argmax(self.output,1), tf.argmax(self.y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        #Initialize all the variables into our computational graph
        self.sess.run(tf.initialize_all_variables())

        for step in range(epochs):
            #For progress
            if step % 100 == 0:
                sys.stdout.write("\r\tSTEP %i" % step)
                sys.stdout.flush()

            #Get our next mb_n number of samples
            """
            I wasn't sure why TensorFlow did mini batch training different in their MNIST examples, 
                but then I learned that using all the data every training step is much more expensive,
                and just getting a new batch each time is "[cheaper] and has much of the same benefit". 

            So that's pretty awesome, and I wish I had known that for Dennis MK4 and before. 
                So from now on, we only have to shuffle and grab a new batch each time, so much easier.

            Here's the documentation I'm citing, from beginner MNIST tutorial:
                Each step of the loop, we get a "batch" from our training set. We run train_step feeding in the batches data to replace the placeholders.

                Using small batches of random data is called stochastic training -- in this case, stochastic gradient descent. Ideally, we'd like to use all our data for every step of training because that would give us a better sense of what we should be doing, but that's expensive. So, instead, we use a different subset every time. Doing this is cheap and has much of the same benefit.
            """
            batch = self.dataset.next_batch(mb_n)

            #Run training step with placeholder values
            train_step.run(feed_dict={self.x: batch[0], self.y: batch[1], self.keep_prob: keep_prob})

            #Get our cost
            step_cost = self.cost.eval(feed_dict={ self.x: batch[0], self.y: batch[1], self.keep_prob: 1.0})

            #Get our various data accuracies
            "Change this to evaluate on the entire training dataset if you see it fit"
            training_accuracy = accuracy.eval(feed_dict={ self.x: batch[0], self.y: batch[1], self.keep_prob: 1.0}) * 100
            validation_accuracy = accuracy.eval(feed_dict={ self.x: self.dataset.validation.x, self.y: self.dataset.validation.y, self.keep_prob: 1.0}) * 100
            test_accuracy = accuracy.eval(feed_dict={ self.x: self.dataset.test.x, self.y: self.dataset.test.y, self.keep_prob: 1.0}) * 100

            #Initialize empty list for our outputs at this step
            output_dict[step] = []

            #Add our output types 
            #We convert to float because we can't json serialize np.float32, we can only do that with python's float
            if output_config['output_cost']:
                output_dict[step].append(float(step_cost))
            if output_config['output_training_accuracy']:
                output_dict[step].append(float(training_accuracy))
            if output_config['output_validation_accuracy']:
                output_dict[step].append(float(validation_accuracy))
            if output_config['output_test_accuracy']:
                output_dict[step].append(float(test_accuracy))

        print ""
        if output_config['output_cost']:
            print "\tCost: %f" % (step_cost)
        if output_config['output_training_accuracy']:
            print"\tTraining Accuracy: %f%%" % (training_accuracy)
        if output_config['output_validation_accuracy']:
            print"\tValidation Accuracy: %f%%" % (validation_accuracy)
        if output_config['output_test_accuracy']:
            print"\tTest Accuracy: %f%%" % (test_accuracy)
        print "\tOPTIMIZATION COMPLETE"

        return output_dict

    def save(self, filename, mean, stddev, nn_layers, input_dims, output_dims, cost_type, weight_init, bias_init): 

        #Adjust for file paths
        filename = "../saved_networks/%s" % filename

        #Save our network params using our previously initialized saver
        self.saver.save(self.sess, filename)

        #Save the rest of our metadata in a pkl so we can open it up easily again later
        metadata = [mean, stddev, nn_layers, input_dims, output_dims, cost_type, weight_init, bias_init]
        metadata_filename = "%s_metadata.pkl.gz" % filename
        f = gzip.open(metadata_filename, "wb")
        pickle.dump((metadata), f, protocol=-1)
        f.close()

    def restore(self, filename):
        #Restore our network from the filename given
        self.sess.run(tf.initialize_all_variables())
        self.saver.restore(self.sess, filename)

    def predict(self, x):
        #Predict output given our test x inputs and return
        predictions = self.output.eval(feed_dict={self.x: x, self.keep_prob: 1.0})
        return predictions
