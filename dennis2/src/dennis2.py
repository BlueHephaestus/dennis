"""network3.py
~~~~~~~~~~~~~~

A Theano-based program for training and running simple neural
networks.

Supports several layer types (fully connected, convolutional, max
pooling, softmax), and activation functions (sigmoid, tanh, and
rectified linear units, with more easily added).

When run on a CPU, this program is much faster than network.py and
network2.py.  However, unlike network.py and network2.py it can also
be run on a GPU, which makes it faster still.

Because the code is based on Theano, the code is different in many
ways from network.py and network2.py.  However, where possible I have
tried to maintain consistency with the earlier programs.  In
particular, the API is similar to network2.py.  Note that I have
focused on making the code simple, easily readable, and easily
modifiable.  It is not optimized, and omits many desirable features.

This program incorporates ideas from the Theano documentation on
convolutional neural nets (notably,
http://deeplearning.net/tutorial/lenet.html ), from Misha Denil's
implementation of dropout (https://github.com/mdenil/dropout ), and
from Chris Olah (http://colah.github.io ).

"""

#### Libraries
# Standard library
import cPickle
import gzip

# Third-party libraries
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import downsample

# Activation functions for neurons
def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh


#### Constants
GPU = True
if GPU:
    print "Trying to run under a GPU.  If this is not desired, then modify "+\
        "network3.py\nto set the GPU flag to False."
    try: theano.config.device = 'gpu'
    except: pass # it's already set
    theano.config.floatX = 'float32'
else:
    print "Running with a CPU.  If this is not desired, then the modify "+\
        "network3.py to set\nthe GPU flag to True."

def generate_input_normalizer(training_data):
    #we initialize our inputs with a normal distribution - this works by generating a normal distribution based on the mean and standard deviation of our training data since it should be a reasonable way to generalize for test data and so on. It helps to make it a normal distribution so that we can most of the time keep our neurons from saturating straight off, just as we do with weights and biases. Just needed to write this out to make sure I gots it
    #See our written notes
    '''The following line is basically:
    for sample in training_data[0]:
        for frame in sample:
            return frame
    '''
    input_x = [frame for sample in training_data[0] for frame in sample]#for sample in x: for frame in x: return frame
    mean = sum(input_x)/float(len(input_x))
    stddev = np.linalg.norm(input_x-mean)/np.sqrt(len(input_x))
    return mean, stddev

def normalize_input(data, mean, stddev):
    data[0] = (data[0]-mean)/stddev
    return data

#### Load the data
def load_data_shared(filename="../data/mnist.pkl.gz", normalize_x=False):
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = cPickle.load(f)

    if normalize_x:
        #normalize input data.
        input_normalizer_mean, input_normalizer_stddev = generate_input_normalizer(training_data)

        training_data = normalize_input(training_data, input_normalizer_mean, input_normalizer_stddev)
        validation_data = normalize_input(validation_data, input_normalizer_mean, input_normalizer_stddev)
        test_data = normalize_input(test_data, input_normalizer_mean, input_normalizer_stddev)

    def shared(data):
        """Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.

        """
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")
    return [shared(training_data), shared(validation_data), shared(test_data)]

#### Main class used to construct and train networks
class Network(object):

    def __init__(self, layers, mini_batch_size):
        """Takes a list of `layers`, describing the network architecture, and
        a value for the `mini_batch_size` to be used during training
        by stochastic gradient descent.

        """
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        #self.params = [param for layer in self.layers for param in layer.params]

        #BEGIN MOMENTUM IMPLEMENTATION
        #Start them both off with the same things, we just SHOULDN'T UPDATE the weights and biases
        #with our velocities, we only update the params
        #should be fine right now, having them both be the same shared variables at first
        #self.velocities = [velocity for layer in self.layers for velocity in layer.params]
        #print np.array(self.params).shape
        '''
        self.w = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=np.sqrt(1.0/n_out), size=filter_shape),
                dtype=theano.config.floatX),
            borrow=True)
        '''
        '''
        self.velocities = theano.shared( 
            np.zeros( 
                shape=self.params, 
                dtype=theano.config.floatX
            ), 
            borrow=True)
        '''
        #self.velocities = theano.shared(np.zeros_like(self.params))
        self.params = [param for layer in self.layers for param in layer.params]
        self.velocities = theano.clone(self.params)
        #END MOMENTUM IMPLEMENTATION

        self.x = T.matrix("x")
        self.y = T.ivector("y")
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        for j in xrange(1, len(self.layers)):
            prev_layer, layer = self.layers[j-1], self.layers[j]
            layer.set_inpt(
                prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout

    def output_config(self, output_filename, training_data_subsections, early_stopping, output_training_cost, output_training_accuracy, output_validation_accuracy, output_test_accuracy, print_results, config_index, config_count, run_index, run_count, output_types):
        
        #Set all our things for graph output
        self.output_filename=output_filename
        self.training_data_subsections=training_data_subsections,
        self.early_stopping=early_stopping
        self.output_training_cost=output_training_cost
        self.output_training_accuracy=output_training_accuracy
        self.output_validation_accuracy=output_validation_accuracy
        self.output_test_accuracy=output_test_accuracy
        self.print_results=print_results
        self.config_index=config_index
        self.config_count=config_count
        self.run_index=run_index
        self.run_count=run_count
        self.output_types=output_types


    def SGD(self, output_dict, training_data, epochs, mini_batch_size, eta,
            validation_data, test_data, lmbda=0.0, momentum_coefficient=0.0):
        """Train the network using mini-batch stochastic gradient descent."""
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        # compute number of minibatches for training, validation and testing
        num_training_batches = size(training_data)/mini_batch_size
        num_validation_batches = size(validation_data)/mini_batch_size
        num_test_batches = size(test_data)/mini_batch_size

        # define the (regularized) cost function, symbolic gradients, and updates
        l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])
        cost = self.layers[-1].cost(self)+\
               0.5*lmbda*l2_norm_squared/num_training_batches

        #grads = T.grad(cost, self.params)
        #updates = [(param, param-eta*grad) for param, grad in zip(self.params, grads)]
        #BEGIN MOMENTUM IMPLEMENTATION

        #need to compute our changes relative to params, never the velocities
        grads = T.grad(cost, self.params)
        self.velocities = [momentum_coefficient*velocity-eta*grad for velocity, grad in zip(self.velocities, grads)]
        updates = [(param, param+velocity) for param, velocity in zip(self.params, self.velocities)]
        #updates = [(param, param-eta*grad) for param, grad in zip(self.params, grads)]
        #grads = T.grad(cost, self.velocities)
        #this line modifies our self.velocities type to be all fucking weird and it's pissing me off
        #self.velocities = [(param, param+momentum_coefficient*velocity-eta*grad) for velocity, grad in zip(self.velocities, grads)]
        #then we update our velocities, trigger this with our functions

        '''
        def get_momentum_updates(cost, params, eta, momentum_coefficient):
            grads = T.grad(cost, params)
            self.velocities = [momentum_coefficient*velocity-eta*grad for velocity, grad in zip(self.velocities, grads)]
            updates = [(param, param+velocity) for param, velocity in zip(params, self.velocities)]
            return updates

        
        '''
        #updates = [(param, param+(momentum_coefficient*velocity-eta*grad)) for param, velocity, grad in zip(self.params, self.velocities, grads)]
        
        #END MOMENTUM IMPLEMENTATION
        # define functions to train a mini-batch, and to compute the
        # accuracy in validation and test mini-batches.
        i = T.lscalar() # mini-batch index
        train_mb = theano.function(
            [i], cost, updates=updates,
            givens={
                self.x:
                training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        train_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        train_mb_cost = theano.function(
            [i], self.layers[-1].cost(self) + 0.5*lmbda*l2_norm_squared/num_training_batches,
            givens={
                self.x:
                training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        validate_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                validation_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                validation_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        test_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                test_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        self.test_mb_predictions = theano.function(
            [i], self.layers[-1].y_out,
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        # Do the actual training
        best_training_accuracy = 0.0
        best_validation_accuracy = 0.0
        for epoch in xrange(epochs):
            for minibatch_index in xrange(num_training_batches):
                cost_ij = train_mb(minibatch_index)

            #I'd put an if statement to see if there are any outputs, but if there are no outputs, then there won't be anything to graph so there will never be a bug that arises from fucked up output_dict, never gets used
            output_dict[self.run_index][epoch] = []
            #output types
            if self.output_training_cost:
                #The rest of this line is already explained in the notes with good reason, but I added the float()
                #So that we don't get json serialization errors for using the numpy.float32 type.
                training_cost = float(np.mean([train_mb_cost(j) for j in xrange(num_training_batches)]))
                output_dict[self.run_index][epoch].append(training_cost)

            if self.output_training_accuracy:
                training_accuracy = np.mean([train_mb_accuracy(j) for j in xrange(num_training_batches)])
                training_accuracy *= 100#Percentage formatting
                output_dict[self.run_index][epoch].append(training_accuracy)

            if self.output_validation_accuracy:
                validation_accuracy = np.mean([validate_mb_accuracy(j) for j in xrange(num_validation_batches)])
                validation_accuracy *= 100#Percentage formatting
                output_dict[self.run_index][epoch].append(validation_accuracy)

            if self.output_test_accuracy:
                test_accuracy = np.mean([test_mb_accuracy(j) for j in xrange(num_test_batches)])
                test_accuracy *= 100#Percentage formatting
                output_dict[self.run_index][epoch].append(test_accuracy)

            #So we don't print until we've already computed calculations for this epoch
            print "Epoch %i" % (epoch)
            if self.print_results:
                if self.output_training_cost:
                    print "\tTraining Cost: %f" % (training_cost)
                if self.output_training_accuracy:
                    print "\tTraining Accuracy: %f%%" % (training_accuracy)
                if self.output_validation_accuracy:
                    print "\tValidation Accuracy: %f%%" % (validation_accuracy)
                if self.output_test_accuracy:
                    print "\tTest Accuracy: %f%%" % (test_accuracy)

        #Using our +1s for pretty print progress
        print "Config %i/%i, Run %i/%i Completed." % (self.config_index+1, self.config_count, self.run_index+1, self.run_count)
        return output_dict

#### Define layer types

class ConvPoolLayer(object):
    """Used to create a combination of a convolutional and a max-pooling
    layer.  A more sophisticated implementation would separate the
    two, but for our purposes we'll always use them together, and it
    simplifies the code, so it makes sense to combine them.

    """

    def __init__(self, filter_shape, image_shape, subsample=(1,1), poolsize=(2, 2),
                 activation_fn=sigmoid):
        """`filter_shape` is a tuple of length 4, whose entries are the number
        of filters, the number of input feature maps, the filter height, and the
        filter width.

        `image_shape` is a tuple of length 4, whose entries are the
        mini-batch size, the number of input feature maps, the image
        height, and the image width.

        `poolsize` is a tuple of length 2, whose entries are the y and
        x pooling sizes.

        """
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.subsample= subsample
        self.poolsize = poolsize
        self.activation_fn=activation_fn
        # initialize weights and biases
        n_out = (filter_shape[0]*np.prod(filter_shape[2:])/np.prod(poolsize))
        self.w = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=np.sqrt(1.0/n_out), size=filter_shape),
                dtype=theano.config.floatX),
            borrow=True)
        self.b = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)),
                dtype=theano.config.floatX),
            borrow=True)
        self.params = [self.w, self.b]
        #self.velocities = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape(self.image_shape)
        conv_out = conv.conv2d(
            input=self.inpt, filters=self.w, filter_shape=self.filter_shape,
            image_shape=self.image_shape, subsample=self.subsample)
        pooled_out = downsample.max_pool_2d(
            input=conv_out, ds=self.poolsize, ignore_border=True)
        self.output = self.activation_fn(
            pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output_dropout = self.output # no dropout in the convolutional layers

class FullyConnectedLayer(object):

    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=np.sqrt(1.0/n_out), size=(n_in, n_out)),
                dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                       dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]
        #self.velocities = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn(
            (1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(
            T.dot(self.inpt_dropout, self.w) + self.b)

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))

class SoftmaxLayer(object):

    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.zeros((n_out,), dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]
        #self.velocities = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = softmax((1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = softmax(T.dot(self.inpt_dropout, self.w) + self.b)

    def cost(self, net):
        "Return the log-likelihood cost."
        return -T.mean(T.log(self.output_dropout)[T.arange(net.y.shape[0]), net.y])

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))


#### Miscellanea
def size(data):
    "Return the size of the dataset `data`."
    return data[0].get_value(borrow=True).shape[0]

def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer*T.cast(mask, theano.config.floatX)
