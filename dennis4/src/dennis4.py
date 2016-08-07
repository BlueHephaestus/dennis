"""
DENNIS MK. 4
Authored by Blake Edwards / Dark Element,
based heavily on mnielsen's code as I credit on the github page this is hosted on currently.

Please read the README for all the info
"""
#### Libraries
# Standard library
import cPickle, pickle
import gzip
import sys

# Third-party libraries
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import pool

# Activation functions for neurons
def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh


#### Constants
GPU = True

if GPU:
    print "Running under GPU"
    try: theano.config.device = 'gpu'
    except: pass # it's already set
    theano.config.floatX = 'float32'
else:
    print "Running under CPU"

class log_likelihood(object):
    "Return the log-likelihood cost."
    #-mean( yln(a) )
    @staticmethod
    def get_cost(a, y, n):

        #When we do [T.arange(y.shape[0]), y]
        #It converts our vector of outputs for y to something like
        #[[0, 1, 2, 3, 4], [7, 2, 5, 6]]
        #Where y[0] is our index of the value
        #Where y[1] is the y value
        return -T.mean(T.log(a)[T.arange(y.shape[0]), y])

class cross_entropy(object):
    "Return Cross-Entropy Cost"
    #Best when combined with sigmoid output layer
    #Consider rewriting this to make sense in terms of matrices
    @staticmethod
    def get_cost(a, y, n):
        #-mean( y*ln(a) + (1-y)*ln(1-a) )

        #Inverted log a is (1-y)ln(1-a)
        #This is basically taking ln(1-a) and assigning all the ones that are equal to the index of y to 0,
        #Since we'd have y as [0, 0, 1, 0] then 1-y = [1, 1, 0, 1] so we keep everything the same
        #Except for those matching the index, which we make = 0
        inverted_log_a = T.set_subtensor(T.log(1-a)[T.arange(y.shape[0]), y], 0.0)
        f = theano.function(inputs=[a, y], outputs=inverted_log_a)
        return -T.mean(
                    T.log(a)[T.arange(y.shape[0]), y]
                    +
                    T.sum(inverted_log_a, axis=1)
                )

class quadratic(object):
    "Return Quadratic Cost"
    @staticmethod
    def get_cost(a, y, n):
        #     ( ||y-a||^2 )
        #mean ( --------- )
        #     (    2      )

        #a_update = (a, T.set_subtensor(a[T.arange(y.shape[0]), y], 1 - a[T.arange(y.shape[0]), y]))
        #f = function([y], updates=[a_update])

        #We have each y as a one hot vector like [0, 0, 1, 0] but from everything i've worked out so far it's better to
        #do 1 - a for the a values that are the same as y, so if we had 
        #y = [0, 0, 1, 0]
        #a = [.1, .1, .8, 0]
        #It's better to get result of 
        #y-a = [.1, .1, .2, 0], rather than getting massive values that make the cost way bigger if we were to make the .1 = 0-.1 = .9
        #since we'd then be squaring it elementwise right after, and since the closer it gets to 0, which we want(since it's not the right output)
        #the higher the value would get, i.e. 0.00001 = .99999^2, that doesn't make any sense at all. I hope i'm not doing something completely wrong,
        #But what I did was set the new a to be a with our 1-a indices of y.
        #Then we square, sum on axis 1, get mean, and then * 1/2 in accordance with the earlier equation.
        new_a = T.set_subtensor(a[T.arange(y.shape[0]), y], 1 - a[T.arange(y.shape[0]), y])
        f = theano.function(inputs=[a, y], outputs=new_a)

        return T.dot((1.0/2.0), T.mean(T.sum(T.sqr(new_a), axis=1)))


#### Main class used to construct and train networks
class Network(object):

    def __init__(self, layers, mini_batch_size, cost=log_likelihood):
        """Takes a list of `layers`, describing the network architecture, and
        a value for the `mini_batch_size` to be used during training
        by stochastic gradient descent.

        """
        self.layers = layers
        self.cost = cost
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]

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

    def output_config(self, output_filename, training_data_subsections, early_stopping, automatic_scheduling, output_training_cost, output_training_accuracy, output_validation_accuracy, output_test_accuracy, print_results, print_perc_complete, config_index, config_count, run_index, run_count, output_types):
        
        #Set all our things for graph output
        self.output_filename=output_filename
        self.training_data_subsections=training_data_subsections,
        self.early_stopping=early_stopping
        self.automatic_scheduling=automatic_scheduling
        self.output_training_cost=output_training_cost
        self.output_training_accuracy=output_training_accuracy
        self.output_validation_accuracy=output_validation_accuracy
        self.output_test_accuracy=output_test_accuracy
        self.print_results=print_results
        self.print_perc_complete=print_perc_complete
        self.config_index=config_index
        self.config_count=config_count
        self.run_index=run_index
        self.run_count=run_count
        self.output_types=output_types


    def predict(self, test_data):
        test_x, test_y = test_data

        i = T.lscalar() # mini-batch index

        self.test_mb_predictions = theano.function(
            [i], self.layers[-1].y_out,
            givens={
                self.x: test_x[:]
            }, on_unused_input='warn')

        return self.test_mb_predictions(0)

    def SGD(self, output_dict, training_data, epochs, mini_batch_size, eta,
            validation_data, test_data, optimization='momentum', 
            lmbda=0.0, optimization_term1=0.0, optimization_term2=0.0,
            scheduler_check_interval=10, param_decrease_rate=10):#Initialize early stopping stuff to reasonable defaults
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
        cost = self.layers[-1].cost(self)

        #grads = T.grad(cost, self.params)
        #updates = [(param, param-eta*grad) for param, grad in zip(self.params, grads)]
        #BEGIN MOMENTUM IMPLEMENTATION

        clip_term = 1e-7#For Adagrad, adadelta, etc.

        if optimization == 'vanilla': 
            grads = T.grad(cost, self.params)
            updates = []
            for param, grad in zip(self.params, grads):
                updates.append((param, param-eta*grad))#Update our params as we were

        elif optimization == 'momentum':
            #need to compute our changes relative to params, never the velocities
            velocities = [theano.shared(value=np.zeros(param.get_value().shape, dtype=theano.config.floatX)) for param in self.params]#Initialize our velocities as zeroes(should this be something different than zeroes?)
            grads = T.grad(cost, self.params)
            #I could make the following one line. I realize this. Who the fuck wants to read that colossus though.
            updates = []
            for param, velocity, grad in zip(self.params, velocities, grads):
                new_velocity = (optimization_term1*velocity) - (eta*grad)
                updates.append((velocity, new_velocity))#We fucking forgot this is how to update the velocity as well *facepalm*
                updates.append((param, param+new_velocity))#Update our params as we were

        elif optimization == 'nesterov':
            #Nesterov's Accelerated Gradient aka NAG
            #nag_terms = [theano.shared(value=np.zeros(param.get_value().shape, dtype=theano.config.floatX)) for param in self.params]#Initialize our velocities as zeroes(should this be something different than zeroes?)
            #nag_terms = theano.shared(value=[np.zeros(param.get_value().shape, dtype=theano.config.floatX) for param in self.params])#Initialize our velocities as zeroes(should this be something different than zeroes?)
            #nag_terms = theano.shared(value=np.zeros(self.params, dtype=theano.config.floatX))
            '''
            self.b = theano.shared(
                np.zeros((n_out,), dtype=theano.config.floatX),
                name='b', borrow=True)
            '''
            #nag_terms = [theano.shared(np.zeros_like(param, dtype=theano.config.floatX), borrow=True) for param in self.params]
            #nag_terms = [theano.shared(np.zeros_like(param, dtype=theano.config.floatX), borrow=True) for param in self.params]
            nag_terms = [T.zeros_like(self.params) for param in self.params]

            print type(self.params), type(self.params[0]), type(self.params[0][0])
            print type(nag_terms), type(nag_terms[0]), type(nag_terms[0][0])

            
            nag_params = [param + optimization_term1*nag_term for param, nag_term in zip(self.params, nag_terms)]
            #nag_params = theano.scan(lambda p, n: p + optimization_term1*n, sequences=[self.params, nag_terms])
            #f = theano.function(inputs=[self.params, nag_terms], outputs=nag_params)

            grads = T.grad(cost, nag_params, disconnected_inputs='warn')


            updates = []
            for param, nag_term, grad in zip(self.params, nag_terms, grads):
                updates.append((nag_term, optimization_term1*nag_term-eta*grad))
                updates.append((param, param+nag_term))#Update our params as we were


        elif optimization == 'adagrad':
            old_grads = [theano.shared(value=np.zeros(param.get_value().shape, dtype=theano.config.floatX)) for param in self.params]#Initialize our history of grads as zeroes(should this be something different than zeroes?)
            grads = T.grad(cost, self.params)
            updates = []
            for param, old_grad, grad in zip(self.params, old_grads, grads):
                #new_velocity = (optimization_term1*velocity) - (eta*grad)
                #updates.append((velocity, new_velocity))#We fucking forgot this is how to update the velocity as well *facepalm*
                adagrad_term = eta/T.sqrt(old_grad + clip_term) * grad
                updates.append((old_grad, old_grad+T.sqr(grad)))
                updates.append((param, param-adagrad_term))#Update our params as we were

        elif optimization == 'adadelta':
            """
                     RMS(p_update)
            p' = p + ------------- * grad
                       RMS(grad)
            """
            #Should these be ones instead of zeros?
            avg_updates = [theano.shared(value=np.zeros(param.get_value().shape, dtype=theano.config.floatX)) for param in self.params] 
            avg_grads = [theano.shared(value=np.zeros(param.get_value().shape, dtype=theano.config.floatX)) for param in self.params]

            grads = T.grad(cost, self.params)
            updates = []
            for param, avg_update, avg_grad, grad in zip(self.params, avg_update, avg_grads, grads):

                new_avg_update = optimization_term1*(avg_update) + (1.0-optimization_term1)*(param**2)#Our E(p^2) term, does it need something other than param**2 at the end? We can't give the update because we don't have it yet
                new_avg_grad = optimization_term1*(avg_grad) + (1.0-optimization_term1)*(grad**2)#Our E(g^2) term

                updates.append((avg_update, new_avg_update))
                updates.append((avg_grad, new_avg_grad))

                new_param = -(T.sqrt(new_avg_update + clip_term)/T.sqrt(new_avg_grad + clip_term)) * grad
                updates.append((param, param+new_param))

        elif optimization == 'rmsprop':
            """
                        -n
            p' = p + --------- * grad
                     RMS(grad)
            """
            #Should these be ones instead of zeros?
            avg_grads = [theano.shared(value=np.zeros(param.get_value().shape, dtype=theano.config.floatX)) for param in self.params]

            grads = T.grad(cost, self.params)
            updates = []
            for param, avg_grad, grad in zip(self.params, avg_grads, grads):

                new_avg_grad = optimization_term1*(avg_grad) + (1.0-optimization_term1)*(grad**2)#Our E(g^2) term

                updates.append((avg_grad, new_avg_grad))

                new_param = -(eta/T.sqrt(new_avg_grad + clip_term)) * grad
                updates.append((param, param+new_param))

        elif optimization == 'adam':
            """
                            grad
            modified_grad2 = ----
                            1-u1

                            grad^2
            modified_grad2 = ----
                            1-u2

                            -n
            p' = p + ------------------ * modified_grad1
                     RMS(modified_grad2)
            """

            avg_grads1 = [theano.shared(value=np.zeros(param.get_value().shape, dtype=theano.config.floatX)) for param in self.params]#Normal grad
            avg_grads2 = [theano.shared(value=np.zeros(param.get_value().shape, dtype=theano.config.floatX)) for param in self.params]#Squared grad
                
            grads = T.grad(cost, self.params)
            updates = []
            for param, avg_grad1, avg_grad2, grad in zip(self.params, avg_grads1, avg_grads2, grads):

                new_avg_grad1 = optimization_term1*(avg_grad1) + (1.0-optimization_term1)*(grad)
                new_avg_grad2 = optimization_term1*(avg_grad2) + (1.0-optimization_term1)*(grad**2)

                new_avg_grad1 = new_avg_grad1 / (1.0 - optimization_term1)
                new_avg_grad2 = new_avg_grad2 / (1.0 - optimization_term2)

                updates.append((avg_grad1, new_avg_grad1))
                updates.append((avg_grad2, new_avg_grad2))

                new_param = -(eta/T.sqrt(new_avg_grad1 + clip_term)) * new_avg_grad2
                updates.append((param, param+new_param))



        
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

        #Changed to look at validation % instead of training cost
        if (self.early_stopping or self.automatic_scheduling) and self.output_validation_accuracy:
            scheduler_results = []
            #Arange so we can do our vector multiplication
            scheduler_x = np.arange(1, scheduler_check_interval+1)
            param_stop_threshold = eta * (param_decrease_rate*10**-6)
            #param_stop_threshold = mini_batch_size * (param_decrease_rate*10**-6)

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

            if self.print_perc_complete:
                #Get our percentage completion
                perc_complete = (((self.config_index)/(float(self.config_count))) + (((self.run_index)/float(self.run_count))/float(self.config_count)) + (((epoch/float(epochs))/float(self.run_count))/float(self.config_count))) * 100.0
                sys.stdout.write("\r%f%% Complete: Config %i/%i, Run %i/%i, Epoch %i/%i" % (perc_complete, self.config_index+1, self.config_count, self.run_index+1, self.run_count, epoch+1, epochs))
            else:
                sys.stdout.write("\rConfig %i/%i, Run %i/%i, Epoch %i/%i" % (self.config_index+1, self.config_count, self.run_index+1, self.run_count, epoch+1, epochs))
            sys.stdout.flush()

            if self.print_results:
                print ""
                if self.output_training_cost:
                    print"\tTraining Cost: %f" % (training_cost)
                if self.output_training_accuracy:
                    print"\tTraining Accuracy: %f%%" % (training_accuracy)
                if self.output_validation_accuracy:
                    print"\tValidation Accuracy: %f%%" % (validation_accuracy)
                if self.output_test_accuracy:
                    print"\tTest Accuracy: %f%%" % (test_accuracy)

            if (self.early_stopping or self.automatic_scheduling) and self.output_validation_accuracy:
                #This is where we change things according to the parameter we want to schedule, since I think it would take a hell of a lot to make it automatic for scheduler and param
                scheduler_results.append(validation_accuracy/100.0)#Convert accuracy back to normal (instead of percentage formatting) here
                if len(scheduler_results) >= scheduler_check_interval:
                    #Do our checks on the last check_interval number of accuracy results(which is the size of the array)
                    #Checks = 
                        #get average slope of interval
                            #(interval * sigma(x*y) - sigma(x)*sigma(y)) / (interval * sigma(x**2) - (sigma(x))**2)
                            #where x is each # in our interval 1, 2, 3... interval
                            # and y is each of our accuracies
                    scheduler_avg_slope = (scheduler_check_interval*1.0*sum(scheduler_x*scheduler_results) - sum(scheduler_x) * 1.0 * sum(scheduler_results))/(scheduler_check_interval*sum([x**2 for x in scheduler_x])*1.0 - (sum(scheduler_x))**2)
                    if scheduler_avg_slope <= 0.0:
                        #This way we keep decreasing until we reach our threshold, at which point we either end this execution(early stopping) or end the decrease of our automatic scheduling
                        if eta <= param_stop_threshold:
                        #if mini_batch_size <= param_stop_threshold:
                            #If we don't have early stopping it will just keep training on the lowest value we allow, at this point.
                            #Otherwise,
                            if self.early_stopping:
                                #Fill in the rest of the output dict with our last values so we don't have to run extra configs
                                for remainder_epoch in range(epochs-(epoch+1)):
                                    if self.output_training_cost:
                                        output_dict[self.run_index][remainder_epoch].append(training_cost)
                                    if self.output_training_accuracy:
                                        output_dict[self.run_index][remainder_epoch].append(training_accuracy)
                                    if self.output_validation_accuracy:
                                        output_dict[self.run_index][remainder_epoch].append(validation_accuracy)
                                    if self.output_test_accuracy:
                                        output_dict[self.run_index][remainder_epoch].append(test_accuracy)
                                print "\nEarly stopped with low threshold"
                                break
                        else:
                            eta /= param_decrease_rate
                            #mini_batch_size /= param_decrease_rate
                            print "\nReducing eta by factor of {0} to {1}".format(param_decrease_rate, eta)
                            #print "Reducing Mini-Batch Size by factor of {0} to {1}".format(param_decrease_rate, mini_batch_size)

                        #If we decrease the param, we reset the interval by clearing our scheduler's results
                        scheduler_results = []
                    else:
                        #remove the first element
                        scheduler_results.pop(0)
        #Using our +1s for pretty print progress
        #print "\nConfig %i/%i, Run %i/%i Completed." % (self.config_index+1, self.config_count, self.run_index+1, self.run_count)
        print " - Completed."#Add to the end of our completed run progress
        return output_dict

    def save(self, filename):
        f = gzip.open(filename, "wb")
        pickle.dump((self.layers), f, protocol=-1)
        f.close()

#### Define layer types
class ConvPoolLayer(object):
    """Used to create a combination of a convolutional and a max-pooling
    layer.  A more sophisticated implementation would separate the
    two, but for our purposes we'll always use them together, and it
    simplifies the code, so it makes sense to combine them.

    """

    def __init__(self, filter_shape, image_shape, subsample=(1,1), poolsize=(2, 2),
                 poolmode="max", activation_fn=sigmoid):
        """`filter_shape` is a tuple of length 4, whose entries are the number
        of filters, the number of input feature maps, the filter height, and the
        filter width.

        `image_shape` is a tuple of length 4, whose entries are the
        mini-batch size, the number of input feature maps, the image
        height, and the image width.

        `poolsize` is a tuple of length 2, whose entries are the y and
        x pooling sizes.

        Just set poolsize = (1, 1) for no pooling.

        """
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.subsample= subsample
        self.poolsize = poolsize
        self.poolmode = poolmode
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
        #pooled_out = downsample.pool_2d( input=conv_out, ds=self.poolsize, ignore_border=True)
        pooled_out = pool.pool_2d( 
            input=conv_out, ds=self.poolsize, ignore_border=True, mode=self.poolmode)
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

    def cost(self, net):
        #net.x.shape[0] is our len(x) aka number of samples
        return net.cost.get_cost(self.output_dropout, net.y, net.x.shape[0])

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
        #net.x.shape[0] is our len(x) aka number of samples
        return net.cost.get_cost(self.output_dropout, net.y, net.x.shape[0])

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
