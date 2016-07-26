import librosa
import scipy.io.wavfile as wav
import numpy as np
import os
import cPickle, gzip #for storing our data 
import pickle
import theano
import theano.tensor as T
#Get our big array of data and seperate it into training, validation, and test data
#According to the percentages passed in

##Defaults:
#80% - Training
#10% - Validation
#10% - Test
def unison_shuffle(a, b):
  rng_state = np.random.get_state()
  np.random.shuffle(a)
  np.random.set_state(rng_state)
  np.random.shuffle(b)

def get_data_subsets(archive_dir="../data/mfcc_samples.pkl.gz", p_training=0.8, p_validation=0.1, p_test=0.1):
  print "Getting Training, Validation, and Test Data..."
  
  f = gzip.open(archive_dir, 'rb')
  data = cPickle.load(f)
  #Randomize our data
  '''
  np.random.shuffle(data)
  How to do this while maintaining the relative position of each?
  print data[0], data[1]
  data[0] = np.random.shuffle(data[0])
  data[1] = np.random.shuffle(data[1])
  print data
  '''
  n_samples = len(data[0])
  #Now we split our samples according to percentage

  #print n_samples

  training_data = [[], []]
  validation_data = [[], []]
  test_data = [[], []]

  n_training_subset = np.floor(p_training*n_samples)
  n_validation_subset = np.floor(p_validation*n_samples)
  n_test_subset = np.floor(p_test*n_samples)
  
  #Shuffle while retaining element correspondence
  print "Shuffling data..."
  unison_shuffle(data[0], data[1])

  #Get actual subsets
  data_x_subsets = np.split(data[0], [n_training_subset, n_training_subset+n_validation_subset])#basically the lines we cut to get our 3 subsections
  data_y_subsets = np.split(data[1], [n_training_subset, n_training_subset+n_validation_subset])

  training_data[0] = data_x_subsets[0]
  validation_data[0] = data_x_subsets[1]
  test_data[0] = data_x_subsets[2]

  training_data[1] = data_y_subsets[0]
  validation_data[1] = data_y_subsets[1]
  test_data[1] = data_y_subsets[2]
  
  return training_data, validation_data, test_data


def generate_input_normalizer(training_data):
    print "Generating Input Normalizer..."
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
    print "Normalizing Input..."
    data[0] = (data[0]-mean)/stddev
    return data

#### Load the data
def load_data_shared(training_data=None, validation_data=None, test_data=None, normalize_x=False, experimental_dir=None):
    print "Initializing Shared Variables..."
    if not training_data and not experimental_dir:
        #Configuration person fucked up
        print "You must supply either data subsets, or choose the experimenatal directory"
        return
    if experimental_dir:
        f = gzip.open(experimental_dir, 'rb')
        training_data, validation_data, test_data = cPickle.load(f)

    if normalize_x:
        #normalize input data.
        input_normalizer_mean, input_normalizer_stddev = generate_input_normalizer(training_data)

        training_data = normalize_input(training_data, input_normalizer_mean, input_normalizer_stddev)
        validation_data = normalize_input(validation_data, input_normalizer_mean, input_normalizer_stddev)
        test_data = normalize_input(test_data, input_normalizer_mean, input_normalizer_stddev)
    else:
        input_normalizer_mean = 0
        input_normalizer_stddev = 1

    def shared(data):
        """Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.

        """
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")
    print "Initializing Configuration..."
    return [shared(training_data), shared(validation_data), shared(test_data), [input_normalizer_mean, input_normalizer_stddev]]

#training_data, validation_data, test_data = get_data_subsets(archive_dir="../data/mfcc_expanded_samples.pkl.gz")
#training_data, validation_data, test_data = load_data_shared(training_data, validation_data, test_data, normalize_x=True)

