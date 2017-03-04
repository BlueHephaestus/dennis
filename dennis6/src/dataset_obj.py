"""
For easily getting an object with the methods and objects we need to access easily in training.

-Blake Edwards / Dark Element
"""

import sys, os, gzip, cPickle, json
import h5py
import numpy as np

import data_normalization
from data_normalization import *


def get_one_hot_m(v, width):
    #Given vector and width of each one hot, 
    #   get one hot matrix such that each index specified becomes a row in the matrix
    m = np.zeros(shape=(len(v), width))
    m[np.arange(len(v)), v] = 1
    return m

def unison_shuffle(a, b):
    #Shuffle our two arrays while retaining the relation between them
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

def get_data_subsets_lira(archive_dir, p_training=0.8, p_validation=0.1, p_test=0.1):
    """
    Get our dataset array and seperate it into training, validation, and test data
        according to the percentages passed in.

    Defaults:
        80% - Training
        10% - Validation
        10% - Test

    For our LIRA implementation, we have some samples that are obtained by splitting a big image into subsection_n subsections,
        and we also have some samples that are individual samples. 
    Our metadata is of the format [subsection_n, sample_n, whole_image_n, individual_sub_n], where 
        subsection_n = # of subsections per whole image
        sample_n = total number of samples, whole_image_n + individual_sub_n
        whole_image_n = # of whole images used to obtain samples
        individual_sub_n = # of individual subsections

    So we're going to take all the whole image samples according to image, shuffle them by image (in sections of subsection_n),
        shuffle the individual ones however, and then put them all back together.
    This way, we avoid getting a subsection in the training data that is right next to one in the test data, 
        which would give us biased accuracies.
    """
    print "Loading Training, Validation, and Test Data..."

    #Load our hdf5 file to get data and metadata
    with h5py.File(archive_dir,'r') as hf:
        data = [np.array(hf.get("x")), np.array(hf.get("y"))]
        metadata = np.array(hf.get("metadata"))

    #Load metadata
    subsection_n, sample_n, whole_image_n, individual_sub_n, label_dict = metadata
    subsection_n = int(subsection_n)
    sample_n = int(sample_n)
    whole_img_n = int(whole_image_n)
    individual_sub_n = int(individual_sub_n)
    label_dict = json.loads(label_dict)

    #Split into our whole image samples and our individual rim samples
    whole_img_samples = [data[0][:whole_img_n], data[1][:whole_img_n]]
    individual_sub_samples = [data[0][whole_img_n:], data[1][whole_img_n:]]

    #Split these according to whole image, whole_image_n//subsection_n to get number of whole images
    if whole_img_n != 0:
        whole_img_samples[0] = np.split(whole_img_samples[0], whole_img_n//subsection_n)
        whole_img_samples[1] = np.split(whole_img_samples[1], whole_img_n//subsection_n)

    #Create these beforehand
    training_data = [[], []]
    validation_data = [[], []]
    test_data = [[], []]

    whole_img_training_data = [[], []]
    whole_img_validation_data = [[], []]
    whole_img_test_data = [[], []]

    individual_sub_training_data = [[], []]
    individual_sub_validation_data = [[], []]
    individual_sub_test_data = [[], []]

    #Get our respective subset sizes, with test being the excess
    whole_img_training_subset_n = int(np.floor(p_training*whole_img_n//subsection_n))
    whole_img_validation_subset_n = int(np.floor(p_validation*whole_img_n//subsection_n))
    whole_img_test_subset_n = whole_img_n - whole_img_training_subset_n - whole_img_validation_subset_n

    individual_sub_training_subset_n = int(np.floor(p_training*individual_sub_n))
    individual_sub_validation_subset_n = int(np.floor(p_validation*individual_sub_n))
    individual_sub_test_subset_n = individual_sub_n - individual_sub_training_subset_n - individual_sub_validation_subset_n

    #Shuffle while retaining element correspondence
    print "Shuffling Data..."
    unison_shuffle(whole_img_samples[0], whole_img_samples[1])
    unison_shuffle(individual_sub_samples[0], individual_sub_samples[1])

    #Get actual subsets
    print "Getting Data Subsets..."
    whole_img_x_subsets = np.split(whole_img_samples[0], [whole_img_training_subset_n, whole_img_training_subset_n+whole_img_validation_subset_n])#basically the lines we cut to get our 3 subsections
    whole_img_y_subsets = np.split(whole_img_samples[1], [whole_img_training_subset_n, whole_img_training_subset_n+whole_img_validation_subset_n])
    individual_sub_x_subsets = np.split(individual_sub_samples[0], [individual_sub_training_subset_n, individual_sub_training_subset_n+individual_sub_validation_subset_n])#basically the lines we cut to get our 3 subsections
    individual_sub_y_subsets = np.split(individual_sub_samples[1], [individual_sub_training_subset_n, individual_sub_training_subset_n+individual_sub_validation_subset_n])

    #Get our respective susbsets
    whole_img_training_data[0] = whole_img_x_subsets[0]
    whole_img_validation_data[0] = whole_img_x_subsets[1]
    whole_img_test_data[0] = whole_img_x_subsets[2]

    whole_img_training_data[1] = whole_img_y_subsets[0]
    whole_img_validation_data[1] = whole_img_y_subsets[1]
    whole_img_test_data[1] = whole_img_y_subsets[2]

    individual_sub_training_data[0] = individual_sub_x_subsets[0]
    individual_sub_validation_data[0] = individual_sub_x_subsets[1]
    individual_sub_test_data[0] = individual_sub_x_subsets[2]

    individual_sub_training_data[1] = individual_sub_y_subsets[0]
    individual_sub_validation_data[1] = individual_sub_y_subsets[1]
    individual_sub_test_data[1] = individual_sub_y_subsets[2]

    """
    For the whole image data,
    Now that we've shuffled and split relative to images(instead of subsections), collapse back to matrix (or vector if Y)
        We do -1 so that it infers we want to combine the first two dimensions, and we have the last argument because we
        want it to keep the same last dimension. repeat this for all of the subsets
    Since Y's are just vectors, we can easily just flatten
    """
    whole_img_training_data[0] = whole_img_training_data[0].reshape(-1, whole_img_training_data[0].shape[-1])
    whole_img_training_data[1] = whole_img_training_data[1].flatten()
    whole_img_validation_data[0] = whole_img_validation_data[0].reshape(-1, whole_img_validation_data[0].shape[-1])
    whole_img_validation_data[1] = whole_img_validation_data[1].flatten()
    whole_img_test_data[0] = whole_img_test_data[0].reshape(-1, whole_img_test_data[0].shape[-1])
    whole_img_test_data[1] = whole_img_test_data[1].flatten()

    #FINALLY, we combine image and rim into entire dataset
    print "Assigning Datasets..."
    training_data[0] = np.concatenate((whole_img_training_data[0], individual_sub_training_data[0]), axis=0)
    training_data[1] = np.concatenate((whole_img_training_data[1], individual_sub_training_data[1]), axis=0)
    validation_data[0] = np.concatenate((whole_img_validation_data[0], individual_sub_validation_data[0]), axis=0)
    validation_data[1] = np.concatenate((whole_img_validation_data[1], individual_sub_validation_data[1]), axis=0)
    test_data[0] = np.concatenate((whole_img_test_data[0], individual_sub_test_data[0]), axis=0)
    test_data[1] = np.concatenate((whole_img_test_data[1], individual_sub_test_data[1]), axis=0)

    #Print our number of samples for everything. Tabs added for cleanliness.
    print "Total Samples:\t\t\t%i" % (sample_n)
    print "\tTraining Samples:\t%i (%f%%)" % (training_data[0].shape[0], p_training*100)
    print "\tValidation Samples:\t%i (%f%%)" % (validation_data[0].shape[0], p_validation*100)
    print "\tTest Samples:\t\t%i (%f%%)" % (test_data[0].shape[0], p_test*100)

    return training_data, validation_data, test_data

def get_data_subsets(archive_dir="../data/mfcc_samples.pkl.gz", p_training=0.8, p_validation=0.1, p_test=0.1):
    """
    Get our dataset array and seperate it into training, validation, and test data
        according to the percentages passed in.

    Defaults:
        80% - Training
        10% - Validation
        10% - Test

    This one is our general function, which will work unless we have a strange dataset bias situation as in the LIRA example.
    """

    print "Getting Training, Validation, and Test Data..."
    f = gzip.open(archive_dir, 'rb')
    data = cPickle.load(f)

    n_samples = len(data[0])

    #Now we split our samples according to percentage
    training_data = [[], []]
    validation_data = [[], []]
    test_data = [[], []]


    n_training_subset = np.floor(p_training*n_samples)
    n_validation_subset = np.floor(p_validation*n_samples)
    #Assign this to it's respective percentage and whatever is left
    n_test_subset = n_samples - n_training_subset - n_validation_subset

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

"""
NECESSARY FORMATS
Each subset must be the following format:
    [x, y]
        x shape: (number of samples, length of input vector)
            where the values are samples, and inputs for each
        y shape: (number of samples, one hot output vector)
            where the values are samples, and a one hot vector of outputs for each

So in the case of MNIST, we'd have 
x: (20, 784)
y: (20, 10) 

if there were only 20 images.
"""

#Called by outside of this, uses the classes defined in this file to return dataset object
def load_dataset_obj(p_training, p_validation, p_test, archive_dir, output_dims, data_normalization=True, lira_data=False):
    #Obtain datasets
    if lira_data:
        #Use our subsection_n and sample_loader specific function
        training_data, validation_data, test_data = get_data_subsets_lira(archive_dir, p_training=p_training, p_validation=p_validation, p_test=p_test)
    else:
        training_data, validation_data, test_data = get_data_subsets(archive_dir, p_training=p_training, p_validation=p_validation, p_test=p_test)
        
    #Do whole data normalization on our input data, by getting the mean and stddev of the training data,
    #Then keeping these metrics and applying to the other data subsets
    if data_normalization:
        input_normalizer_mean, input_normalizer_stddev = generate_input_normalizer(training_data)
    else:
        input_normalizer_mean = 0.0
        input_normalizer_stddev = 1.0
    normalization_data = [input_normalizer_mean, input_normalizer_stddev]

    training_data = normalize_input(training_data, input_normalizer_mean, input_normalizer_stddev)
    validation_data = normalize_input(validation_data, input_normalizer_mean, input_normalizer_stddev)
    test_data = normalize_input(test_data, input_normalizer_mean, input_normalizer_stddev)

    #Convert ys in each to one hot vectors
    training_data[1] = get_one_hot_m(training_data[1], output_dims)
    validation_data[1] = get_one_hot_m(validation_data[1], output_dims)
    test_data[1] = get_one_hot_m(test_data[1], output_dims)

    #return dataset obj
    return Dataset(training_data, validation_data, test_data), normalization_data
    

class Dataset(object):

    #Initialize our data subset objects
    def __init__(self, training_data, validation_data, test_data):
        self.training = training_subset(training_data)
        self.validation = validation_subset(validation_data)
        self.test = test_subset(test_data)

    #Get a new mb_n number of entries from our training subset, after shuffling both sides in unison
    def next_batch(self, mb_n):
        #Shuffle our training dataset,
        #Return first mb_n elements of shuffled dataset
        unison_shuffle(self.training.x, self.training.y)
        return [self.training.x[:mb_n], self.training.y[:mb_n]]

#So we assure we have the same attributes for each subset
class DataSubset(object):
    def __init__(self, data):
        #self.whole_data = data
        self.x = data[0]
        self.y = data[1]

class training_subset(DataSubset):
    def __init__(self, data):
        DataSubset.__init__(self, data)

class validation_subset(DataSubset):
    def __init__(self, data):
        DataSubset.__init__(self, data)

class test_subset(DataSubset):
    def __init__(self, data):
        DataSubset.__init__(self, data)

#load_dataset_obj(0.7, 0.15, 0.15, os.path.expanduser("~/programming/machine_learning/tuberculosis_project/lira/lira1/data/samples.h5"), 6, data_normalization=True, lira_data=True)
