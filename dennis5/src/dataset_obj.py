"""
For easily getting an object with the methods and objects we need to access easily in training.

-Blake Edwards / Dark Element
"""

import sys
import numpy as np

import sample_loader
from sample_loader import *


def get_one_hot_m(v, width):
    #Given vector and width of each one hot, 
    #   get one hot matrix such that each index specified becomes a row in the matrix
    m = np.zeros(shape=(len(v), width))
    m[np.arange(len(v)), v] = 1
    return m


#Called by outside of this, uses the classes defined in this file to return dataset object
def load_dataset_obj(p_training, p_validation, p_test, archive_dir, output_dims):
    #Obtain datasets
    training_data, validation_data, test_data = get_data_subsets(p_training = p_training, p_validation = p_validation, p_test = p_test, archive_dir=archive_dir)

    #Convert ys in each to one hot vectors
    training_data[1] = get_one_hot_m(training_data[1], output_dims)
    validation_data[1] = get_one_hot_m(validation_data[1], output_dims)
    test_data[1] = get_one_hot_m(test_data[1], output_dims)

    #return dataset obj
    return Dataset(training_data, validation_data, test_data)
    

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
