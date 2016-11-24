"""
For normalizing our input data, 
    currently only whole input normalization, have
    yet to implement batch normalization

-Blake Edwards / Dark Element
"""
import numpy as np

def generate_input_normalizer(training_data):
    print "Generating Input Normalizer..."
    #we initialize our inputs with a gaussian distribution - this works by generating a gaussian distribution based on the mean and standard deviation of our training data since it should be a reasonable way to generalize for test data and so on. It helps to make it a gaussian distribution so that we can most of the time keep our neurons from saturating straight off, just as we do with weights and biases. Just needed to write this out to make sure I gots it
    #See written notes
    '''The following line is basically:
    for sample in training_data[0]:
        for input in sample:
            return input
    '''
    input_x = [input for sample in training_data[0] for input in sample]#for sample in x: for input in x: return input
    mean = np.mean(input_x)
    stddev = np.linalg.norm(input_x-mean)/np.sqrt(len(input_x))
    return mean, stddev

def normalize_input(data, mean, stddev):
    print "Normalizing Input..."
    data[0] = data[0]*stddev + mean
    return data
