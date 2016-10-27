
"""
For configuring Dennis to run any sort of comparison or experimental configuration.

-Blake Edwards / Dark Element
"""
import sys, json
import numpy as np
import tensorflow as tf

import dennis5
from dennis5 import *

import dennis_base
from dennis_base import *

import output_grapher
from output_grapher import *

import layers
from layers import *

import costs
from costs import *

import weight_inits
from weight_inits import *

import bias_inits
from bias_inits import *

input_dims = 784
output_dims = 10
run_count = 10
epochs = 10#Have to have this here since it needs to be the same across configs

output_config = {
    'output_title': "NN Test 1",
    'graph_output': True,
    'update_output': True,
    'output_cost' : True,
    'subplot_seperate_configs': False,
    'print_times': False,
    'output_training_accuracy' : True,
    'output_validation_accuracy' : True,
    'output_test_accuracy' : True,

}

""" configs = [
            [
                [Network([ 
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
                    SoftmaxLayer(10, 10)], input_dims, output_dims, cost=cross_entropy]
                for r in range(run_count)
            ],
        ]
"""
configs = [
            [
                [FullyConnectedLayer(784, 10, activation_fn=tf.nn.sigmoid), 
                SoftmaxLayer(10, 10)], 
                {    
                    'input_dims': input_dims,
                    'output_dims': output_dims,
                    'cost': cross_entropy, 
                    'mb_n': 50,
                    'optimization_type': 'gd',
                    'optimization_term1': 0.5,
                    'optimization_term1_decay_rate': 0.9,
                    'optimization_term2': 0.0,
                    'optimization_term3': 0.0,
                    'optimization_defaults': True,
                    'regularization_rate': 0.0,
                    'keep_prob': 1.0,
                    'label': "Initial Test 1"
                }
            ],
          ]
#Default Network init values
cost = cross_entropy
weight_init = glorot_bengio
bias_init = standard


#Clean up our title to get a filename, convert characters to lowercase and spaces to underscores
output_filename = output_config['output_title'].lower().replace(" ", "_")

#If we're updating output, clear out the old file if it exists.
if output_config['update_output']: open('{0}_output.txt'.format(output_filename), 'w').close()

#Loop through entire configurations
for config_i, config in enumerate(configs):
    #Initialize a new output_dict to store the results
    #   of running this configuration over our runs
    output_dict = {}

    #Loop through number of times to test(runs)
    if output_config['update_output']:
        for run_i in range(run_count):

            #Print approximate percentage completion
            perc_complete = perc_completion(config_i, run_i, len(configs), run_count)
            print "--------%02f%% PERCENT COMPLETE--------" % perc_complete

            #Get our hyper parameter dictionary
            hps = config[1]#Hyper Parameters

            #Get our Network object differently depending on if optional parameters are present
            if 'cost' in hps:
                cost = hps['cost']
            if 'weight_init' in hps:
                weight_init = hps['weight_init']
            if 'bias_init' in hps:
                bias_init = hps['bias_init']

            #Initialize Network
            net = Network(config[0], hps['input_dims'], hps['output_dims'], cost, weight_init, bias_init)

            #Finally, optimize and store the outputs
            output_dict[run_i] = net.optimize(output_config, epochs, hps['mb_n'], hps['optimization_type'], hps['optimization_term1'], hps['optimization_term1_decay_rate'], hps['optimization_term2'], hps['optimization_term3'], hps['optimization_defaults'], hps['regularization_rate'], hps['keep_prob'])

        #Get our average extra run if there were >1 runs
        if run_count > 1:#
            output_dict = get_avg_run(output_dict, epochs, run_count, get_output_types_n(output_config))

        #Save our output to the filename
        save_output(output_filename, output_dict)

    #For prettiness, print completion
    if config_i == len(configs)-1: print "--------100%% PERCENT COMPLETE--------" 

if output_config['graph_output']:
    output_grapher = OutputGrapher(output_config, output_filename, configs, epochs, run_count)
    output_grapher.graph()
