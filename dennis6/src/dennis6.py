"""
DENNIS MK 6
This one uses keras!

-Blake Edwards / Dark Element
"""

import os, sys, copy, time
import numpy as np

import keras 

import dataset_obj
from dataset_obj import *

import output_grapher
from output_grapher import *

import keras_test_callback
from keras_test_callback import TestCallback

def handle_models(global_config, model_configs):

    model_layers = copy.deepcopy(model_configs[0][0][0])

    input_dims = global_config["input_dims"]
    output_dims = global_config["output_dims"]
    run_count = global_config["run_count"]
    epochs = global_config["epochs"]
    archive_dir = global_config["archive_dir"]
    p_training = global_config["p_training"]
    p_validation = global_config["p_validation"]
    p_test = global_config["p_test"]
    lira_data = global_config["lira_data"]
    subsection_n = global_config["subsection_n"]

    #Clean up our title to get a filename, convert characters to lowercase and spaces to underscores
    output_filename = global_config['output_title'].lower().replace(" ", "_")

    #If we're updating output, clear out the old file if it exists.
    if global_config['update_output']: open('{0}_output.txt'.format(output_filename), 'w').close()

    #Initialize an empty list to store our time-to-execute for each model
    model_times = []

    #Loop through entire models
    for model_config_i, model_config in enumerate(model_configs):
        #Initialize a new output_dict to store the results
        #   of running this model over our runs
        output_dict = {}

        #Start a new timer for us to record duration of running model
        model_start = time.time()

        #Loop through number of times to test(runs)
        if global_config['update_output']:
            for run_i in range(run_count):

                #Get our hyper parameter dictionary
                hps = model_config[run_i][1]

                #Do this every run so as to avoid any accidental bias that might arise
                #Load our datasets with our lira_data = True and subsection_n = number of subsections we get from our original training images.
                dataset, normalization_data = load_dataset_obj(p_training, p_validation, p_test, archive_dir, output_dims, hps['data_normalization'], lira_data=global_config["lira_data"], subsection_n=global_config["subsection_n"])
                
                if len(input_dims) == 2:
                    #Reshape our datasets accordingly if we have multidimensional input
                    dataset.training.x = np.reshape(dataset.training.x, [-1, input_dims[0], input_dims[1], 1])
                    dataset.validation.x = np.reshape(dataset.validation.x, [-1, input_dims[0], input_dims[1], 1])
                    dataset.test.x = np.reshape(dataset.test.x, [-1, input_dims[0], input_dims[1], 1])
                elif len(input_dims) == 3:
                    #Reshape our datasets accordingly if we have multidimensional input
                    dataset.training.x = np.reshape(dataset.training.x, [-1, input_dims[0], input_dims[1], input_dims[2]])
                    dataset.validation.x = np.reshape(dataset.validation.x, [-1, input_dims[0], input_dims[1], input_dims[2]])
                    dataset.test.x = np.reshape(dataset.test.x, [-1, input_dims[0], input_dims[1], input_dims[2]])

                #Print approximate percentage completion
                perc_complete = perc_completion(model_config_i, run_i, len(model_configs), run_count)
                print "--------%02f%% PERCENT COMPLETE--------" % perc_complete

                #Get our Network object differently depending on if optional parameters are present
                if 'cost' in hps:
                    cost = hps['cost']
                if 'weight_init' in hps:
                    weight_init = hps['weight_init']
                if 'bias_init' in hps:
                    bias_init = hps['bias_init']


                #Set up Keras model
                model = model_config[run_i][0][0]#Sequential, etc
                model_layers = model_config[run_i][0][1:]#Rest of the model
                for layer in model_layers:
                    model.add(layer)

                #Compile our model with hyper parameters
                model.compile(loss=hps["cost"], optimizer=hps["optimizer"], metrics=["accuracy"])

                #Train our model with all our data and parameters
                test_callback = TestCallback(model, (dataset.test.x, dataset.test.y))

                #Combine our results into arrays for each step.
                #We do the following formats so it will play nice with our output grapher.
                if global_config["output_validation_accuracy"] and global_config["output_test_accuracy"]:
                    outputs = model.fit(dataset.training.x, dataset.training.y, validation_data=(dataset.validation.x, dataset.validation.y), callbacks=[test_callback],  nb_epoch=epochs, batch_size=hps["mb_n"])
                    results = np.vstack((outputs.history["loss"], outputs.history["acc"], outputs.history["val_acc"], test_callback.acc)).transpose()
                elif global_config["output_validation_accuracy"]:
                    outputs = model.fit(dataset.training.x, dataset.training.y, validation_data=(dataset.validation.x, dataset.validation.y), nb_epoch=epochs, batch_size=hps["mb_n"])
                    results = np.vstack((outputs.history["loss"], outputs.history["acc"], outputs.history["val_acc"])).transpose()
                elif global_config["output_test_accuracy"]:
                    outputs = model.fit(dataset.training.x, dataset.training.y, callbacks=[test_callback], nb_epoch=epochs, batch_size=hps["mb_n"])
                    results = np.vstack((outputs.history["loss"], outputs.history["acc"], test_callback.acc)).transpose()
                else:
                    outputs = model.fit(dataset.training.x, dataset.training.y, nb_epoch=epochs, batch_size=hps["mb_n"])
                    results = np.vstack((outputs.history["loss"], outputs.history["acc"])).transpose()

                #Assign the outputs at each epoch to a new epoch key in our dictionary for this run
                run_dict = {}
                for epoch in range(epochs):
                    run_dict[epoch] = results[epoch].tolist()
                output_dict[run_i] = run_dict
                
                """
                #If this isn't our last run, delete our model
                if run_i != run_count-1:
                    for layer in model.layers:
                        layer.build()
                """
                """
                print results.history
                print test_callback.loss
                print test_callback.acc
                sys.exit()
                """
                """
                #print results.history["loss"]
                #print results.history["acc"]

                #loss_and_metrics = model.evaluate(dataset.test.x, dataset.test.y, batch_size=32)

                #print loss_and_metrics
                """
                """
                #Initialize Network
                net = Network(model[0], hps['input_dims'], hps['output_dims'], dataset, cost, weight_init, bias_init)

                #Finally, optimize and store the outputs
                output_dict[run_i] = net.optimize(global_config, epochs, hps['mb_n'], hps['optimization_type'], hps['optimization_term1'], hps['optimization_term1_decay_rate'], hps['optimization_term2'], hps['optimization_term3'], hps['optimization_defaults'], hps['regularization_type'], hps['regularization_rate'], hps['keep_prob'])
                """

            #Record times once all runs have executed
            model_time = time.time() - model_start
            model_times.append(model_time)

            #Get our average extra run if there were >1 runs
            if run_count > 1:#
                output_dict = get_avg_run(output_dict, epochs, run_count, get_output_types_n(global_config))

            #Save our output to the filename
            save_output(output_filename, output_dict)

        #For prettiness, print completion
        if model_config_i == len(model_configs)-1: print "--------100%% PERCENT COMPLETE--------" 

    if global_config['save_net'] and global_config['update_output']:
        #Save all of our network data
        print "Saving Model..."
        net.save(output_filename, normalization_data[0], normalization_data[1], model_layers, input_dims, output_dims, cost, weight_init, bias_init)

    #Print time duration for each model
    for model_config_i, model_time in enumerate(model_times):
        print "Config %i averaged %f seconds" % (model_config_i, model_time / float(run_count))

    #Graph all our results
    if global_config['graph_output']:
        output_grapher = OutputGrapher(global_config, output_filename, model_configs, epochs, run_count)
        output_grapher.graph()
