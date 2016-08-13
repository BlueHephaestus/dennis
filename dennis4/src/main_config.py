
#std libs
import json
import time

#external libs
import dennis4
from dennis4 import *

import config_base

import sample_loader
from sample_loader import *

import output_grapher
from output_grapher import *

import sms_notifications

#General
run_count =                 3#Number of times to run the same config over again to get an accurate representation of it's accuracy and results
epochs =                    3000#Number of times we do a full training loop
training_data_subsections=  None#Enable this to get accuracy quicker. Good for initially figuring stuff out and saving time in experimentation

#Percentage of entire data for each subset
p_training_data =           0.8
p_validation_data =         0.1
p_test_data =               0.1

archive_dir = "../data/mfcc_expanded_samples.pkl.gz"#Dir to get all of our data from
whole_normalization =       True#To normalize on the entire training data. Will add batch normalization soon

#Scheduling
early_stopping=             False#Enable to stop training after our schedule parameter reaches a stop threshold, and fill in the remaining space with last obtained value
automatic_scheduling=       True#Enable to automatically judge the average slope of our validation % over the past scheduler_check_interval epochs and decrease by param_decrease_rate if it is < 0.0
forced_scheduler_interval = 150#If this isn't none it will ignore automatic scheduling and automatically decrease every n epochs
scheduler_check_interval =  50#Period to look back for judging if it's time to automatically schedule
param_decrease_rate =       2#Amount to decrease by, i.e. (eta /= param_decrease_rate) if our average slope in our past scheduler_check_interval is < 0.0

#Outputs
output_training_cost=       True
output_training_accuracy=   False
output_validation_accuracy= True
output_test_accuracy=       False

#Ensemble
training_ensemble =         False

#Output
output_title = "Miscellaneous Comparisons"#Title to show on the graph if graphing output
output_filename = "misc_comparisons"#The filename to save our output, and layers if we choose to save net
print_results =             False#Print our accuracies
print_perc_complete =       False#Give a live percentage until completions
update_output =             True#Actually update the output with our new configs. Don't disable this unless you just want to graph again or use pre-existing output
graph_output =              True#Graphs the results with output_grapher.py
save_net =                  False#Save our network layers as <output_filename>.pkl.gz
literal_print_output =      False#Literally print this to a networked printer. Not working (yet)
print_times =               False#Print the elapsed time each config took to train, on average if run_count > 1
sms_alerts =                False#Send finished alert to assigned number with sms_notifications.py
subplot_seperate_configs =  False#Seperate the config graphs into different subplots, rather than combining into one for ease of comparison.

#With our values set, we get the data:
training_data, validation_data, test_data = get_data_subsets(p_training = p_training_data, p_validation = p_validation_data, p_test = p_test_data, archive_dir=archive_dir)
training_data, validation_data, test_data, normalize_data = load_data_shared(training_data=training_data, validation_data=validation_data, test_data=test_data, normalize_x=whole_normalization)
input_dims = 51*51
output_dims = 6

#Gotta make seperate Network instances for each run else the params don't get re-initialized

'''
Miscellaneous configs that I don't want to have to write again
            [
                [Network([ 
                    ConvPoolLayer(image_shape=(14, 1, 51, 51),
                        filter_shape=(20, 1, 8, 8),
                        poolsize=(2,2)),
                    FullyConnectedLayer(n_in=22*22*20, n_out=2000), 
                    FullyConnectedLayer(n_in=2000, n_out=100), 
                    FullyConnectedLayer(n_in=100, n_out=30), 
                    SoftmaxLayer(n_in=30, n_out=7)], 14), 14, 
                    1.0, 0.0, 0.0, 100, 10, ""] 
                for r in range(run_count)
            ],
            [
                [Network([ 
                    FullyConnectedLayer(n_in=51*51, n_out=300, activation_fn=sigmoid), 
                    FullyConnectedLayer(n_in=300, n_out=80, activation_fn=sigmoid), 
                    FullyConnectedLayer(n_in=80, n_out=20, activation_fn=sigmoid), 
                    SoftmaxLayer(n_in=20, n_out=7)], 10, cost=log_likelihood), 10, 
                    0.138, 'vanilla', 0.0, 0.0, 0.0, scheduler_check_interval, param_decrease_rate, "control"] 
                for r in range(run_count)
           ],
            [
                [Network([ 
                    FullyConnectedLayer(n_in=input_dims, n_out=300, activation_fn=sigmoid, p_dropout=0), 
                    FullyConnectedLayer(n_in=300, n_out=80, activation_fn=sigmoid, p_dropout=0), 
                    FullyConnectedLayer(n_in=80, n_out=20, activation_fn=sigmoid, p_dropout=0), 
                    FullyConnectedLayer(n_in=20, n_out=output_dims, activation_fn=linear, p_dropout=0)], 9, cost=quadratic), 9, 
                    .29, 'momentum', 0.257, 0.0, 0, 50, 2, "Quadratic + Linear + Sigmoid"] 
                for r in range(run_count)
           ],

           [
                [Network([ 
                    ConvPoolLayer(image_shape=(56, 1, 51, 51),
                        filter_shape=(64, 1, 4, 4),
                        poolsize=(2,2),
                        poolmode='average_exc_pad'),
                    ConvPoolLayer(image_shape=(56, 64, 24, 24),
                        filter_shape=(64, 64, 5, 5),
                        poolsize=(2,2),
                        poolmode='average_exc_pad'),
                    FullyConnectedLayer(n_in=10*10*64, n_out=300, activation_fn=sigmoid, p_dropout=0.0), 
                    FullyConnectedLayer(n_in=300, n_out=30, activation_fn=sigmoid, p_dropout=0.2), 
                    SoftmaxLayer(n_in=30, n_out=output_dims, p_dropout=0.0)], 56, cost=log_likelihood), 56, 
                    .17, 'momentum', 0.0, 0.0, 3.0, scheduler_check_interval, param_decrease_rate, "l=3.0 d=0.2 new 2conv layout"] 
                for r in range(run_count)
           ],
            [
                [Network([ 
                    FullyConnectedLayer(n_in=51*51, n_out=300, activation_fn=sigmoid), 
                    FullyConnectedLayer(n_in=300, n_out=80, activation_fn=sigmoid), 
                    FullyConnectedLayer(n_in=80, n_out=20, activation_fn=sigmoid), 
                    SoftmaxLayer(n_in=20, n_out=7)], 10, cost=log_likelihood), 10, 
                    0.001, 'rmsprop', 0.9, 0.0, 0.0, 50, 2, "LL + Softmax + Sigmoid"] 
                for r in range(run_count)
           ],
            [
                [Network([ 
                    FullyConnectedLayer(n_in=51*51, n_out=300, activation_fn=sigmoid), 
                    FullyConnectedLayer(n_in=300, n_out=80, activation_fn=sigmoid), 
                    FullyConnectedLayer(n_in=80, n_out=20, activation_fn=sigmoid), 
                    SoftmaxLayer(n_in=20, n_out=7)], 10, cost=log_likelihood), 10, 
                    .17, 'momentum', 0.0, 0.0, 0.0, 50, 2, "LL + Softmax + Sigmoid"] 
                for r in range(run_count)
           ],
'''
configs = [
           [
                [Network([ 
                    ConvPoolLayer(image_shape=(20, 1, 51, 51),
                        filter_shape=(64, 1, 4, 4),
                        poolsize=(2,2),
                        poolmode='average_exc_pad'),
                    ConvPoolLayer(image_shape=(20, 64, 24, 24),
                        filter_shape=(128, 64, 5, 5),
                        poolsize=(2,2),
                        poolmode='average_exc_pad'),
                    FullyConnectedLayer(n_in=10*10*128, n_out=600, activation_fn=sigmoid, p_dropout=0.5), 
                    FullyConnectedLayer(n_in=600, n_out=30, activation_fn=sigmoid, p_dropout=0.5), 
                    SoftmaxLayer(n_in=30, n_out=output_dims, p_dropout=0.5)], 20, cost=log_likelihood), 20, 
                    .07977, 'momentum', 0.0, 0.0, 5.0, forced_scheduler_interval, scheduler_check_interval, param_decrease_rate, "l=5.0, d=0.5"]
                for r in range(run_count)
           ],
        ]

config_count = len(configs)
output_types = config_base.get_output_types(output_training_cost, output_training_accuracy, output_validation_accuracy, output_test_accuracy)

if update_output:
    #First, we run our configurations
    f = open('{0}_output.txt'.format(output_filename), 'w').close()
    if training_ensemble: ensemble_nets = []
    output_dict = {}
    config_times = []
    
    for config_index, config in enumerate(configs):
        config_start = time.time()
        for run_index in range(run_count): 
            output_dict[run_index] = {}
            net = config[run_index][0]
            net.output_config(
                output_filename=output_filename, 
                training_data_subsections=training_data_subsections, 
                early_stopping=early_stopping,
                automatic_scheduling=automatic_scheduling,
                output_training_cost=output_training_cost,
                output_training_accuracy=output_training_accuracy,
                output_validation_accuracy=output_validation_accuracy,
                output_test_accuracy=output_test_accuracy,
                print_results=print_results,
                print_perc_complete=print_perc_complete,
                config_index=config_index,
                config_count=config_count,
                run_index=run_index,
                run_count=run_count,
                output_types=output_types)

            #For clarity
            mini_batch_size = config[run_index][1]
            learning_rate = config[run_index][2]
            optimization = config[run_index][3]
            optimization_term1 = config[run_index][4]
            optimization_term2 = config[run_index][5]
            regularization_rate = config[run_index][6]
            forced_scheduler_interval = config[run_index][7]
            scheduler_check_interval = config[run_index][8]
            param_decrease_rate = config[run_index][9]
            output_dict = net.SGD(output_dict, training_data, epochs, mini_batch_size, learning_rate, validation_data, test_data, optimization=optimization, optimization_term1=optimization_term1, optimization_term2=optimization_term2, lmbda=regularization_rate, 
                    forced_scheduler_interval=forced_scheduler_interval, scheduler_check_interval=scheduler_check_interval, param_decrease_rate=param_decrease_rate)

            if training_ensemble:
                ensemble_nets.append(Network(net.layers, config_base.size(test_data)))


        
        #After all runs have executed
        config_time = time.time() - config_start
        config_times.append(config_time)

        #If there were more than one runs for this configuration, we average them all together for a new one
        #We do this by looping through all our y values for each epoch, and doing our usual mean calculations
        if run_count > 1:
            output_dict = config_base.get_avg_run(output_dict, epochs, run_count, output_types)

        #Write the output for this config's runs
        f = open('{0}_output.txt'.format(output_filename), 'a')
        f.write(json.dumps(output_dict))
        #add a newline to seperate our configs
        f.write("\n")
        #wrap up by closing our file behind us.
        f.close()

    #Save layers
    if save_net:
        config_base.save_net(net, output_filename, normalize_data, input_dims)

    for config_index, config_time in enumerate(config_times):
        print "Config %i averaged %f seconds" % (config_index, config_time / float(run_count))

    if sms_alerts:
        sms_notifications.send_sms("\nMain Configuration Finished.")

    if training_ensemble:
        print "Training Accuracy: %f" % config_base.get_ensemble_accuracy(training_accuracy)
        print "Validation Accuracy: %f" % config_base.get_ensemble_accuracy(validation_data)
        print "Test Accuracy: %f" % config_base.get_ensemble_accuracy(test_data)
            

if graph_output:
    output_grapher = OutputGrapher(output_title, output_filename, configs, epochs, run_count, output_types, output_training_cost, output_training_accuracy, output_validation_accuracy, output_test_accuracy, subplot_seperate_configs, print_times, update_output)
    output_grapher.graph()
