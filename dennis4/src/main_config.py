#FOR TESTING CHO'S RESULTS
import dennis4
from dennis4 import Network
from dennis4 import sigmoid, tanh, ReLU, ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer

import json
import sample_loader
from sample_loader import *

training_data, validation_data, test_data = get_data_subsets(p_training = 0.8, p_validation = 0.1, p_test = 0.1, archive_dir="../data/mfcc_samples.pkl.gz")
training_data, validation_data, test_data, normalize_data = load_data_shared(training_data=training_data, validation_data=validation_data, test_data=test_data, normalize_x=True)

#So we have similarities to use for our graphing / these need to be the same for it to be more or less reasonable
#Basically these are our global things
output_types = 4#DON'T FORGET TO UPDATE THIS WITH THE OTHERS
run_count = 3
epochs = 100
training_data_subsections=None#Won't be needing this for our tiny dataset!

#Currently too fast for these to be of much use, we might be able to use them well when we get deeper and more convolutional
early_stopping=False
automatic_scheduling=False

output_training_cost=True
output_training_accuracy=True
output_validation_accuracy=True
output_test_accuracy=True

output_title="For Field Testing"
output_filename="dennis4_shallow_unexpanded"
output_type_names = ["Training Cost", "Training % Accuracy", "Validation % Accuracy", "Test % Accuracy"]
print_results = False
print_perc_complete = False
update_output = True
graph_output = True
save_net = True
literal_print_output = False
#Will by default subplot the output types, will make config*outputs if that option is specified as well.
subplot_seperate_configs = False

#With this setup we don't need redundant config info or data types :D
#Network object, mini batch size, learning rate, momentum coefficient, regularization rate

#Gotta make seperate Network instances for each run else the params don't get re-initialized

#Black --> white
'''
                [Network([ 
                    ConvPoolLayer(image_shape=(100, 1, 47, 47),
                        filter_shape=(20, 1, 6, 6),
                        poolsize=(2,2)),
                    ConvPoolLayer(image_shape=(100, 20, 21, 21),
                        filter_shape=(40, 20, 4, 4),
                        poolsize=(2,2)),
                    FullyConnectedLayer(n_in=3240, n_out=800), 
                    FullyConnectedLayer(n_in=800, n_out=200), 
                    FullyConnectedLayer(n_in=200, n_out=50), 
                    SoftmaxLayer(n_in=50, n_out=7)], 100), 100, 
                    1.0, 0.0, 0.0, 100, 10, ""] 
                for r in range(run_count)


                [Network([ 
                    ConvPoolLayer(image_shape=(100, 1, 47, 47),
                        filter_shape=(20, 1, 8, 8),
                        poolsize=(2,2)),
                    FullyConnectedLayer(n_in=20**3, n_out=2000), 
                    FullyConnectedLayer(n_in=2000, n_out=100), 
                    FullyConnectedLayer(n_in=100, n_out=30), 
                    SoftmaxLayer(n_in=30, n_out=7)], 100), 100, 
                    1.0, 0.0, 0.0, 100, 10, ""] 
                for r in range(run_count)
'''
configs = [
            [
                [Network([ 
                    FullyConnectedLayer(n_in=47*47, n_out=100), 
                    FullyConnectedLayer(n_in=100, n_out=30), 
                    SoftmaxLayer(n_in=30, n_out=7)], 15), 15, 
                    0.1778, 0.21, 1.87, 100, 10, ""] 
                for r in range(run_count)
            ],
        ]

config_count = len(configs)
if update_output:
    #First, we run our configurations
    f = open('{0}_output.txt'.format(output_filename), 'w').close()
    output_dict = {}
    
    for config_index, config in enumerate(configs):
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
            momentum_coefficient = config[run_index][3]
            regularization_rate = config[run_index][4]
            scheduler_check_interval = config[run_index][5]
            param_decrease_rate = config[run_index][6]
            output_dict = net.SGD(output_dict, training_data, epochs, mini_batch_size, learning_rate, validation_data, test_data, momentum_coefficient=momentum_coefficient, lmbda=regularization_rate, 
                    scheduler_check_interval=scheduler_check_interval, param_decrease_rate=param_decrease_rate)
        
        #After all runs have executed

        #If there were more than one runs for this configuration, we average them all together for a new one
        #We do this by looping through all our y values for each epoch, and doing our usual mean calculations
        if run_count > 1:
            output_dict[run_count+1] = {}#For our new average entry
            for j in range(epochs):
                output_dict[run_count+1][j] = []#For our new average entry
                for o in range(output_types):
                    avg = sum([output_dict[r][j][o] for r in range(run_count)]) / run_count
                    output_dict[run_count+1][j].append(avg)

        '''
        for r in range(run_count):
            for j in range(epochs):
                for o in range(output_types):
                    try:
                        a = json.dumps(output_dict[r][j][o])
                    except:
                        print o, type(o)
        '''
                    
        #Write the output for this config's runs
        f = open('{0}_output.txt'.format(output_filename), 'a')
        #print type(float(output_dict[0][0][0]))
        f.write(json.dumps(output_dict))
        #add a newline to seperate our configs
        f.write("\n")
        #wrap up by closing our file behind us.
        f.close()

        #Save layers
        if save_net:
            print "Saving Neural Network Layers..."
            net.save('../saved_networks/%s.pkl.gz' % output_filename)
            f = open('../saved_networks/%s_metadata.txt' % (output_filename), 'w')
            f.write("{0}\n{1}\n{2}".format(normalize_data[0], normalize_data[1], 47*47))
            f.close()


if graph_output:
    #Then we graph the results
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('GTK')#So we print on the host computer when ssh'ing
    from collections import OrderedDict

    f = open('{0}_output.txt'.format(output_filename), 'r')

    config_index = 0
    #plt.figure(1)#Unnecessary
    plt.suptitle(output_title)
    for config in f.readlines():
        config_results = json.loads(config, object_pairs_hook=OrderedDict)#So we preserve the order of our stored json
        N = epochs
        x = np.linspace(0, N, N)
        for output_type in range(output_types):
            if subplot_seperate_configs:
                plt.subplot(config_count, output_types, config_index*output_types+output_type+1)#number of rows, number of cols, number of subplot
            else:
                plt.subplot(1, output_types, output_type+1)#number of rows, number of cols, number of subplot
                plt.title(output_type_names[output_type])
                
            for r in config_results:
                y = []
                for j in config_results[r]:
                    y.append(config_results[r][j][output_type])
                #The brighter the line, the later the config(argh i wish it was the other way w/e)
                if int(r) >= run_count:
                    #Our final, average run
                    if len(configs[config_index]) > 5:
                        plt.plot(x, y, c=str(config_index*1.0/config_count), lw=2.0, label=configs[config_index][5])
                    else:
                        plt.plot(x, y, c=str(config_index*1.0/config_count), lw=2.0)
                    #plt.plot(x, y, lw=4.0)
                else:
                    #plt.plot(x, y, c=np.random.randn(3,1), ls='--')
                    plt.plot(x, y, c=str(config_index*1.0/config_count), ls='--')
        if len(configs[config_index]) > 7:
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        config_index+=1

    '''
    if literal_print_output:
        plt.savefig(output_filename + ".png", bbox_inches='tight')
    '''
    plt.show()

