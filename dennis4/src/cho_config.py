#For Cho's use optimizing
import dennis4
from dennis4 import Network
from dennis4 import sigmoid, tanh, ReLU, ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer

import json
import sample_loader
from sample_loader import *

class Configurer(object):
    def __init__(self, run_count, epochs, output_types, output_training_cost, output_training_accuracy, output_validation_accuracy, output_test_accuracy):
        self.run_count = run_count
        self.epochs = epochs

        self.training_data, self.validation_data, self.test_data = get_data_subsets(p_training = 0.8, p_validation = 0.1, p_test = 0.1, archive_dir="../data/mfcc_expanded_samples.pkl.gz")
        self.training_data, self.validation_data, self.test_data = load_data_shared(training_data=self.training_data, validation_data=self.validation_data, test_data=self.test_data, normalize_x=True)

        #Our default values
        self.output_types = output_types#DON'T FORGET TO UPDATE THIS WITH THE OTHERS
        self.training_data_subsections=None#Won't be needing this for our tiny dataset!

        #Currently too fast for these to be of much use, we might be able to use them well when we get deeper and more convolutional
        self.early_stopping=False
        self.automatic_scheduling=False

        self.output_training_cost=output_training_cost
        self.output_training_accuracy=output_training_accuracy
        self.output_validation_accuracy=output_validation_accuracy
        self.output_test_accuracy=output_test_accuracy

        self.output_title="Cho Tests"
        self.output_filename="cho_tests"
        self.output_type_names = ["Training Cost", "Training % Accuracy", "Validation % Accuracy", "Test % Accuracy"]
        self.print_results = False
        self.print_perc_complete = False
        self.update_output = True
        self.graph_output = False
        #literal_print_output = False#Doesn't work yet
        #Will by default subplot the output types, will make config*outputs if that option is specified as well.
        subplot_seperate_configs = False

    def run_config(self, mini_batch_size, learning_rate, momentum_coefficient, regularization_rate, p_dropout, config_index, config_count):#Last two are for progress

        #We run one config each time from cho, adding to our output dict each time

        #Gotta make seperate Network instances for each run else the params don't get re-initialized

        #Black --> white
        config = [
                    [Network([ 
                        FullyConnectedLayer(n_in=47*47, n_out=100, p_dropout=p_dropout), 
                        FullyConnectedLayer(n_in=100, n_out=30, p_dropout=p_dropout), 
                        SoftmaxLayer(n_in=30, n_out=7, p_dropout=p_dropout)], mini_batch_size), mini_batch_size, 
                        learning_rate, momentum_coefficient, regularization_rate, 100, 10, ""] 
                    for r in range(self.run_count)
                 ]

        #First, we run our configuration
        '''
        f = open('{0}_output.txt'.format(output_filename), 'w').close()
        '''
        output_dict = {}
        #for config_index, config in enumerate(configs):
        for run_index in range(self.run_count): 
            output_dict[run_index] = {}
            net = config[run_index][0]
            net.output_config(
                output_filename=self.output_filename, 
                training_data_subsections=self.training_data_subsections, 
                early_stopping=self.early_stopping,
                automatic_scheduling=self.automatic_scheduling,
                output_training_cost=self.output_training_cost,
                output_training_accuracy=self.output_training_accuracy,
                output_validation_accuracy=self.output_validation_accuracy,
                output_test_accuracy=self.output_test_accuracy,
                print_results=self.print_results,
                print_perc_complete=self.print_perc_complete,
                config_index=config_index,
                config_count=config_count,
                run_index=run_index,
                run_count=self.run_count,
                output_types=self.output_types)

            #For clarity
            '''
            mini_batch_size = config[run_index][1]
            learning_rate = config[run_index][2]
            momentum_coefficient = config[run_index][3]
            regularization_rate = config[run_index][4]
            '''
            scheduler_check_interval = config[run_index][5]
            param_decrease_rate = config[run_index][6]
            output_dict = net.SGD(output_dict, self.training_data, self.epochs, mini_batch_size, learning_rate, self.validation_data, self.test_data, momentum_coefficient=momentum_coefficient, lmbda=regularization_rate, 
                    scheduler_check_interval=scheduler_check_interval, param_decrease_rate=param_decrease_rate)
        
        #After all runs have executed

        #If there were more than one runs for this configuration, we average them all together for a new one
        #We do this by looping through all our y values for each epoch, and doing our usual mean calculations
        if self.run_count > 1:
            output_dict[self.run_count+1] = {}#For our new average entry
            for j in range(self.epochs):
                output_dict[self.run_count+1][j] = []#For our new average entry
                for o in range(self.output_types):
                    avg = sum([output_dict[r][j][o] for r in range(self.run_count)]) / self.run_count
                    output_dict[self.run_count+1][j].append(avg)
            return output_dict[self.run_count+1]#Return our average end result
        else:
            return output_dict[0]#Return our run, since we only did one.


                
    '''
    #Write the output for this config's runs
    f = open('{0}_output.txt'.format(output_filename), 'a')
    #print type(float(output_dict[0][0][0]))
    f.write(json.dumps(output_dict))
    #add a newline to seperate our configs
    f.write("\n")
    #wrap up by closing our file behind us.
    f.close()
    '''

"""
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
                        plt.plot(x, y, c=str(config_index*1.0/config_count), lw=2.0, label=configs[config_index][7])
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

"""
