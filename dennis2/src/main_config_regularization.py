import dennis2
from dennis2 import Network
from dennis2 import sigmoid, tanh, ReLU, ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer

import json
#import sample_loader

#training_data, validation_data, test_data = dennis2.load_data_shared(filename="../data/mnist.pkl.gz")
#training_data, validation_data, test_data = dennis2.load_data_shared(filename="../data/samples.pkl.gz", normalize_x=True)
#training_data, validation_data, test_data = dennis2.load_data_shared(filename="../data/expanded_samples.pkl.gz", normalize_x=True)
training_data, validation_data, test_data = dennis2.load_data_shared(filename="../data/mfcc_samples.pkl.gz", normalize_x=True)


#So we have similarities to use for our graphing / these need to be the same for it to be more or less reasonable
#Basically these are our global things
output_types = 4#DON'T FORGET TO UPDATE THIS WITH THE OTHERS
run_count = 8
epochs = 1000
training_data_subsections=None#Won't be needing this for our tiny dataset!
early_stopping=False

output_training_cost=True
output_training_accuracy=True
output_validation_accuracy=True
output_test_accuracy=True

output_title="regularization comparisons"
output_filename="regularization_comparisons"
output_type_names = ["Training Cost", "Training % Accuracy", "Validation % Accuracy", "Test % Accuracy"]
print_results = False
update_output = True
graph_output = True
print_output = True
#Will by default subplot the output types, will make config*outputs if that option is specified as well.
subplot_seperate_configs = False

#With this setup we don't need redundant config info or data types :D
#the only redundant thing we need is our mini_batch_size, far as I can tell q_q
#Network object, mini batch size, learning rate, momentum coefficient, regularization rate

#Black --> white
configs = [
            [Network([
                FullyConnectedLayer(n_in=38*38, n_out=100),
                FullyConnectedLayer(n_in=100, n_out=30),
                FullyConnectedLayer(n_in=30, n_out=10),
                SoftmaxLayer(n_in=10, n_out=2)], 10), 10,
                .1, 0.0, 0.02, "lambda=0.02"
            ],
            [Network([
                FullyConnectedLayer(n_in=38*38, n_out=100),
                FullyConnectedLayer(n_in=100, n_out=30),
                FullyConnectedLayer(n_in=30, n_out=10),
                SoftmaxLayer(n_in=10, n_out=2)], 10), 10,
                .1, 0.0, 0.025, "lambda=0.025"
            ],
            [Network([
                FullyConnectedLayer(n_in=38*38, n_out=100),
                FullyConnectedLayer(n_in=100, n_out=30),
                FullyConnectedLayer(n_in=30, n_out=10),
                SoftmaxLayer(n_in=10, n_out=2)], 10), 10,
                .1, 0.0, 0.03, "lambda=0.03"
            ]
        ]

config_count = len(configs)
if update_output:
    #First, we run our configurations
    f = open('{0}_output.txt'.format(output_filename), 'w').close()
    output_dict = {}
    
    for config_index, config in enumerate(configs):
        for run_index in range(run_count): 
            output_dict[run_index] = {}
            net = config[0]
            net.output_config(
                output_filename=output_filename, 
                training_data_subsections=training_data_subsections, 
                early_stopping=early_stopping,
                output_training_cost=output_training_cost,
                output_training_accuracy=output_training_accuracy,
                output_validation_accuracy=output_validation_accuracy,
                output_test_accuracy=output_test_accuracy,
                print_results=print_results,
                config_index=config_index,
                config_count=config_count,
                run_index=run_index,
                run_count=run_count,
                output_types=output_types)

            #For clarity
            mini_batch_size = config[1]
            learning_rate = config[2]
            momentum_coefficient = config[3]
            regularization_rate = config[4]
            output_dict = net.SGD(output_dict, training_data, epochs, mini_batch_size, learning_rate, validation_data, test_data, momentum_coefficient=momentum_coefficient, lmbda=regularization_rate)
        
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

if graph_output:
    #Then we graph the results
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import OrderedDict

    f = open('{0}_output.txt'.format(output_filename), 'r')

    config_index = 0
    #plt.figure(1)#Unnecessary
    plt.suptitle(output_title)
    for config in f.readlines():
        config_results = json.loads(config, object_pairs_hook=OrderedDict)#So we preserve the order of our stored json
        if training_data_subsections:
            N = epochs*training_data_subsections
        else:
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
                    if training_data_subsections:
                        for s in config_results[r][j]:
                            #I wanna do the [a for b in c] but it's messier q_q
                            y.append(config_results[r][j][s][output_type])
                    else:
                        y.append(config_results[r][j][output_type])
                #The brighter the line, the later the config(argh i wish it was the other way w/e)
                if int(r) >= run_count:
                    #Our final, average run
                    if len(configs[config_index]) > 5:
                        plt.plot(x, y, c=str(config_index*1.0/config_count), lw=4.0, label=configs[config_index][5])
                    else:
                        plt.plot(x, y, c=str(config_index*1.0/config_count), lw=4.0)
                    #plt.plot(x, y, lw=4.0)
                else:
                    #plt.plot(x, y, c=np.random.randn(3,1), ls='--')
                    plt.plot(x, y, c=str(config_index*1.0/config_count), ls='--')
                    #insert plt.title here for when we add our config name metadata
        if len(configs[config_index]) > 5:
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        config_index+=1

    '''
    if print_output:
        plt.savefig(output_filename + ".png", bbox_inches='tight')
    '''
    plt.show()

