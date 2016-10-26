#Extraneous outlier classes 
import json

#Given two categories and index progress through them, return a percentage completion float.
def perc_completion(i, j, i_n, j_n):
    i_n = float(i_n)
    j_n = float(j_n)
    if i == 0 and i_n == 1.0:
        #Edge case in which it will only return 0, so we ignore the first
        return (j/j_n)* 100
    else:
        return (i/i_n + j/(i_n * j_n)) * 100

#Get our average next run given our preexisting runs
def get_avg_run(output_dict, epochs, run_count, output_types):
    output_dict[run_count+1] = {}#For our new average entry
    for j in range(epochs):
        output_dict[run_count+1][j] = []#For our new average entry
        for o in range(output_types):
            avg = sum([output_dict[r][j][o] for r in range(run_count)]) / run_count
            output_dict[run_count+1][j].append(avg)
    return output_dict

def get_output_types_n(output_config):
    n = 0
    if output_config['output_cost']:
        n+=1
    if output_config['output_training_accuracy']:
        n+=1
    if output_config['output_validation_accuracy']:
        n+=1
    if output_config['output_test_accuracy']:
        n+=1
    return n

def save_output(filename, output_dict):
    f = open('{0}_output.txt'.format(filename), 'a')
    f.write(json.dumps(output_dict))
    #add a newline to seperate our configs
    f.write("\n")
    #wrap up by closing our file behind us.
    f.close()
