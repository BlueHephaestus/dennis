import itertools#Cartesian product stuff
import numpy as np
from scipy.optimize import minimize

import main_config
from main_config import Configurer

def frange(start, end=None, inc=None):
    "A range function, that does accept float increments..."

    if end == None:
        end = start + 0.0
        start = 0.0

    if inc == None:
        inc = 1.0

    L = []
    while 1:
        next = start + len(L) * inc
        if inc > 0 and next >= end:
            break
        elif inc < 0 and next <= end:
            break
        L.append(next)
    return L

def step_roundup(num, step):
    return np.ceil(num/float(step))*float(step)

def step_rounddown(num, step):
    return np.floor(num/float(step))*float(step)

class HyperParameter(object):
    def __init__(self, min, max, step, step_decrease_factor, stop_threshold, label):
        self.min = min
        self.max = max
        self.step = step#How much we step to get our values through the range specified by min max
        self.step_decrease_factor = step_decrease_factor#Pretty self explanatory
        self.stop_threshold = stop_threshold#When we stop decreasing
        self.label = label

    def get_vector(self):
        if self.step != 0:
            #print self.min, self.max, self.step
            #print frange(self.min, self.max+self.step, self.step)
            val_vector = frange(self.min, self.max+self.step, self.step)
            #mini batch size brought this about, check if integer
            #Since we won't let mini batch become decimals
            for val_index, val in enumerate(val_vector):
                if val % 1 == 0.0: 
                    val_vector[val_index] = int(val)

            return val_vector
        else:
            #Since if our step = 0, we only have a constant value here.
            return [self.min]
        
def cartesian_product(vectors):
    return [i for i in itertools.product(*vectors)]

def hp_function(hps):
    #Our main function to minimize once we have our coefficients
    return sum([coef[0] + coef[1]*hp + coef[2]*hp**2 for coef, hp in zip(coefs, hps)])#Why the fuck are quadratic regression coefficient orders backwards

n_y = 5#Number of y values we look at the end of our output
output_type = 0#Type to judge our optimization on
#0 = Training Cost, 1 = Training Accuracy, etc.

#Initialize our configurer
configurer = Configurer(3, 100)

#TEMP
#FOR NOW I'M USING SHORTHAND SYMBOLS FOR VAR NAMES
#Set our initial HPs for cho to search through and optimize
m = HyperParameter(10, 30, 10, .1, 1, "Mini Batch Size")
n = HyperParameter(1.0, 3.0, .5, .1, 0.05, "Learning Rate")
u = HyperParameter(0.0, 0.0, 0, .1, 0.1, "Momentum Coefficient")
l = HyperParameter(0.0, 2.0, .5, .1, 0.1, "L2 Regularization Rate")
p = HyperParameter(0.0, 0.0, 0, .1, 0.1, "Dropout Regularization Percentage")

hps = [m, n, u, l, p]
n_hp = len(hps)

while True:

    #Get our vectors to make cartesian product out of
    hp_vectors = [hp.get_vector() for hp in hps]#When we need the actual values
    print hp_vectors

    #Get cartesian product
    hp_cp = cartesian_product(hp_vectors)

    hp_config_count = len(hp_cp)
    hp_cp_results = []#For the results in the cartesian product format, before averaging

    hp_ys = [np.copy(hp_vector).astype(float) for hp_vector in hp_vectors]
    coefs = []#For the coefficients of our quadratic regression of each hyper parameter

    #For our minimization
    #Uses the local min and max of each hp range
    bounds = [(hp.min, hp.max) for hp in hps]
    bounds = tuple(bounds)

    #Since we can just have 1s for each of our hps to plug in here.
    placeholder_hps = [1 for hp in hps]

    #Get our raw cp results/ys
    for hp_config_index, hp_config in enumerate(hp_cp):

        #Convert our np.float64 types to float
        hp_config = list(hp_config)
        hp_config[1:] = [float(hp) for hp in hp_config[1:]]

        #Execute configuration, get the average entry in the output_dict as a list of it's items
        config_avg_result = list(configurer.run_config(hp_config[0], hp_config[1], hp_config[2], hp_config[3], hp_config[4], hp_config_index, hp_config_count).items())

        #Get our average last n_y values from the respective output_type values
        config_y_vals = [config_y[1][output_type] for config_y in config_avg_result[-n_y:]]#We have our config_y[1] so we get the value, not the key
        config_avg_y_val = sum(config_y_vals)/float(n_y)

        #Add our result to each of our configs in hp_results
        hp_cp_results.append(config_avg_y_val)

    #hp_ys is used to get the average output using our hp caused, so if we had 3 mini batches and 3 regularization rates,
    #our associated hp_y value for our first mini batch size will be the average over the 3 runs that used the first mini batch size.
    #This is where we get those averages.
    for hp_index, hp in enumerate(hp_vectors):
        for hp_val_index, hp_val in enumerate(hp):
            hp_val_output_sum = 0
            n_hp_val = 0
            for config_index, config in enumerate(hp_cp):
                if hp_val == config[hp_index]:#
                    hp_val_output_sum += hp_cp_results[config_index]
                    n_hp_val += 1
            hp_ys[hp_index][hp_val_index] = hp_val_output_sum/float(n_hp_val)

    #Get our coefficients by doing a quadratic regression on each of our average output for each hyper parameter set
    for hp_index, hp in enumerate(hp_vectors):
        if len(hp) > 1:
            coefs.append(np.polynomial.polynomial.polyfit(hp, hp_ys[hp_index], 2))
        else:
            coefs.append([hp[0], 0, 0])

    print coefs

    res = minimize(hp_function, placeholder_hps, bounds=bounds, method='TNC', tol=1e-10, options={'xtol': 1e-8, 'disp': True})

    #Now our res.x are our new center point values
    center_points = res.x

    print center_points

    for hp_index, center_point in enumerate(center_points):
        if len(hp_vectors[hp_index]) > 1:
            step = hps[hp_index].step
            step_decrease_factor = hps[hp_index].step_decrease_factor
            stop_threshold = hps[hp_index].stop_threshold
            new_step = step*step_decrease_factor

            print new_step, stop_threshold
            if new_step < stop_threshold:
                #Time to mark this value as final and stop modifying.
                #This means we no longer update it, we just replace the min and max with our center point,
                #and make the step 0. Just as we do with dependent variables at the start
                new_min = center_point
                new_max = center_point
                new_step = 0
            else:
                #We get our inclusive range, i.e if center point is 19.14, 
                #We'd get 14.14, and 24.14. Then we round up and round down respectively,
                #to get 15 and 24
                new_min = step_roundup(center_point-(step*.5), new_step)
                new_max = step_rounddown(center_point+(step*.5), new_step)

            #We update with our new params if an independent hyper parameter
            hps[hp_index].min = new_min
            hps[hp_index].max = new_max
            hps[hp_index].step = new_step
            print new_min, new_max
    
    #Check if we have set them all as constant
    for hp in hp_vectors:
        if len(hp) > 1:
            break
    else:
        print "Optimization Finished, Hyper Parameters are: {0}".format(hp_vectors)
        break





'''
#Now we use our hp_results to optimize and prepare the next configs
#First, we do a <insert correct vocabulary> regression on our results,
    #Where the first n_hp values are our inputs to the function, and our last is our output

print np.polynomial.polynomial.polyfit(hp_cp, hp_results)

coefs = []
for hp in hp_vectors:
    coefs.append(np.polynomial.polynomial.polyfit(hp, y, 2))
'''
