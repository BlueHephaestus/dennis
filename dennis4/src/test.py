import numpy as np
import itertools
from sklearn import linear_model
from scipy.optimize import minimize

clf = linear_model.LinearRegression()


def cartesian_product(vectors):
    return [i for i in itertools.product(*vectors)]

#x = [[10, 0.3, 0.1], [20, 0.3, 0.1], [30, 0.3, 0.1]]
#x = [[10, 0.3, 0.1], [20, 0.3, 0.1], [30, 0.3, 0.1]]
#x = [[10], [20], [30]]
m = [10, 20, 30]
l = [0.3, 0.4, 0.5]
#n = [0.1, 0.1, 0.1]

cp_ys = [8.4, 8.22, 8.54, 4.4, 4.1, 4.6, 9.4, 9.2, 9.5]
#In this instance

hps = [m, l]
#print hps[0]
hp_cp = cartesian_product([m, l])

hp_ys = [np.copy(hp).astype(float) for hp in hps]#For our averaged output of each hp set from putting in all the cartesian product values, we just need the same shape as our original hps list

#this is the part where we'd get our outputs in the actual program from main config
for hp_index, hp in enumerate(hps):
    for hp_val_index, hp_val in enumerate(hp):
        hp_val_output_sum = 0
        n_hp_val = 0
        for config_index, config in enumerate(hp_cp):
            if hp_val == config[hp_index]:#
                hp_val_output_sum += cp_ys[config_index]
                n_hp_val += 1
        hp_ys[hp_index][hp_val_index] = hp_val_output_sum/float(n_hp_val)


#Get our multivariate regression for the 3 dimensional input
#clf.fit(x, y)
coefs = []
#print hps[0], hp_ys[0]
for hp_index, hp in enumerate(hps):
    coefs.append(np.polynomial.polynomial.polyfit(hp, hp_ys[hp_index], 2))


#print coefs
#x = [[1.0, 1.0, 1.0]]
#x = [[20, 0.3, 0.1]]
def hp_function(hps):
    #return sum([coef[0]*hp**2 + coef[1]*hp + coef[2] for coef, hp in zip(coefs, hps)])
    return sum([coef[0] + coef[1]*hp + coef[2]*hp**2 for coef, hp in zip(coefs, hps)])#Why the fuck are quadratic regression coefficient orders backwards
    #return sum([coef*hp_input for coef, hp_input in zip(clf.coef_, hp_inputs)])# + sum(abs(hp_input) for hp_input in hp_inputs)
    #return sum([1*hp_input for hp_input in hp_inputs]) + sum(abs(hp_input) for hp_input in hp_inputs)
'''
def hp_function(x):
    return clf.coef_[0]*x**2 + 4
    #return 2*x**2+x+4
'''
'''
def hp_function(hp_inputs):
    return sum([clf.coef_[0]*hp_input**2 for hp_input in hp_inputs])
'''


#Get the minimum of our regression function
#res = minimize(hp_function, x, method='nelder-mead', bounds=((-10, 10), (-10, 10), (-10, 10)), options={'xtol': 1e-8, 'disp': True})
#bounds = (((-10, 30), (-10, 30), (-10, 30)), ((0.3, 0.3), (0.3, 0.3), (0.3, 0.3)), ((0.1, 0.1), (0.1, 0.1), (0.1, 0.1)))
#bounds = ((10, 30), (0.3, 0.3), (0.1, 0.1),(10, 30), (0.3, 0.3), (0.1, 0.1),(10, 30), (0.3, 0.3), (0.1, 0.1))
#bounds = ((0, 30), (0, 30), (0, 30), (0, 30), (0, 30), (0, 30), (0, 30), (0, 30), (0, 30))
#bounds = ((0, 30), (0, 30), (0, 30), (0, 30), (0, 30), (0, 30), (0, 30), (0, 30), (0, 30))
#bounds = ((0, 30), (0, 30), (0, 30))
#x = [10, 0.3, 0.1]
#x = [1, 1]
#bounds = [(hp.min, hp.max) for hp in hps]
bounds = ((10, 30), (0.3, 0.5))
bounds = tuple(bounds)
#bounds = ((0, 30), (0, 0.4))

hps = [1, 1]
print coefs
print hps
res = minimize(hp_function, hps, bounds=bounds, method='TNC', tol=1e-10, options={'xtol': 1e-8, 'disp': True})
#res = minimize(hp_function, x, method='TNC', tol=1e-10, options={'xtol': 1e-8, 'disp': True})
#res = minimize(hp_function, x, bounds=bounds)

#problem is that our x values returned should be between 10 and 30, like our first value for each sub array of x
#However, it keeps returning extremely high values, i.e. 1.2345e^54


#print clf.coef_
#print hp_function([10, 0.3, 0.1])
def step_roundup(num, step):
    return np.ceil(num/float(step))*float(step)

def step_rounddown(num, step):
    return np.floor(num/float(step))*float(step)
            

print res.x
center_points = res.x
for hp_index, center_point in enumerate(center_points):
    #hp_step = hps[hp_index].step
    step = 10
    new_step = 10*.1
    new_min = step_roundup(center_point-(step*.5), new_step)
    new_max = step_rounddown(center_point+(step*.5), new_step)
    print new_min, new_max
    sys.exit()
    #hps[hp_index].step
    

#print hp_function(res.x)
#print np.polynomial.polynomial.polyfit(x, y, n+1)

