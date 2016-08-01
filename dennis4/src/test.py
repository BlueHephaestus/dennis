import numpy as np
import theano
import theano.tensor as T

x = T.scalar()
def logistic(x):
    return 1.0/(1+np.exp(-x))

f = theano.function(inputs=[x], outputs=logistic(x))
print f(0)

#y = np.zeros(shape=(5,))#Since we know it to be a vector
#print [np.arange(y.shape[0]), y]
