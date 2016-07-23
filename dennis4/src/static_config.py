import dennis4
import pickle, cPickle
import gzip
import librosa
import numpy as np

from dennis4 import StaticNetwork
from dennis4 import sigmoid, tanh, ReLU, ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer

archive_dir = "dennis4.pkl.gz"
n_mfcc=20
max_mfcc_len = 2209
f = gzip.open(archive_dir, 'rb')
layers = cPickle.load(f)

y, sr = librosa.load("test.wav")
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).flatten()
x = np.resize(mfcc, (max_mfcc_len))

net = StaticNetwork(layers)
predicted_output = net.predict(x)
#print predicted_output

