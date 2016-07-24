import dennis4
import theano
import pickle
import gzip
import librosa
import numpy as np

from dennis4 import Network
from dennis4 import sigmoid, tanh, ReLU, ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer

nn = "dennis4_shallow"
archive_dir = "../saved_networks/%s.pkl.gz" % nn
metadata_dir = "../saved_networks/%s_metadata.txt" % nn

f = open(metadata_dir, 'r')
nn_metadata = [l.strip() for l in f.readlines()]

n_mfcc = 20
theano.config.floatX = 'float32'
mean = nn_metadata[0]
stddev = nn_metadata[1]
max_mfcc_len = nn_metadata[2]

f = gzip.open(archive_dir, 'rb')
layers = pickle.load(f)

y, sr = librosa.load("../data/expanded_audio/play1.wav")
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).flatten()

x_mfccs = np.resize(mfcc, (max_mfcc_len)).flatten()
x = x_mfccs.astype(np.float32)
normalized_x = (x-mean)/stddev
x = [np.array([np.array(normalized_x.astype(np.float32))])]
x = theano.shared(np.asarray(x, dtype=theano.config.floatX), borrow=True)

net = Network(layers, 1)
#net = StaticNetwork(layers)
#net = Network([ FullyConnectedLayer(n_in=47*47, n_out=100), FullyConnectedLayer(n_in=100, n_out=30), SoftmaxLayer(n_in=30, n_out=7)], 1)

predicted_output = net.predict(x)
print predicted_output[0], predicted_output[1]

