import theano
import dennis4
import scipy.io.wavfile as wav#For temp messing with frames
import alsaaudio, librosa, wave
import pickle, gzip
import numpy as np
import sys

from dennis4 import Network
from dennis4 import sigmoid, tanh, ReLU, ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer

nn = "dennis4_shallow_unexpanded"
archive_dir = "../saved_networks/%s.pkl.gz" % nn
metadata_dir = "../saved_networks/%s_metadata.txt" % nn

f = open(metadata_dir, 'r')
nn_metadata = [float(l.strip()) for l in f.readlines()]
#vocab = ["Misc", "Shuffle", "Dennis", "Next", "Play", "Pause", "Back"]
vocab = ["Misc", "Play", "Shuffle", "Back", "Pause", "Next", "Dennis"]

#Load metadata
n_mfcc = 20
theano.config.floatX = 'float32'
mean = nn_metadata[0]
stddev = nn_metadata[1]
max_mfcc_len = int(nn_metadata[2])

#Prepare audio input stuff
period_size = 1024
inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE)
inp.setchannels(1)
inp.setrate(44100)
inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)
inp.setperiodsize(period_size)
audio_backlog = []
#i'm calling this at 65536 frames, being 64 frames of 1024 period size each(64*1024=65536)
#Should be enough to catch when dennis is said
#Not sure how much processing this will take up
#audio_backlog_max = 65536
#audio_backlog_max = 56320
audio_backlog_max = 32768
n_mfcc = 20
max_mfcc_len = 2209
frame_num = 1

#Load layers
f = gzip.open(archive_dir, 'rb')
layers = pickle.load(f)
net = Network(layers, 1)#m=1 because we don't need it

'''
w = wave.open('back_test.wav', 'w')
w.setnchannels(1)
w.setsampwidth(2)
w.setframerate(44100)
'''
#Open our sample
#samplerate, audio_raw = wav.read("../data/expanded_audio/play1.wav")
#print list(audio_raw)
#sys.exit()
#y, sr = librosa.load("test.wav")
#mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).flatten()
#print mfcc
#sys.exit()
#Get our data to feed in
'''asdf'''
'''
import os
for sample in os.listdir(os.path.abspath("./")):
    #y, sr = librosa.load("../data/audio/dennis4.wav")
    if "back_test" in sample:
        y, sr = librosa.load(sample)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).flatten()
        mfcc = np.resize(mfcc, (max_mfcc_len))
        x = mfcc.astype(np.float32)

        #Normalize with our acquired metadata 
        normalized_x = (x-mean)/stddev
        x = [np.array([np.array(normalized_x.astype(np.float32))])]
        x = theano.shared(np.asarray(x, dtype=theano.config.floatX), borrow=True)

        predicted_output_index = net.predict(x)[0][0]
        predicted_word = vocab[predicted_output_index]
        print sample, predicted_output_index, predicted_word
'''
'''
asdf
sys.exit()
'''
print "Speak now"
while True:
    l, data = inp.read()
    audio_frame_raw = np.fromstring(data, dtype='int16')

    #we were combining them all together, so instead of multiple arrays of len(1024) [1024, 1024...] we just combine them all
    #this is evidenced by the fact that all our lengths are multiples of 1024. So we extend to combine them all together so as
    #to match the way we already did it
    #w.writeframes(data)
    audio_frame = ((audio_frame_raw+32768)/65536.0).flatten()
    if len(audio_backlog) >= audio_backlog_max:
        #make room for new frame
        del audio_backlog[:1024]
        
        #So that we only start checking this after we have filled the backlog and are refreshing
        if frame_num == 43:
            frame_num = 1
            #Every second aka every 43 frames(43*1024=44032), convert our backlog to mfccs and put through net

            #w.close()
            #sys.exit()
            print "Stop Speaking"
            y, sr = np.array(audio_backlog), 44100
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).flatten()
            print len(mfcc)
            mfcc = np.resize(mfcc, (max_mfcc_len))
            x = mfcc.astype(np.float32)

            #Normalize with our acquired metadata 
            normalized_x = (x-mean)/stddev
            x = [np.array([np.array(normalized_x.astype(np.float32))])]
            x = theano.shared(np.asarray(x, dtype=theano.config.floatX), borrow=True)

            '''
            asdf
            '''
            '''
            import os
            for sample in os.listdir(os.path.abspath("./")):
                #y, sr = librosa.load("../data/audio/dennis4.wav")
                if ".wav" in sample:
                    y, sr = librosa.load(sample)
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).flatten()
                    mfcc = np.resize(mfcc, (max_mfcc_len))
                    x = mfcc.astype(np.float32)

                    #Normalize with our acquired metadata 
                    normalized_x = (x-mean)/stddev
                    x = [np.array([np.array(normalized_x.astype(np.float32))])]
                    x = theano.shared(np.asarray(x, dtype=theano.config.floatX), borrow=True)
            '''
            '''
            asdf
            '''

            predicted_output_index = net.predict(x)[0][0]
            predicted_word = vocab[predicted_output_index]
            print predicted_output_index, predicted_word
            sys.exit()

        else:
            #print "Listening"
            frame_num+=1

    audio_backlog.extend(audio_frame)

'''
#Convert sample to mfccs
x_mfccs = np.resize(mfcc, (max_mfcc_len)).flatten()
x = x_mfccs.astype(np.float32)
normalized_x = (x-mean)/stddev
x = [np.array([np.array(normalized_x.astype(np.float32))])]
x = theano.shared(np.asarray(x, dtype=theano.config.floatX), borrow=True)

#Get predicted output
predicted_output_index = net.predict(x)[0]
predicted_word = vocab[predicted_output_index]
'''
