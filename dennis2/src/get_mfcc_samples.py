import librosa
import scipy.io.wavfile as wav
import numpy as np
import os
import cPickle, gzip #for storing our data 
import pickle

mfcc_data_dir = "../data/mfcc_samples.pkl.gz"
training_data_dir = "../data/training_data/"
validation_data_dir = "../data/validation_data/"
test_data_dir = "../data/test_data/"

#Training, validation, and test
initial_sizes = [80, 20, 20]
n_mfcc = 20
max_audio_len = 70
max_len = 1444

def get_mfcc_data(data_dir, data):

    #Get the mfccs of our data
    print "Getting MFCCs..."
    label_num = 0
    sample_num = 0
    for f in os.listdir(os.path.abspath(data_dir)):
        f = os.path.abspath(data_dir + f)

        if os.path.isdir(f):
            for sample in os.listdir(f):
                input_fname = f + "/" + sample
                print "\tAugmenting %s..." % sample

                y, sr = librosa.load(input_fname)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).flatten()
                mfcc = np.resize(mfcc, (max_len))

                data[0][sample_num] = mfcc
                data[1][sample_num] = label_num

                sample_num+=1
        label_num += 1

    return data

def regenerate_mfcc_data():
    #training_data = [np.zeros(shape=(80, max_audio_len), dtype=np.float32), np.zeros(shape=(80), dtype=np.int)]
    training_data = [np.zeros(shape=(80, max_len), dtype=np.float32), np.zeros(shape=(80), dtype=np.int)]
    validation_data = [np.zeros(shape=(20, max_len), dtype=np.float32), np.zeros(shape=(20), dtype=np.int)]
    test_data = [np.zeros(shape=(20, max_len), dtype=np.float32), np.zeros(shape=(20), dtype=np.int)]

    training_data = get_mfcc_data(training_data_dir, training_data)
    validation_data = get_mfcc_data(validation_data_dir, validation_data)
    test_data = get_mfcc_data(test_data_dir, test_data)
    
    f = gzip.open(mfcc_data_dir, "wb")
    pickle.dump((training_data, validation_data, test_data), f)
    f.close()

regenerate_mfcc_data()

