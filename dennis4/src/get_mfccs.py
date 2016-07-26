import librosa
import scipy.io.wavfile as wav
import numpy as np
import os
import cPickle, gzip #for storing our data 
import pickle
import re

archive_dir = "../data/mfcc_samples.pkl.gz"
expanded_archive_dir = "../data/mfcc_expanded_samples.pkl.gz"
data_dir = "../data/audio"
expanded_data_dir = "../data/expanded_audio"

n_mfcc = 20#The default
#max_audio_len = 70
#max_mfcc_len = 1560
#max_mfcc_len = 1600
max_mfcc_len = 2209

def get_name(s):
    s = re.sub("\d+", "", s)#Remove digits if they exist(in the case of wikimedia)
    s = re.sub("(\.wav)", "", s)#Remove filename
    return s

def get_mfccs(data_dir):
    #Determine the dims of our big data array#lol big data KappaPride
    sample_total = len(os.listdir(os.path.abspath(data_dir)))
    data = [np.zeros(shape=(sample_total, max_mfcc_len), dtype=np.float32), np.zeros(shape=(sample_total), dtype=np.int)]
    '''
    data =
                      X                               Y
    |number     [mfcc1, mfcc2, ..., mfcc max len], [label]
    |of         [mfcc1, mfcc2, ..., mfcc max len], [label]
    |samples    [mfcc1, mfcc2, ..., mfcc max len], [label]
    v
    '''

    #Get the mfccs of our data
    print "Getting MFCCs..."
    label_total = 0#Increments so that each sample has it's own label
    label_dict = {}#Keep track of sample:label numbers 

    for sample_num, sample in enumerate(os.listdir(os.path.abspath(data_dir))):
        print "\tAugmenting #%i: %s..." % (sample_num, sample)

        input_fname = data_dir + "/" + sample
        sample = get_name(sample)

        if sample not in label_dict:
            label_dict[sample] = label_total
            label_total+=1
            print label_dict
        label_num = label_dict[sample] 

        y, sr = librosa.load(input_fname)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).flatten()
        mfcc = np.resize(mfcc, (max_mfcc_len))

        data[0][sample_num] = mfcc
        data[1][sample_num] = label_num

    return data

def regenerate_mfccs(archive_dir, get_expanded=False):
    '''
    #training_data = [np.zeros(shape=(80, max_audio_len), dtype=np.float32), np.zeros(shape=(80), dtype=np.int)]
    training_data = [np.zeros(shape=(80, max_mfcc_len), dtype=np.float32), np.zeros(shape=(80), dtype=np.int)]
    validation_data = [np.zeros(shape=(20, max_mfcc_len), dtype=np.float32), np.zeros(shape=(20), dtype=np.int)]
    test_data = [np.zeros(shape=(20, max_mfcc_len), dtype=np.float32), np.zeros(shape=(20), dtype=np.int)]

    training_data = get_mfcc_data(training_data_dir, training_data)
    validation_data = get_mfcc_data(validation_data_dir, validation_data)
    test_data = get_mfcc_data(test_data_dir, test_data)
    '''
    archive_dir = "../data/mfcc_samples.pkl.gz"
    expanded_archive_dir = "../data/mfcc_expanded_samples.pkl.gz"
    data_dir = "../data/audio"
    expanded_data_dir = "../data/expanded_audio"

    if get_expanded:
        data_dir = expanded_data_dir
        archive_dir = expanded_archive_dir

    data = get_mfccs(data_dir)
    
    f = gzip.open(archive_dir, "wb")
    pickle.dump((data), f, protocol=-1)
    f.close()

regenerate_mfccs(archive_dir, get_expanded=False)
