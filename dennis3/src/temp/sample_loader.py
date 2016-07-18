import librosa
import scipy.io.wavfile as wav
import numpy as np
import os
import cPickle, gzip #for storing our data 
import pickle

mfcc_archive_dir = "../data/mfcc_samples.pkl.gz"
mfcc_expanded_archive_dir = "../data/mfcc_expanded_samples.pkl.gz"
mfcc_data_dir = "../data/audio/"
mfcc_expanded_data_dir = "../data/expanded_audio"
'''
training_data_dir = "../data/audio/"
validation_data_dir = "../data/audio/"
test_data_dir = "../data/audio/"
'''

#Note: Only really works when we've already got all the data expanded

#Training, validation, and test
expansion_factor = 3
n_mfcc = 20
max_audio_len = 70
max_len = 1444

initial_sizes = [80*expansion_factor, 20*expansion_factor, 20*expansion_factor]

def remove_digits(s)
    return re.sub("\d+", "", s)#Remove digits if they exist(in the case of wikimedia)

def get_mfcc_data(data_dir, expanded_data_dir):

    sample_dict = {}
    if expansion_factor > 1:
        #Expand the data
        print "Expanding Data..."
        f_index = 0
        for f in os.listdir(os.path.abspath(data_dir)):
            f = os.path.abspath(data_dir + f)

            if os.path.isdir(f):
                f_index += 1
                #local_sample_total = len(os.listdir(f))#for use with new increments of filenames
                #local_sample_num = local_sample_total
                for sample in os.listdir(f):

                    sample = remove_digits(sample)

                    #Always get our specific number instance for this sample
                    if sample in sample_dict:
                        sample_dict["sample"] += 3
                    else:
                        sample_dict["sample"] = 0
                    local_sample_index = sample_dict["sample"]

                    input_fname = f + "/" + sample
                    print "\tAugmenting %s..." % sample

                    #For our output dir
                    expanded_f = f.split("/")
                    expanded_f[-2] = "expanded_audio"
                    expanded_f = "".join(expanded_f)

                    #Do our librosa stuff
                    #input_fname = expanded_f + "/" + "%i.wav" % (local_sample_index)
                    output_slow_fname = expanded_f + "/" + "%i.wav" % (local_sample_index+1)
                    output_fast_fname = expanded_f + "/" + "%i.wav" % (local_sample_index+2)

                    y, sr = librosa.load(input_fname)
                    y_slow = librosa.effects.time_stretch(y, 0.9)
                    y_fast = librosa.effects.time_stretch(y, 1.1)

                    librosa.output.write_wav(output_slow_fname, y_slow, sr)
                    librosa.output.write_wav(output_fast_fname, y_fast, sr)




                    y, sr = librosa.load(input_fname)
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).flatten()
                    mfcc = np.resize(mfcc, (max_len))

                    data[0][sample_num] = mfcc
                    data[1][sample_num] = label_num




        data_dir = expanded_data_dir

        #Get the mfccs of our data
        print "Getting MFCCs..."
        label_num = 0
        sample_num = 0
        sample_dict = {}#Gotta revive this so we can 
        for expanded_f in os.listdir(os.path.abspath(expanded_data_dir)):
            expanded_f = os.path.abspath(expanded_data_dir + expanded_f)

            if os.path.isdir(expanded_f):
                for sample in os.listdir(expanded_f):
                    input_fname = expanded_f + "/" + sample
                    print "\tAugmenting %s..." % sample

                    if sample in sample_dict:
                        sample_dict["sample"] += 3
                    else:
                        sample_dict["sample"] = 0
                    local_sample_index = sample_dict["sample"]

                    y, sr = librosa.load(input_fname)
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).flatten()
                    mfcc = np.resize(mfcc, (max_len))

                    data[0][sample_num] = mfcc
                    data[1][sample_num] = label_num

                    sample_num+=1
            label_num += 1

    return data

def regenerate_mfcc_data(archive_dir):
    '''
    training_data = [np.zeros(shape=(80*expansion_factor, max_len), dtype=np.float32), np.zeros(shape=(80*expansion_factor), dtype=np.int)]
    validation_data = [np.zeros(shape=(20*expansion_factor, max_len), dtype=np.float32), np.zeros(shape=(20*expansion_factor), dtype=np.int)]
    test_data = [np.zeros(shape=(20*expansion_factor, max_len), dtype=np.float32), np.zeros(shape=(20*expansion_factor), dtype=np.int)]
    '''

    training_data, validation_data, test_data = get_mfcc_data(mfcc_data_dir, mfcc_expanded_data_dir)
    '''
    training_data = get_mfcc_data(training_data_dir, training_data)
    validation_data = get_mfcc_data(validation_data_dir, validation_data)
    test_data = get_mfcc_data(test_data_dir, test_data)
    '''
    
    f = gzip.open(archive_dir, "wb")
    pickle.dump((training_data, validation_data, test_data), f)
    f.close()

regenerate_mfcc_data("../data/mfcc_expanded_samples.pkl.gz")



