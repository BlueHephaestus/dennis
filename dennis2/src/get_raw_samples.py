import scipy.io.wavfile as wav
import numpy as np
import os  #for directories
import cPickle, gzip #for storing our data
import pickle

#They recommended just doing this so we get a spectrogram instead of using mfccs
expanded_data_dir = "../data/expanded_samples.pkl.gz"
raw_data_dir = "../data/raw_samples.pkl.gz"
mfcc_data_dir = "../data/mfcc_samples.pkl.gz"
training_data_dir = "../data/training_data/"
validation_data_dir = None
test_data_dir = "../data/test_data/"
#max_audio_len = 83968
#max_audio_len = 84100
#max_audio_len = 784
max_audio_len = 83521#289*289

max = 0.0
min = 0.0

'''
#Get maximum length
for f in os.listdir(os.path.abspath(training_data_dir)):
    f = os.path.abspath(training_data_dir + f)

    if os.path.isdir(f):
        for sample_index, sample in enumerate(os.listdir(f)):
            #get our spectrogram for the sample
            samplerate, audio_raw = wav.read(f + "/" + sample)
            if len(audio_raw) > max:
                max = len(audio_raw)

for f in os.listdir(os.path.abspath(test_data_dir)):
    f = os.path.abspath(training_data_dir + f)

    if os.path.isdir(f):
        for sample_index, sample in enumerate(os.listdir(f)):
            #get our spectrogram for the sample
            samplerate, audio_raw = wav.read(f + "/" + sample)
            if len(audio_raw) > max:
                max = len(audio_raw)
print max
'''

def get_raw_data(data_dir, data):
    label_num = 0
    sample_num = 0
    for f in os.listdir(os.path.abspath(data_dir)):
        f = os.path.abspath(data_dir + f)

        label = f.split("/")[-1]

        if os.path.isdir(f):
            for sample in os.listdir(f):
                print sample
                #get our spectrogram for the sample
                samplerate, audio_raw = wav.read(f + "/" + sample)

                #start with zeros so we have our blank trailing space
                audio = np.zeros(max_audio_len)

                #convert our signed integer to an unsigned one by adding 32768, then divide by max of unsigned integer to get percentage and use that as our float value
                #fill as far as we've got
                for a_index, a in enumerate(audio_raw):
                    audio[a_index] = float((a+32768)/65536.0)
                    if a_index == max_audio_len-1:
                        break

                data[0][sample_num] = audio
                data[1][sample_num] = label_num
                
                sample_num+=1
        label_num += 1
    return data

def regenerate_raw_data():
    training_data = [np.zeros(shape=(80, max_audio_len), dtype=np.float32), np.zeros(shape=(80), dtype=np.int)]
    validation_data = [np.zeros(shape=(20, max_audio_len), dtype=np.float32), np.zeros(shape=(20), dtype=np.int)]
    test_data = [np.zeros(shape=(20, max_audio_len), dtype=np.float32), np.zeros(shape=(20), dtype=np.int)]

    training_data = get_raw_data(training_data_dir, training_data)
    validation_data = get_raw_data(test_data_dir, test_data)
    test_data = get_raw_data(test_data_dir, test_data)
    
    f = gzip.open(raw_data_dir, "wb")
    pickle.dump((training_data, validation_data, test_data), f)
    f.close()


#regenerate_data()
