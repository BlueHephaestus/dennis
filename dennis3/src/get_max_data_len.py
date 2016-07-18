import os
import scipy.io.wavfile as wav
import librosa

data_dir = "../data/audio"
expanded_data_dir = "../data/expanded_audio"

def get_max_raw_audio_len(data_dir):
    max = 0.0
    #Get maximum length
    for sample_index, sample in enumerate(os.listdir(os.path.abspath(data_dir))):
       #get our spectrogram for the sample
        samplerate, audio_raw = wav.read(data_dir + "/" + sample)
        if len(audio_raw) > max:
            max = len(audio_raw)
    return max

def get_max_mfcc_len(data_dir):
    max = 0.0
    n_mfcc = 20

    #Get the mfccs of our data
    print "Getting MFCCs..."
    for sample in os.listdir(os.path.abspath(data_dir)):
        input_fname = data_dir + "/" + sample
        print "\tAugmenting %s..." % sample

        y, sr = librosa.load(input_fname)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).flatten()

        if len(mfcc) > max:
            max = len(mfcc)
    return max
                

#print get_max_raw_audio_len(data_dir)
#print get_max_raw_audio_len(expanded_data_dir)
print get_max_mfcc_len(expanded_data_dir)
    

