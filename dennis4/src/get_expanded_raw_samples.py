import sys
import os
import re
import shutil
import librosa
import scipy
from scipy import io as sp
import numpy as np

#This program takes our unformatted data from the data scrapers and moves the files into one big file
data_dir = "../data/audio"
expanded_data_dir = "../data/expanded_audio"

def get_name(s):
    s = re.sub("\d+", "", s)#Remove digits if they exist(in the case of wikimedia)
    s = re.sub("(\.wav)", "", s)#Remove filename
    return s

def expand_audio(data_dir, expanded_data_dir):

    if os.listdir(os.path.abspath(expanded_data_dir)) != []:
        #We've already expanded
        return

    sample_dict = {}#For keeping count of repeated samples
    sample_num = 0#For global pretty print number

    #Expand the data
    print "Expanding Data..."
    for sample in os.listdir(os.path.abspath(data_dir)):
        sample_num += 1
        #f = os.path.abspath(data_dir + "/" + f)

        #if os.path.isdir(f):
        #for sample in os.listdir(f):

        input_fname = data_dir + "/" + sample#Before we get the name
        sample = get_name(sample)

        if sample in sample_dict:
            sample_dict[sample] += 3#Expansion factor
        else:
            sample_dict[sample] = 0
        local_sample_index = sample_dict[sample]

        #output_original_fname = expanded_data_dir + "/" + "%s%i.wav" % (sample, local_sample_index)
        #shutil.copyfile(input_fname, output_original_fname)#Copy original

        #local_sample_total = len(os.listdir(f))#for use with new increments of filenames
        #local_sample_num = local_sample_total

        #Do our librosa stuff
        print "\tAugmenting #%i: %s%i.wav..." % (sample_num, sample, local_sample_index)
        #input_fname = expanded_f + "/" + "%i.wav" % (local_sample_index)
        output_original_fname = expanded_data_dir + "/" + "%s%i.wav" % (sample, local_sample_index)
        output_slow_fname = expanded_data_dir + "/" + "%s%i.wav" % (sample, local_sample_index+1)
        output_fast_fname = expanded_data_dir + "/" + "%s%i.wav" % (sample, local_sample_index+2)

        #print output_original_fname
        try:
            #Get our noise
            y, sr = librosa.load("noise.wav")
            noise = list(y)[8192:16384]

            y, sr = librosa.load(input_fname)

            #Add noise before expanding
            y = list(y)
            noise.extend(y)
            y = np.array(noise, dtype=np.float32)

            #Expand
            y_slow = librosa.effects.time_stretch(y, 0.9)
            y_fast = librosa.effects.time_stretch(y, 1.1)

            #Write the same one using librosa so we get float 32 bit endians for all of them 
            librosa.output.write_wav(output_original_fname, y, sr)
            librosa.output.write_wav(output_slow_fname, y_slow, sr)
            librosa.output.write_wav(output_fast_fname, y_fast, sr)
        except:
            #This shouldn't happen anymore because of the code in get_raw_samples.py,
            #However I have it just in case something arises.
            print "Error Augmenting"
            print os.stat(input_fname).st_size
            sys.exit()
            #print "\tFaulty File, removing..."
            #os.remove(output_original_fname)



expand_audio(data_dir, expanded_data_dir)
