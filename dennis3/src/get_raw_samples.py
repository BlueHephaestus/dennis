import sys
import os
import re
import shutil

#This program takes our unformatted data from the data scrapers and moves the files into one big file
raw_dir = "../data/raw_audio"
new_dir = "../data/audio"

def get_name(s):
    s = re.sub("\d+", "", s)#Remove digits if they exist(in the case of wikimedia)
    s = re.sub("(\.wav)", "", s)#Remove filename
    return s

def reformat_audio_layout(raw_dir, new_dir):
  sample_num = 0#For the global number we're at

  if os.listdir(os.path.abspath(new_dir)) != []:
    #We've already reformatted
    return

  sample_dict = {}#For keeping count of repeated samples

  for f in os.listdir(os.path.abspath(raw_dir)):
      f = os.path.abspath(raw_dir + "/" + f)
      #F is our subfolder

      if os.path.isdir(f):
        for sample in os.listdir(f):
          sample_num+=1
          input_fname = f + "/" + sample
          sample = get_name(sample)

          if sample in sample_dict:
              sample_dict[sample] += 1
          else:
              sample_dict[sample] = 0
          local_sample_index = sample_dict[sample]

          output_fname = new_dir + "/" + "%s%i.wav" % (sample, local_sample_index)
          if os.stat(input_fname).st_size > 100:
              #So that if it's a broken or empty file, we don't copy.
              print "Copying #%i: %s" % (sample_num, sample)
              shutil.copyfile(input_fname, output_fname)

reformat_audio_layout(raw_dir, new_dir)
