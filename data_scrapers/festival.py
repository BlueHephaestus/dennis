import subprocess
import sys

audio_dir = "audio/festival/"

def textToWav(input_filename, output_filename):
   subprocess.call(["text2wave", "-o", output_filename, input_filename])

def regenerate_wavs():
    f = open("music_words.txt", "r")
    for word_index, word in enumerate(f.readlines()):
        word = word.strip()
        w = open("temp.txt", "w")
        w.write(word)
        w.close()
        print "Generating #%i: '%s'" % (word_index, word)
        textToWav("temp.txt", audio_dir + word + ".wav")

regenerate_wavs()
