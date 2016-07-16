import subprocess
import sys

audio_dir = "audio/espeak/"

def textToWav(text, file_name):
   subprocess.call(["espeak", "-w" + file_name, text])

def regenerate_wavs():
    f = open("music_words.txt", "r")
    for word_index, word in enumerate(f.readlines()):
        word = word.strip()
        print "Generating #%i: '%s'" % (word_index, word)
        textToWav(word, audio_dir + word + ".wav")

regenerate_wavs()
