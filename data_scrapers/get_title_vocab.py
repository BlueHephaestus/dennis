import os
import re
import enchant#For english word checking

music_dir = "/home/darkelement/downloads/music/"
forvo_data_dir = "music_words.txt"

word_list = []
#Our regexes and matching variables
re_id = re.compile(ur'[-]...........(\.mp3)')
re_normal_chars = re.compile(ur'[^a-zA-Z\' ]')
en_d = enchant.Dict("en_US")

def parse_title(title):
    #First we remove our -***********.mp3 from the end
    title = re.sub(re_id, "", title)

    #UTF-8 Normalization
    title = title.decode('utf-8')

    #Lowercase
    title = title.lower()

    #Remove every non space or alphabet character or apostrophe
    #Some of these MEPs are just fucking weird with their character choices holy shit
    title = re.sub(re_normal_chars, "", title)

    #Split on spaces
    title_words = title.split()

    #Remove every element that isn't an english word
    #Consider adding proper nouns and more dictionaries here
    title_words = [w for w in title_words if en_d.check(w)]
    
    return title_words if title_words else None

def regenerate_vocab():
    for f in os.listdir(os.path.abspath(music_dir)):
        f = os.path.abspath(music_dir + f)

        if os.path.isdir(f):
            for sample in os.listdir(f):
                title_words = parse_title(sample)
                #If we still have anything left we add if we haven't already got the word
                if title_words:
                    for title_word in title_words:
                        if title_word not in word_list:
                            word_list.append(title_word)
    f = open(forvo_data_dir, "w")
    for word in word_list:
        f.write("%s\n" % word)
    f.close()

regenerate_vocab()
