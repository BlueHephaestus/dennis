#I didn't even need to use ruby for this but I already committed
require 'open-uri'

base_url = "http://soundoftext.com/static/sounds/en/"
audio_dir = "audio/google_translate/"

word_index = 0
File.open("music_words.txt", "r") do |file|
  file.each_line do |word|
    word = word.strip()

    puts "Generating ##{word_index}: '#{word}'" 

    query_url = base_url + word + ".mp3"
    output_mp3 = audio_dir + word + ".mp3"
    output_wav = audio_dir + word + ".wav"

    system("wget", "-q", "-P", audio_dir, query_url)

    system("mpg123", "-q", "-w", output_wav, output_mp3)
    system("rm", output_mp3)#Remove old mp3
    word_index += 1
  end
end

