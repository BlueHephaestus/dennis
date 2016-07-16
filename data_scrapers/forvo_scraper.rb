require 'open-uri'
require 'nokogiri'
require 'mechanize'
require 'base64'

base_url = "http://forvo.com/word/"
base_download_url = "http://audio.forvo.com/audios/mp3/q/c/qc_"
audio_dir = "audio/forvo/"

def get_html(url)
  while true do
    begin
      return Nokogiri::HTML.parse(open url)
    rescue
      #puts "Something went wrong, loading page again..."
      sleep(4)
    end
  end
end

word_index = 0
File.open("music_words.txt", "r") do |file|
  file.each_line do |word|
    word = word.strip()

    puts "Generating ##{word_index}: '#{word}'" 

    query_url = base_url + word
    word_page = get_html(query_url)

    #Check if it's a 404, if not, then go through each mp3 on the page
    if not word_page.css('title')[0].text.include? "Page not found"
      #Ayy they've got results, get the english ones
      
      play_links = word_page.css('a.play')
      download_links = word_page.css('p.download a') 
      #p play_links, "asdf"

      links = play_links.zip(download_links)
      
      link_index = 0
      links.each do |play_link, download_link|
        
        puts "\"#{play_link}\""
        puts "\"#{download_link}\""
        if play_link != nil and download_link != nil
          play_link = play_link["onclick"]
          download_link = download_link["href"]#Get actual link
          if download_link.include? "/en/" and download_link.include? word#Is it english? Does it have our word?
            #Our indices still stay in order because the foreign pronunciations come after english ones
            puts Base64.decode64(play_link.split('\'')[0])
            puts Base64.decode64(play_link.split('\'')[1])
            puts Base64.decode64(play_link.split('\'')[2])
            puts Base64.decode64(play_link.split('\'')[3])
            puts Base64.decode64(play_link.split('\'')[4])

            play_link = Base64.decode64(play_link.split('\'')[1])
            query_url = base_download_url + play_link
            #puts audio_dir + word, query_url
            system("wget", "-q", "-P", audio_dir + word, query_url)
          else
            break
          end
        else
          break
        end
        link_index+=1
      end
    end




    output_mp3 = audio_dir + word + ".mp3"
    output_wav = audio_dir + word + ".wav"
    puts output_mp3, output_wav

    exit
    #system("mpg123", "-q", "-w", output_wav, output_mp3)
    #system("rm", output_mp3)#Remove old mp3
    word_index += 1
  end
end


