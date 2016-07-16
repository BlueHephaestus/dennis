require 'open-uri'
require 'nokogiri'

audio_dir = "audio/wikimedia/"

#don't forget to add audio to the end (or beginning) of each term in the file
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
    word = word.strip
    search_url = "https://commons.wikimedia.org/w/index.php?title=Special:Search&profile=default&fulltext=Search&search=%22En-us-#{word}.ogg%22&uselang=en&searchToken=chyj6ebmk4lz7pnjrkrlga2nn"
    search_results = get_html(search_url)
    if search_results.at_css(".mw-search-results")
      #If we get results
      search_results =  search_results.css("table.searchResultImage")
      search_results.each_with_index do |search_result, search_result_index|
        search_link = search_result.css("td")[1].at_css("a")["href"]
        search_result_page = get_html("https://commons.wikimedia.org" + search_link)
        query_url = search_result_page.at_css("div.fullMedia").at_css("a")["href"]

        puts "Generating #{word_index}-#{search_result_index}: '#{word}'"

        output_ogg = audio_dir + word + search_result_index.to_s + ".ogg"
        output_wav = audio_dir + word + search_result_index.to_s + ".wav"
        system("wget", "-q", "-O", output_ogg, query_url)#silent

        #convert ogg to wav
        system("oggdec", "-Q",  "-o", output_wav, output_ogg)#silent
        system("rm", output_ogg)#Remove old ogg
      end
    else
      next
    end
    word_index += 1
  end
end

