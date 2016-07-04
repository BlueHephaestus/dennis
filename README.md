#Description

While much of the code here is just my experimentation with machine learning, like [here](https://github.com/DarkElement75/machine-learning-tinkering), it's also for the project that i've decided to call Dennis, who I plan on using to manage my local music library with voice commands. I will be doing incremental updates inside each Mark, but when I have reached certain goals I had for it or it has been upgraded a significant amount (i.e. when I decide I want to) I will add a new version file and description(Mark 1, Mark 2, Mark 3). Below I will summarize the improvements done in each Mark, and the current work if it is the one currently in progres.

##Mk. 1
The first (barely) functional version, took in raw audio spectrogram from total of 120 wav files containing either "dennis" or "play", and output the word given. With **extremely** basic functionality it obtained 67% accuracy, or a 33% margin of error. 

These are results that already have been greatly improved upon in Mk. 2, which is good news considering the results were on the training data, not even the validation and test accuracies!

Used Fully connected sigmoid layers with 83968 inputs, 100 and 30 neurons in the two hidden layers, and 2 softmax outputs.

##Mk. 2
Current WIP version

Used same samples as last time, with 80 training, 20 validation, and 20 test, divided evenly for each word.

After changing to using [Mel-frequency cepstral coefficients](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum), AKA MFCCs, it obtained(as of the last experiment) 100% training accuracy(as hoped), 90% validation accuracy, and 72.5% test accuracy (iirc). 

I've been working on my own momentum implementation since I feel bad about just cheating and using theano's, so we'll see how that goes and if it improves things. It tests extremely fast using the Mk. 1 sorta-deep NN setup, so that's what a lot of testing new features will be on. I'll also likely add some matplotlib graphs, since I updated that implementation as well.

dennis2.py is based off of Network3.py by mnielsen, as I modify, improve, tinker, and add features until it's an entirely new Ship of Theseus. 

Will use a deeper convolutional model when this is finalized, so current high scores will likely be updated.

I will likely start Mk. 3 after I finish adding the following:

1. Momentum based Stochastic Gradient Descent
2. Automatic scheduling of learning rates
..*And many other things that i'd like to automatically schedule to see how they impact things
3. Expand the MFCC data as well
..*Already did this a bit with the raw samples, increasing speed to 1.1 and decreasing to 0.9, but actually yielded far worse results in all sections.
4. More improvements to graphing outputs
5. Possibly more automation for finding Hyper Parameters
6. More optimization techniques possibly
7. If I believe it will not take a much larger amount of work/mks to get the test accuracy good enough, i'll move on to actually implementing it into my computer. If this works out, i'll add how I did that here as well.



##Credit to Michael Nielsen for his [awesome book on machine learning](http://neuralnetworksanddeeplearning.com/chap1.html), and also for his [extremely helpful and well-written neural network and machine learning code](https://github.com/mnielsen/neural-networks-and-deep-learning/). I can't thank him enough for getting me started and for the resources he has provided to everyone.

##License
MIT License

Copyright (c) 2016 Blake Edwards / Dark Element

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
