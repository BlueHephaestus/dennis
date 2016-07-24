#Description

While much of the code here is just my experimentation with machine learning, like [here](https://github.com/DarkElement75/machine-learning-tinkering), it's also for the project that i've decided to call Dennis, who I plan on using to manage my local music library with voice commands. I will be doing incremental updates inside each Mark, but when I have reached certain goals I had for it or it has been upgraded a significant amount (i.e. when I decide I want to) I will add a new version file and description(Mark 1, Mark 2, Mark 3). Below I will summarize the improvements done in each Mark, and the current work if it is the one currently in progres.

Note: If you ever want the data I used, contact me. I don't want to be like all the sources that keep their data private or at a price.

##MK. 1
The first (barely) functional version, took in raw audio spectrogram from total of 120 wav files containing either "dennis" or "play", and output the word given. With **extremely** basic functionality it obtained 67% accuracy, or a 33% margin of error. 

These are results that already have been greatly improved upon in Mk. 2, which is good news considering the results were on the training data, not even the validation and test accuracies!

Used Fully connected sigmoid layers with 83968 inputs, 100 and 30 neurons in the two hidden layers, and 2 softmax outputs.

##MK. 2
####Best Config:

With Hyper-Parameters of:

Mini Batch Size = 10
Learning Rate = 0.1
Momentum Coefficient = 0.0 (Wasn't functional at this time)
Regularization Rate(L2 Regularization) = 0.02
No Dropout

Input was 38*38 into Fully Connected sigmoid layers with 100 neurons, 30 neurons, and then a Softmax output layer of 2 neurons.

Training Cost: ~0.7, Training Accuracy: 100%, Validation Accuracy: ~97%, Test Accuracy: ~94% (Each saturated after ~300 epochs although sample was taken from 600 epochs just in case)

Used same samples as last time, with 80 training, 20 validation, and 20 test, divided evenly for each word.

After changing to using [Mel-frequency cepstral coefficients](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum), AKA MFCCs, it obtained(as of the last experiment) 100% training accuracy(as hoped), 90% validation accuracy, and 72.5% test accuracy (iirc). 

I've been working on my own momentum implementation since I feel bad about just cheating and using theano's, so we'll see how that goes and if it improves things. It tests extremely fast using the Mk. 1 sorta-deep NN setup, so that's what a lot of testing new features will be on. I'll also likely add some matplotlib graphs, since I updated that implementation as well.

dennis2.py is based off of Network3.py by mnielsen, as I modify, improve, tinker, and add features until it's an entirely new Ship of Theseus. 

I decided to go to MK. 3 when implementing the entirely new data setup, so several of the things I planned on doing ended up being with that. I ran one last run with MK. 2 for the graph, however here are the final things I did end up completing before moving on:

1. Automatic scheduling of learning rates(And many other things that i'd like to automatically schedule to see how they impact things) - I added this for learning rates but i'm not sure if mini batch size is a good thing to schedule. If it ends up working out, i'll schedule that as well. Until then, I did implement this for learning rates. Only with the training cost however, since the accuracy increases in large jumps due to the small amount of data to compute accuracy on.
2. Expand the MFCC data as well (Already did this a bit with the raw samples, increasing speed to 1.1 and decreasing to 0.9, but actually yielded far worse results in all sections) - Upon doing this with MFCCs I got the best data in all respects. I am hence using this in future experiments unless I have good reason to try a (extremely long) test with raw samples
3. Did several improvements to graphing outputs such as labels, titles, display on machine that is ssh'ing if being run remotely.
4. Normalized input according to distribution made from training data, also showed good improvement
5. Tried expanding with 0.8, 0.9, 1.1, and 1.2, however it yielded worse results than control and 0.9, 1.1 .

Due to some recently discovered bugs in the graphing part of things, I will not be including it since it is misleading. The results listed for MK. 2 are still accurate, however the graph would be misleading for the following reasons:

##MK. 3
####Current Results:

![Current MK. 3 Results](/dennis3/comparisons/initial_setup2.png)

Input is currently 40*40 into Fully Connected sigmoid layers with 100 neurons, 30 neurons, and then a Softmax output layer of 4936 neurons.

To get an idea of the results, it currently has it's best TRAINING accuracy as 1.6%. As for cost, it's ~3.0. This might not be so dismal if it took 5 seconds instead of 5 hours.

Many of the experiments for testing implementation of features are done with experimental_config.py, that uses the old 38*38 input, 2 output MFCC data. 

Unfortunately, the results so far are so dismal, I believe largely due to the lack of data, that I will likely not be able to take this as far as I wanted. I may record my own sayings of several other words to just have simple commands I can use, such as "shuffle", "loop", "stop", "pause", etc, however I don't know if I will due this since I am not able to take this project as far as I wanted. Even if I do have enough data(which let's be honest, I don't unless I figure out the algorithm the brain uses for generalization), then I would likely have to use a DCNN so deep that it would take longer to train than it would for me to get to university. I may add those new commands if I think it worthwhile, otherwise I will move to image recognition, reinforcement learning, or something like that.

As far as I have found, there are little to no free sources of spoken english audio files. Perhaps I could make a fancy youtube crawler to get words from videos using user-made captions, however it would likely not 100% accurate which could be devastating, and I also kinda got annoyed at the fact that it should already exist but didn't. The only sources I could find were either private or expensive. So I did my best to circumvent this, using the following:

1. I got every word in my music library (~5,000)
2. Put them into espeak, festival, and pico2wave(linux CLI text to speech) to get the resulting wav files
3. Definitely didn't put them into a 3rd party google translate text to speech, but if I did, it would have been the best TTS
4. Scraped all that I could off of wikimedia, which has words spoken by actual people.
5. After all this, I ended up with ~20,000 .wav files. 
6. Expanding with speeds of 0.9 and 1.1, I got 60,000 files (3.3 GB of data which is why the raw files aren't here)

Features added:
1. Momentum-Based Stochastic Gradient Descent 
2. Percentage Completion out of config/run/epoch
3. New setup with our 60,000 samples, 40*40(1600) inputs, and 4936 outputs
4. Cleaned up and improved the input of our samples and loading of everything drastically
5. Cut me some slack I started 3 days ago(maybe 4? idk)

Bug Fixes:
1. The reason I did not post the other graphs is because I realized it was not resetting the parameters of the network when doing a new run for each config. This has been solved, and tested rigorously to make sure it has been solved. Since the previous networks saturated quite quickly it was not very noticable, so the results listed are still accurate for the best they achieved, at least with the supplied Hyper-Parameters.

Features to add:
1. I created an algorithm to do a very efficient form of grid search, based on the way that I optimize hyper parameters such as mini batch, where values of 10, 20, 30, ... 100 are tried, then if 20 and 30 do well, it would search between 20 and 30 like 20, 21, 22, ... 30, until the best mini batch size was found. However, I realized that applying this to the entire cartesian product of hyper parameters would involve finding the best fitting polynomial function of degree number of hyper parameters(dimensions), and also finding the minimum of an n-dimensional function. So in order to make a bot to optimize hyper parameters efficiently for my machine learning, I will be making a bot to do machine learning on my hyper parameters for machine learning. Unfortunately, I kinda want this to have the best results possible by the time i'm at university, so i'm going to be doing that manually for now. Gotta have something to show, eh? :D
2. Possibly experiment with different types of Fast Fourier Transforms outside of MFCCs
3. Possibly experiment with different types of data expansion on top of current, such as background noise
4. Test Ensemble networks
5. Test more deep and convolutional setups
6. Test scheduling now that accuracy is actually changing with reasonable intervals
7. Improve automatic scheduling to work better with automation

##MK. 4

Due to what I believe is largely a lack of sufficient data and also likely computational power(or an algorithm for generalization >= the brain's), I left MK. 3 and decided on the design for MK. 4.

Instead of attempting to do a 5000 vocabulary speech recognition of ~5 samples each, I took after MK. 2 & MK. 1, recording 60 samples of four more words, and 120 of random miscellaneous words. Supplemented with the 60 each of the words from MK. 1 and MK. 2, the following is the vocab class - sample:

0. Misc -     120
1. Shuffle -  60
2. Dennis -   60
3. Next -     60
4. Play -     60
5. Pause -    60
6. Back -     60

Totalling 480 raw samples, which I expanded with speed factors of .9 and 1.1 as earlier, and transformed to MFCCs, giving 1440 MFCC samples. 

I will post the results soon, however the best Test Accuracy is ~70%. This type of topology is much faster and easier to train, while also giving me 6 voice commands to control my music with. While I would very much like to have gotten MK. 3 working, I believe that to be infeasible. I am confident that I will be able to get to a < 5% error margin on test accuracy with this layout, especially considering the 70% accuracy was on a two-hidden-layer network with 100 and 30 neurons respectively. Before I move on to the features to add, there is one that I believe warrants it's own subheading:

###CHO - Cool Hyper-Parameter Optimizer

This is the idea I mentioned earlier in the features to add section of MK. 3, come to fruition. In summary,
1. Generate vectors of Hyper Parameter(HP) values
2. Get cartesian product
3. Get average point output for each HP in the Cartesian Product via if three learning rates 0.3, 0.4, 0.5 and three mini batch sizes 10, 20, 30, we'd get the three values for (10, 0.3), (10, 0.4), (10, 0.5) and average these outputs since the effect of m=10 should be equivalent across the different learning rates. This gives us one value for our m=10(and the rest of the mini batch sizes) to use in the following steps. Note: I realize m is a relatively independent HP and can be determined as such.
4. Use these new points for each HP to generate a quadratic linear regression of the form A + Bx + Cx^2 for each HP
5. Add all of our quadratic linear regressions together to get a multivariable equation such as 5 + 4m + 7.2m^2 + .32n + 6.32n^2
6. Compute the minimum to 1e-8 of our multivariable function using scipy.optimize.minimize with respective bounds
7. Use our new minimum HP values to generate new ranges after decreasing step size, if step size is under our threshold then stop for this HP
8. Go to Number 1 until all HPs are done

This has shown to be way faster and better than me, in fact what I thought was a bug at once was actually the most efficient mini batch size I could have obtained, and it just knew better than me. While it is still in constant development and improvement, it determined the ideal(at least I hope they were ideal) HPs for the shallow network topology I mentioned earlier. Upon comparing with my own hypothesis for the best values, it gained a 20% improvement in test accuracy, the metric upon which I chose it to search them with(although this can be REALLY easily changed, I chose test accuracy because I see that as most relevant. I have considered a combination of accuracies for it to look at). 

###Now back to MK. 4:

Features Added:
1. CHO
2. Saving and Loading Networks

Features to Add:
1. Actual implementation of live audio being fed into saved networks, currently only gets one sample at a time but this should not be a problem
2. I may still improve scheduling, however I may just spend that time improving CHO instead.
3. CHO - I will be adding more efficient step ranges to minimize the amount of configs that must be run like a human would do
4. CHO - Work when there is only one run instead of only when run_count > 1 and uses average; This will be fixed very soon because new test topologies take FOREVER
5. CHO - Possibly remove some step calculations from the end if #3 works out
6. Possibly experiment with different types of Fast Fourier Transforms outside of MFCCs
7. Possibly experiment with different types of data expansion on top of current, such as background noise
8. Test Ensemble networks
9. Test more deep and convolutional setups


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
