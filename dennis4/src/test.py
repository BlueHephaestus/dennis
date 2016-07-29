import numpy as np
import os
import librosa

y, sr = librosa.load("noise.wav")
y = list(y)[8192:16384]
#print np.mean(y), np.std(y)
print len(y)
#noise = list(np.random.normal(0, .001, 1024))
#noise = list(0.01* np.random.random_sample(32768))

#Since we can't extend numpy arrays inline :(
#y = np.random.shuffle(y)
#noise = list(np.random.normal(2.907e-05, .0145983, 8192))
#noise.extend(y)
y = np.array(y, dtype=np.float32)
librosa.output.write_wav("test_noise.wav", y, sr)
os.system("aplay test_noise.wav")
