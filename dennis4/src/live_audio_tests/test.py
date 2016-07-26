import alsaaudio, sys, librosa
import numpy as np

'''
if recording:
    w.close()
else:
'''
period_size = 1024
inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE)
inp.setchannels(1)
inp.setrate(44100)
inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)
inp.setperiodsize(period_size)

'''
w = wave.open('%s_data/%s/%s.wav' % (data_type, sample, str(recording_num)), 'w')
w.setnchannels(1)
w.setsampwidth(2)
w.setframerate(44100)
'''
audio_backlog = []
#i'm calling this at 65536 frames, being 64 frames of 1024 period size each(64*1024=65536)
#Should be enough to catch when dennis is said
#Not sure how much processing this will take up
#audio_backlog_max = 65536
#Using this so our mfccs remain smaller than max len and we don't lose data
audio_backlog_max = 56320
n_mfcc = 20
max_mfcc_len = 2209
frame_num = 1

while True:
    l, data = inp.read()
    audio_frame_raw = np.fromstring(data, dtype='int16')

    #we were combining them all together, so instead of multiple arrays of len(1024) [1024, 1024...] we just combine them all
    #this is evidenced by the fact that all our lengths are multiples of 1024. So we extend to combine them all together so as
    #to match the way we already did it
    audio_frame = ((audio_frame_raw+32768)/65536.0).flatten()
    if len(audio_backlog) >= audio_backlog_max:
        #make room for new frame
        del audio_backlog[:1024]
        
        #So that we only start checking this after we have filled the backlog and are refreshing
        if frame_num == 43:
            frame_num = 1
            #Every second aka every 43 frames(43*1024=44032), convert our backlog to mfccs and put through net
            y, sr = np.array(audio_backlog), 44100
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).flatten()
            print len(mfcc)
            mfcc = np.resize(mfcc, (max_mfcc_len))
        else:
            frame_num+=1

    audio_backlog.extend(audio_frame)
    '''
    #y, sr = librosa.load(input_fname)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).flatten()
    mfcc = np.resize(mfcc, (max_mfcc_len))
    '''
    #for a_index, a in enumerate(audio_frame_raw):
        #audio_frame.append(float((a+32768)/65536.0))
    #print np.abs(audio_frame).mean()
    #w.writeframes(data)

