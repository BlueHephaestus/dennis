import curses, alsaaudio, wave

word = "misc"

stdscr = curses.initscr()
curses.cbreak()
stdscr.keypad(1)
stdscr.nodelay(1)#So we're non-blocking

stdscr.refresh()

recording = False
recording_num = 0

key = ''
while key != ord('q'):
    key = stdscr.getch()
    if key == ord(' '):
        if recording:
            #Already recording, stop
            recording = False

            w.close()

            stdscr.addstr(0,0,"Stopped recording #{0}".format(recording_num))
            stdscr.refresh()
            recording_num+=1

        else:
            #Not recording, start
            recording = True

            inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE)
            inp.setchannels(1)
            inp.setrate(44100)
            inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)
            inp.setperiodsize(1024)

            w = wave.open('../data/audio/%s%i.wav' % (word, recording_num), 'w')
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(44100)

            stdscr.addstr(0,0,"Started recording #{0}".format(recording_num))
            stdscr.refresh()

    if recording:
        #continue updating ourselves
        l, data = inp.read()
        w.writeframes(data)

curses.endwin()

