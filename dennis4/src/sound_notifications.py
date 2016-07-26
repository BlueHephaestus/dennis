import os

def emit_beeps(freq, duration, num_beeps=1):
  for beep in range(num_beeps):
    os.system("( speaker-test -t sine -f %f > tmp.txt)& pid=$! ; sleep %fs ; kill -9 $pid" % (freq, duration))

def default_beeps():
  emit_beeps(350, 0.2, 1)
  emit_beeps(450, 0.5, 1)
  emit_beeps(350, 0.4, 1)

