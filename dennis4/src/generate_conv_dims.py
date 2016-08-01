import numpy as np

image = 51
strides = [float(s) for s in xrange(1, 2+1)]
filters = [float(f) for f in xrange(2, 116)]

for s in strides:
  for f in filters:
    c = (image-f)/s + 1
    if c.is_integer() and int(c) % 2 == 0 and c > 1:
        print "%i x %i, Stride: %i, Filter: %i x %i" % (int(c), int(c), int(s), int(f), int(f))
