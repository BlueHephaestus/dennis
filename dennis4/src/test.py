import numpy as np

a = [[1, ["a", "b", "c"], 3, 4], [5, 6, 7, 8]]

def unison_shuffle(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

unison_shuffle(a[0], a[1])
print a
a_0 = np.split(a[0], [3])#basically the lines we cut to get our 3 subsections
a_1 = np.split(a[1], [3])#basically the lines we cut to get our 3 subsections
print a_0, a_1
