import numpy as np

f_low = np.arange(0, 80, 5, dtype=np.int_)
f_high = np.arange(10, 90, 5, dtype=np.int_)
bands = np.stack((f_low, f_high), axis=1)

freqs = bands.mean(axis=1).astype(int)
