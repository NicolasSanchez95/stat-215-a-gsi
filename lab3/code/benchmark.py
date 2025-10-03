import numpy as np
import time
import sys

# import the similarity() function from similarity.py
from similarity import similarity

# USAGE: python benchmark.py 10
# where 10 is the number of times similarity() is called
# when benchmarking (to reduce variance)

# OUTPUT: average time taken to run similarity() in seconds
# first for q=5000, then for q=29000.

# read the number of times similarity() is called
# from the command line
assert len(sys.argv) == 2, "Usage: python benchmark.py 10"
T = int(sys.argv[1])

# for different values of q
for q in [5000, 29000]:

    # create random membership vectors
    m1 = np.random.randint(0, 10, q)
    m2 = np.random.randint(0, 10, q)

    # call similarity once so it compiles
    # (if using numba or cython or something)
    similarity(m1, m2)

    # get the start time
    start = time.time()

    # run similarity() T times
    for i in range(T):
        sim = similarity(m1, m2)

    # get the end time
    end = time.time()

    # print the time per iteration
    print((end-start)/T)

