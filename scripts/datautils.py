import numpy as np
import random
import concurrent.futures
import functools
from math import floor, log, sqrt
from typing import Dict

def power_law_distributed_rng(n, x_min, scale, seed=None):
    # This algorithm is taken from Appendix D in Clauset et al's"Power Law Distributions
    # in Empirical Data", SIAM Review, Vol 51 (4), pp. 661-703
    rng = np.random.default_rng(seed)
    sample = rng.random((n,1))
    for i in range(n):
        r = sample[i]
        sample[i] = x_min * (1-r)**(-1/(scale-1))
    return sorted(sample)

def generalized_zeta(alpha, x_min, n):
    val = 0
    for i in range(1,n):
        val += (i + x_min)/alpha

    return val

def continuous_power_mle(data, x_min):
    # Clauset et al
    n = len(data)

    sumfactor = 0
    for i in range(n):
        sumfactor += log(data[i]/x_min)
    
    exponent = 1 + n/sumfactor
    err = (exponent - 1)/sqrt(n)

    return -1*exponent, err

def discrete_power_mle_approx(data, x_min_index):
    '''
    Expects `data` to be in decreasing order. Returns an estimate of the scaling parameter.
    '''
    x_min = data[x_min_index]
    return continuous_power_mle(data[x_min_index:], x_min-0.5)

def choose_proportional_dict(d: Dict, total_size):
    '''
    '''

    #print(f"CP got {d}")
    if total_size == 0:
        raise ValueError("At least one element in `sizes` must be non-zero.")

    r = total_size * random.random()
    #print(f"r = {r}")
    bin_start = 0.0
    bin_end = 0.0
    for k,v in d.items():
        bin_end += v
        if bin_end >= r >= bin_start:
            return k
        bin_start = bin_end
        

def choose_proportional(ls, sizes, n_sizes):
    '''
    Randomly chooses and returns an element from the list `ls` proportional to its
    size as determined by the corresponding index in `sizes`. In other words, draw
    an element of `ls` out of a hat, where each `ls[i]` has `sizes[i]` entries
    in the draw.

    NOTE: `ls` and `sizes` must be one-to-one, i.e. `ls[i]` has a size of `sizes[i]`
    '''
    total_size = 0
    for val in sizes:
        total_size += val

    if total_size == 0:
        raise ValueError("At least one element in `sizes` must be non-zero.")

    proportions = sizes.copy()

    r = total_size * random.random()
    bin_start = 0.0
    bin_end = 0.0
    temp = 0
    for i in range(n_sizes):
        temp = float(proportions[i])
        if temp == 0: # skip zero-weighted entries
            continue
        bin_end += temp
        if r >= bin_start and r <= bin_end:
            return ls[i]
        bin_start = bin_end
