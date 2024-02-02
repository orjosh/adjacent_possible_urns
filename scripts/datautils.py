import numpy as np
import random

def power_law_distributed_rng(n, x_min, scale, seed=None):
    # This algorithm is taken from Appendix D in Clauset et al's"Power Law Distributions
    # in Empirical Data", SIAM Review, Vol 51 (4), pp. 661-703
    rng = np.random.default_rng(seed)
    sample = rng.random((n,1))
    for i in range(n):
        r = sample[i]
        sample[i] = x_min * (1-r)**(-1/(scale-1))
    return sorted(sample)

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

    proportions = sizes
    for i in range(n_sizes):
        proportions[i] = proportions[i] / total_size

    r = random.random() # btwn 0 and 1
    bin_start = 0.0
    bin_end = 0.0
    for i in range(n_sizes):
        bin_end += proportions[i]
        if r >= bin_start and r <= bin_end:
            return ls[i]
        bin_start = bin_end 