import numpy as np
import random
import concurrent.futures
import functools
from math import floor

def power_law_distributed_rng(n, x_min, scale, seed=None):
    # This algorithm is taken from Appendix D in Clauset et al's"Power Law Distributions
    # in Empirical Data", SIAM Review, Vol 51 (4), pp. 661-703
    rng = np.random.default_rng(seed)
    sample = rng.random((n,1))
    for i in range(n):
        r = sample[i]
        sample[i] = x_min * (1-r)**(-1/(scale-1))
    return sorted(sample)

def choose_proportional_lazy(old, n, ls, sizes, n_sizes):
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

    proportions = old
    _calc_proportions_inplace(proportions, (0, n), sizes, total_size)

    r = random.random() # btwn 0 and 1
    # print("after:")
    # print(proportions)
    # print(f"random: {r}")
    bin_start = 0.0
    bin_end = 0.0
    for i in range(n_sizes):
        bin_end += proportions[i]
        if r >= bin_start and r <= bin_end:
            # print(f"returning {ls[i]}")
            return (ls[i], proportions)
        bin_start = bin_end

def choose_proportional(ls, sizes, n_sizes):
    '''
    Randomly chooses and returns an element from the list `ls` proportional to its
    size as determined by the corresponding index in `sizes`. In other words, draw
    an element of `ls` out of a hat, where each `ls[i]` has `sizes[i]` entries
    in the draw.

    NOTE: `ls` and `sizes` must be one-to-one, i.e. `ls[i]` has a size of `sizes[i]`
    '''
    print(f"Got:\n\tls: {ls}\n\tsizes: {sizes}\n\tn_sizes: {n_sizes}")
    total_size = 0
    for val in sizes:
        total_size += val

    if total_size == 0:
        raise ValueError("At least one element in `sizes` must be non-zero.")

    proportions = sizes.copy()

    r = total_size * random.random()
    print(f"Random: {r}")
    bin_start = 0.0
    bin_end = 0.0
    temp = 0
    for i in range(n_sizes):
        temp = float(proportions[i])
        if temp == 0: # skip zero-weighted entries
            continue
        bin_end += temp
        if r >= bin_start and r <= bin_end:
            print(f"Returning (id: {ls[i]}, i: {i})")
            return (ls[i], i)
        bin_start = bin_end

def choose_proportional_multithread(ls, sizes, n_sizes, max_workers=1):
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

    ranges = []
    range_size = floor(n_sizes/max_workers)
    rem = n_sizes % max_workers
    for i in range(max_workers):
        ranges.append((i*range_size, (i+1)*range_size-1))
    if n_sizes % max_workers != 0:
        ranges.append((max_workers*range_size, max_workers*range_size+rem-1))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        calc_prop = functools.partial(_calc_proportions, sizes=sizes, total_size=total_size)
        results = executor.map(calc_prop, ranges)

    proportions = []
    for r in list(results):
        proportions += r

    r = random.random() # btwn 0 and 1
    # print("after:")
    # print(proportions)
    # print(f"random: {r}")
    bin_start = 0.0
    bin_end = 0.0
    for i in range(n_sizes):
        bin_end += proportions[i]
        if r >= bin_start and r <= bin_end:
            # print(f"returning {ls[i]}")
            return ls[i]
        bin_start = bin_end

def _calc_proportions(index_range, sizes, total_size):
    proportions = []
    for i in range(index_range[0], index_range[1]+1):
        proportions.append(sizes[i] / total_size)
    
    return proportions

def _calc_proportions_inplace(ls, index_range, sizes, total_size):
    for i in range(index_range[0], index_range[1]+1):
        ls[i] = sizes[i] / total_size