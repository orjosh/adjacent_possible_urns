import numpy as np
import random
import concurrent.futures
import functools
import pandas as pd
import networkx as nx
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

    return exponent, err

def discrete_power_mle_approx(data, x_min_index):
    '''
    Expects `data` to be in increasing order. Returns an estimate of the scaling parameter.
    '''
    # Clauset et al
    x_min = data[x_min_index]
    data_rev = data[x_min_index:].copy()
    data_rev.reverse()
    return continuous_power_mle(data_rev, x_min-0.5)

def choose_proportional_dict(d: Dict, total_size):
    '''
    '''
    if total_size == 0:
        print(d)
        raise ValueError("At least one element in `sizes` must be non-zero.")

    r = total_size * random.random()
    # print(f"r = {r}")
    bin_start = 0.0
    bin_end = 0.0
    for k,v in d.items():
        bin_end += v
        if bin_start <= r <= bin_end:
            # print(f"returning {k}")
            return k
        bin_start = bin_end

# Network analysis --------------------------------------------------------------------------------
def csv_to_graph(csv_path):
    """
    Takes a csv file representing a network/graph and returns a NetworkX Graph
    representation of it. The csv file must contain the two columns 'caller'
    and 'receiver' whose rows depict an edge (c, r).
    """

    df = pd.read_csv(csv_path)
    callers = df['caller']
    receivers = df['receiver']

    edges = []
    for i,c in enumerate(callers):
        r = receivers.iloc[i]
        edges.append((c,r))

    G = nx.Graph()
    G.add_edges_from(edges)

    return G
