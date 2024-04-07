import numpy as np
import random
import pandas as pd
import pickle
import networkx as nx
import powerlaw
from os.path import exists
from pathlib import Path
from math import log, sqrt, exp
from scipy.special import zeta, erfc
from typing import Dict
from edugalt_TwitterHashtags_src.modules_distributor import fit

def fit_and_pickle(data_lists, data_names, pickle_filename):
    all_fits = {}

    if not pickle_filename.endswith(".pickle"):
        pickle_filename = pickle_filename + ".pickle"

    if exists(pickle_filename):
        ans = input(f"Warning! \'{pickle_filename}\' already exists! Type \'OVERWRITE\' to proceed anyway.")
        if ans != "OVERWRITE":
            return

    for i, data in enumerate(data_lists):
        this_fit = powerlaw.Fit(data)
        all_fits[data_names[i]] = this_fit

    with open(pickle_filename, 'wb') as f:
        pickle.dump(all_fits, f)

def fit_and_pickle_eduardo(data_lists, data_names, models_ls, nreps, pickle_filename):
    all_fits = {}

    if not pickle_filename.endswith(".pickle"):
        pickle_filename = pickle_filename + ".pickle"

    if exists(pickle_filename):
        ans = input(f"Warning! \'{pickle_filename}\' already exists! Type \'OVERWRITE\' to proceed anyway.")
        if ans != "OVERWRITE":
            return

    for i, ls in enumerate(data_lists):
        this_dataset_fits = {}
        for j, model in enumerate(models_ls):
            print(f"Fitting (dataset {i+1}/{len(data_lists)}, model ({j+1}/{len(models_ls)})", end='\r')
            res = fit(model = model, counts = ls, nrep = nreps)
            this_model = {"parameters": res[0], "-loglikelihood": res[1]}
            this_dataset_fits[model] = this_model

        all_fits[data_names[i]] = this_dataset_fits
    
    with open(pickle_filename, 'wb') as f:
        pickle.dump(all_fits, f)
            

def load_all_csvs(folder, pattern=None):
    if not str(folder).endswith("/"):
        folder = folder + "/"

    if pattern:
        if not str(pattern).endswith(".csv"):
            path_pattern = folder + pattern + ".csv"
        else:
            path_pattern = folder + pattern
    else:
        path_pattern = folder + "*.csv"

    dataframes = []
    names = []
    for p in Path(".").glob(folder + pattern):
        filename = str(p).split(folder)[1]
        filename = filename.split(".")[0]
        names.append(filename)

        data = pd.read_csv(str(p))
        dataframes.append(data)

    return dataframes, names

def loglikelihoodratio(logs1, logs2):
    dist1_mean = sum(logs1)/len(logs1)
    dist2_mean = sum(logs2)/len(logs2)

    ratio = 0
    variance = 0
    n = len(logs1)
    for i in range(n):
        ratio += logs1[i] - logs2[i]
        variance += (logs1[i]-logs2[i]-dist1_mean+dist2_mean)**2
    
    variance = variance/n

    p_val = erfc(ratio/sqrt(2*n*variance))

    return ratio, p_val

def discrete_powerlaw(x, x_min, gamma):
    if x <= 0 or x < x_min:
        return ValueError("x must be > 0 and >= x_min")

    return (x**(-gamma))/zeta(gamma, q=x_min)

def pdf_lognormal(x, mu, sigma):
    if x <= 0:
        return ValueError("x must be > 0")

    exponential = exp(-((log(x)-mu)**2)/(2*sigma**2))
    fraction = 1 / (sigma*x*sqrt(2*np.pi))

    return fraction * exponential

def get_lognormal_dist(max_rank, m, s):
    """
    Generates and returns the log-normal curve (as a list of y values) for a distribution
    with the given parameters.

    Params:     -   `max_rank` is the number of data points to generate, i.e. values will
                    be calculated for ranks 1,2, ..., `max_rank`.
                - `m` and `s`: See https://arxiv.org/pdf/2004.12707.pdf Table 1
    """
    res = []
    for r in range(1, max_rank+1):
        y = (1/r)*exp( -(1/2)*pow((log(r)-m)/s, 2) )
        res.append(y)

    return res

def get_powerlaw_dist(rank_range, scaling):
    """
    Generates and returns the power law curve (as a list of y values)
    for a distribution with the given parameters.

    Params:         -   `rank_range` = (lower, upper) is the range of values to calculate
                        for, non-inclusive of the upper bound as with python's range function.
                    -   `scaling` is the power law exponent
    """
    res = []
    for r in range(rank_range[0], rank_range[1]):
        y = pow(r, -scaling)
        res.append(y)

    return res

def get_doublepower_2gamma_dist(max_rank, gamma1, gamma2, rank_switch):
    """
    Generates and returns the 2-gamma double power law curve (as a list of y values)
    for a distribution with the given parameters.

    Params:         -   `max_rank` is the number of data points to generate
                    -   `gamma1` and `gamma2` are the two scaling components, where
                        `gamma1` is for rank r <= `rank_switch`
                    -   `rank_switch` is the rank at which the scaling changes 
    """
    res = []

    for r in range(1, rank_switch+2): # +2 so inclusive of rank_switch
        y = pow(r, -gamma1)
        res.append(y)
    
    for r in range(rank_switch+2, max_rank+1):
        y = pow(rank_switch, gamma2-gamma1) * pow(r, -gamma2)
        res.append(y)

    return res


def power_law_distributed_rng(n, x_min, scale, seed=None):
    # This algorithm is taken from Appendix D in Clauset et al's"Power Law Distributions
    # in Empirical Data", SIAM Review, Vol 51 (4), pp. 661-703
    rng = np.random.default_rng(seed)
    sample = rng.random((n,1))
    for i in range(n):
        r = sample[i]
        sample[i] = x_min * (1-r)**(-1/(scale-1))
    return sorted(sample)

def continuous_power_mle(data, x_min): # Deprecated, use Eduardo et. al's code
    # Clauset et al
    n = len(data)

    sumfactor = 0
    for i in range(n):
        sumfactor += log(data[i]/x_min)
    
    exponent = 1 + n/sumfactor
    err = (exponent - 1)/sqrt(n)

    return exponent, err

def discrete_power_mle_approx(data, x_min_index): # Deprecated, use Eduardo et. al's code
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
