import pandas as pd
import sys
import numpy as np
import scipy
from matplotlib import pyplot as plt
from pathlib import Path

model_fitting_path = "scripts/edugalt_TwitterHashtags_src/"
sys.path.append(model_fitting_path)

from modules_distributor import fit
from general import *

sys.path.append("scripts/")

from analysisfuncs import get_lognormal_dist, get_doublepower_2gamma_dist, get_powerlaw_dist

# Data Processing -------------------------------
csv_path_prefix = "twitter_data/retweeted_freq"

filenames = []
dataset_dicts = []
users_lists = []
freq_datasets = []
for p in Path(".").glob(csv_path_prefix + "*trump*"):
    filename = str(p).split(csv_path_prefix + "_")[1]
    filename = filename.split(".")[0]
    filenames.append(filename)
    print(filename)

    data = pd.read_csv(str(p))
    freqs = data["freq"].to_list()
    users = data["user"].to_list()

    freqs_norm = [x/freqs[0] for x in freqs]
    ranks = [x for x in range(len(freqs_norm))]
    freq_datasets.append((ranks, freqs_norm))

    users_lists.append(set(users))

    dataset_dicts.append(dict(zip(users, freqs)))

lower_freq_datasets = [] # want to look at only ranks >= some number
cutoff = int(500)
for d in freq_datasets:
    ranks = d[0]
    freqs = d[1]
    
    freqs_cutoff = freqs[cutoff:len(freqs)+1] # note: is normalized
    ranks_cutoff = ranks[cutoff:len(ranks)+1]

    lower_freq_datasets.append((ranks_cutoff, freqs_cutoff))

ecdfs = []
for d in freq_datasets:
    freqs = d[1]
    ecdf_func_obj = scipy.stats.ecdf(freqs).cdf
    ecdfs.append(ecdf_func_obj)

print(ecdfs)

for i in range(len(ecdfs)-1):
    for j in range(i+1, len(ecdfs)):
        print(f"{filenames[i]} vs. {filenames[j]}")

        leni = len(freq_datasets[i][1])
        lenj = len(freq_datasets[j][1])
        
        if lenj > leni:
            largest_len = lenj
        else:
            largest_len = leni

        max_dist = 0
        for k in range(largest_len):
            r = largest_len - k
            r_i = ecdfs[i].evaluate(r)
            r_j = ecdfs[j].evaluate(r)
            print(f"{r_i}, {r_j}")

            dist = abs(r_i - r_j)
            if dist > max_dist:
                max_dist = dist

        print(f"\tD = {max_dist}")