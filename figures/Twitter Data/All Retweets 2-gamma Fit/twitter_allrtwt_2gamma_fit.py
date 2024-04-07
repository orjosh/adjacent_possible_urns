 import pandas as pd
import os, sys, codecs
import numpy as np
import matplotlib.cm as cm
import gzip
import pickle
from matplotlib import pyplot as plt
from pathlib import Path

model_fitting_path = "scripts/edugalt_TwitterHashtags_src/"
sys.path.append(model_fitting_path)

from modules_distributor import fit
from general import *

sys.path.append("scripts/")

from analysisfuncs import get_lognormal_dist, get_doublepower_2gamma_dist

# Data Processing -------------------------------
csv_path_prefix = "twitter_data/retweeted_freq"

dataset_dicts = []
users_lists = []
freq_datasets = []
for p in Path(".").glob(csv_path_prefix + "*"):
    print(p)
    data = pd.read_csv(str(p))
    freqs = data["freq"].to_list()
    users = data["user"].to_list()

    freqs_norm = [x/freqs[0] for x in freqs]
    ranks = [x for x in range(len(freqs_norm))]
    freq_datasets.append((ranks, freqs_norm))

    users_lists.append(set(users))

    dataset_dicts.append(dict(zip(users, freqs)))

common_users = users_lists[0]
for hset in users_lists:
    common_users = hset & common_users

common_users_total_freqs = []
for i,h in enumerate(common_users):
    print(f"{i+1}/{len(common_users)}")
    total_freq = 0
    for d in dataset_dicts:
        total_freq += d[h]

    common_users_total_freqs.append(total_freq)

common_users_total_freqs.sort(reverse=True)
common_users_total_freqs_norm = [x/common_users_total_freqs[0] for x in common_users_total_freqs]
common_users_rank = [x for x in range(len(common_users_total_freqs_norm))]

# model fitting ---------------------------------
pickle_filename = "twitterdata_all_retweet_scalingfits.pickle"
do_model_fitting = False # this code kept here for replication
if do_model_fitting:
    fits = []
    models = ['simple', 'lognormal', 'double_2gammas'] # removed 'double_powerlaw' because apparently no convergence?
    nrep = 10 # number of repetitions

    for (rank, freqs) in freq_datasets:
        this_dataset_fits = []
        for model in models:
            res = fit(model = model, counts = freqs, nrep = nrep)
            this_dataset_fits.append(res)
            print(f"\t{model}:\n\t\t{str(res)}")

        fits.append(this_dataset_fits)
        with open("twitter_data/" + pickle_filename, 'wb') as f:
            pickle.dump(fits, f)

# Plotting --------------------------------------
with open("twitter_data/" + pickle_filename, 'rb') as f:
    fits = pickle.load(f) # list of 3 models in order of simple power law, lognormal, 2 gamma powerlaw

double_2gammas_params = []
for f in fits:
    double_2gammas_params.append(f[2][0])

fig, axs = plt.subplots(1,1)

for i,d in enumerate(freq_datasets):
    rank = d[0]
    freq = d[1]
    axs.scatter(rank, freq)

for i,d in enumerate(freq_datasets): # seperate because of plot legend
    rank = d[0]

    fit_params = double_2gammas_params[i]
    gamma1 = fit_params[0]
    gamma2 = fit_params[1]
    rank_switch = int(fit_params[2])

    doublepower_fit = get_doublepower_2gamma_dist(len(rank), gamma1, gamma2, rank_switch)
    doublepower_fit_norm = [x/doublepower_fit[0] for x in doublepower_fit]
    axs.plot(doublepower_fit_norm, linestyle="dashed")

axs.set_xscale('log')
axs.set_yscale('log')

fig.supxlabel("Rank")
fig.supylabel("Frequency")

plt.show()
