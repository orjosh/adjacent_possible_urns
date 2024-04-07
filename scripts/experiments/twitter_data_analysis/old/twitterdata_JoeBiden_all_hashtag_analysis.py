import pandas as pd
import os, sys, codecs
import numpy as np
import matplotlib.cm as cm
import gzip
from matplotlib import pyplot as plt

model_fitting_path = "scripts/edugalt_TwitterHashtags_src/"
sys.path.append(model_fitting_path)

from modules_distributor import fit
from general import *

sys.path.append("scripts/")

from analysisfuncs import get_lognormal_dist

# Data Processing -------------------------------
csv_paths = ["twitter_data/hashtags_freq_biden_03_11_2021.csv", 
    "twitter_data/hashtags_freq_biden_JoeBiden_18_09_2022.csv",
    "twitter_data/hashtags_freq_election_midterm_09_11_2022.csv",
    "twitter_data/hashtags_freq_trump_realDonaldTrump_04_09_2020.csv"]

dataset_dicts = []
hashtag_lists = []
freq_datasets = []
for p in csv_paths:
    data = pd.read_csv(p)
    freqs = data["freq"].to_list()
    hashtags = data["hashtag"].to_list()

    freqs_norm = [x/freqs[0] for x in freqs]
    ranks = [x for x in range(len(freqs_norm))]
    freq_datasets.append((ranks, freqs_norm))

    hashtag_lists.append(set(hashtags))

    dataset_dicts.append(dict(zip(hashtags, freqs)))

common_hashtags = hashtag_lists[0]
for hset in hashtag_lists:
    common_hashtags = hset & common_hashtags

common_hashtags_total_freqs = []
for i,h in enumerate(common_hashtags):
    print(f"{i+1}/{len(common_hashtags)}")
    total_freq = 0
    for d in dataset_dicts:
        total_freq += d[h]

    common_hashtags_total_freqs.append(total_freq)

common_hashtags_total_freqs.sort(reverse=True)
common_hashtags_total_freqs_norm = [x/common_hashtags_total_freqs[0] for x in common_hashtags_total_freqs]
common_hashtags_rank = [x for x in range(len(common_hashtags_total_freqs_norm))]

print(f"first: {common_hashtags_total_freqs[0]}, norm first: {common_hashtags_total_freqs_norm[0]}")

#freq_datasets.append((common_hashtags_rank, common_hashtags_total_freqs_norm))

# model fitting ---------------------------------
do_model_fitting = False # this code kept here for replication
if do_model_fitting:
    models = ['simple', 'lognormal', 'double_2gammas'] # removed 'double_powerlaw' because apparently no convergence?
    nrep = 10 # number of repetitions

    for (rank, freqs) in freq_datasets:
        for model in models:
            res = fit(model = model, counts = freqs, nrep = nrep)
            print(f"\t{model}:\n\t\t{str(res)}")

lognorm_params = [ \
        [7.850283066655269, 3.179538132953169], \
        [7.4689002914209475, 3.183365468591284], \
        [8.274818367622572, 3.0008534923898025], \
        [7.041514129319809, 3.2468614046106] \
]

"""
        simple:
                ([1.1184549432496977], 11.020584024800087, 10)
        lognormal:
                ([7.850283066655269, 3.179538132953169], 10.44275666750449, 10)
        double_2gammas:
                ([0.7311049368383986, 1.4627800318291886, 6849.2741616619005], 10.472270558250234, 10)

        simple:
                ([1.123869465317834], 10.608353849216495, 10)
        lognormal:
                ([7.4689002914209475, 3.183365468591284], 10.06986657277687, 10)
        double_2gammas:
                ([0.7460278857500877, 1.4723833166435547, 5295.992819075884], 10.096039839828457, 10)

        simple:
                ([1.113050981732592], 11.469956354615856, 10)
        lognormal:
                ([8.274818367622572, 3.0008534923898025], 10.798212449916246, 10)
        double_2gammas:
                ([0.68819349405285, 1.4791142229178513, 8488.359711339472], 10.832243422764243, 10)

        simple:
                ([1.1302997691187777], 10.160706046759335, 10)
        lognormal:
                ([7.041514129319809, 3.2468614046106], 9.679496642733998, 10)
        double_2gammas:
                ([0.7469135522509062, 1.4332384026707286, 2715.875242224265], 9.708352062428926, 10)

        simple:
                ([1.1305892759313765], 10.14113283076236, 10)
        lognormal:
                ([7.091222568385581, 2.6197187049042685], 9.47709861900691, 10)
        double_2gammas:
                ([0.7323334128946011, 1.8526131169328224, 8093.699949346721], 9.475498781039985, 10)
"""

# Plotting --------------------------------------
fig, axs = plt.subplots(1,2)

for i,d in enumerate(freq_datasets):
    rank = d[0]
    freq = d[1]
    axs[0].scatter(rank, freq)


for i,d in enumerate(freq_datasets): # seperate because of plot legend
    fit_params = lognorm_params[i]
    lognorm_fit = get_lognormal_dist(len(rank), fit_params[0], fit_params[1])
    lognorm_fit_norm = [x/lognorm_fit[0] for x in lognorm_fit]
    axs[0].plot(lognorm_fit_norm, linestyle="dashed")

axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].set_title("Frequency vs. rank for hashtag datasets")
axs[0].legend(["Biden 03-11-21", "Biden 18-09-22", "Midterm 09-11-22", "Trump 04-09-20"])

common_rank = [x for x in range(len(common_hashtags_total_freqs))]
axs[1].scatter(common_rank, common_hashtags_total_freqs_norm)
axs[1].set_xscale('log')
axs[1].set_yscale('log')
axs[1].set_title("Total frequency of common hashtags over all datasets vs. rank")

common_hashtag_lognorm_params = [7.091222568385581, 2.6197187049042685]
common_hashtag_2gammas_params = [0.7323334128946011, 1.8526131169328224, 8093.699949346721]

common_hashtag_lognorm_fit = get_lognormal_dist(len(common_rank), \
        common_hashtag_lognorm_params[0], common_hashtag_lognorm_params[1])
common_lognorm_fit_norm = [x/common_hashtag_lognorm_fit[0] for x in common_hashtag_lognorm_fit]
axs[1].plot(common_lognorm_fit_norm, linestyle='dashed')

fig.supxlabel("Rank")
fig.supylabel("Frequency")

plt.show()