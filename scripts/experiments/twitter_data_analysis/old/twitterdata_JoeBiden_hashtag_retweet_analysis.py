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
hashtag_path = "twitter_data/hashtags_freq_biden_JoeBiden_18_09_2022.csv"
retweet_path = "twitter_data/retweeted_freq_biden_JoeBiden_18_09_2022.csv"

hashtag_data = pd.read_csv(hashtag_path)
hashtag_freqs = hashtag_data.iloc[:,0]
hashtag_rank = [x for x in range(len(hashtag_freqs))]
hashtag_freqs_norm = [x/hashtag_freqs[0] for x in hashtag_freqs]

retweet_data = pd.read_csv(retweet_path)
retweet_freqs = retweet_data.iloc[:,0]
retweet_rank = [x for x in range(len(retweet_freqs))]
retweet_freqs_norm = [x/retweet_freqs[0] for x in retweet_freqs]

# model fitting
do_model_fitting = False # this code kept here for replication
if do_model_fitting:
    models = ['simple', 'lognormal', 'double_2gammas'] # removed 'double_powerlaw' because apparently no convergence?
    nrep = 10 # number of repetitions

    freq_datasets = [hashtag_freqs, retweet_freqs]

    for freqs in freq_datasets:
        for model in models:
            res = fit(model = model, counts = freqs, nrep = nrep)
            print(f"\t{model}:\n\t\t{str(res)}")

hashtag_params = {
    "simple": 1.122604156284941,
    "lognormal": [7.564783369269384, 3.098977756138056],
    "double_2gammas": [0.7207917839456419, 1.46230005084918, 4575.1514784141455]
}

hashtag_likelihoods = { # actually -log L, so smaller is better
    "simple": 10.701826070992553,
    "lognormal": 10.131711924532679,
    "double_2gammas": 10.165177006828797
}

retweet_params = {
    "simple": 1.1194426998914966,
    "lognormal": [7.757928713765043, 3.3557247356484297],
    "double_2gammas": [0.7968770691977869, 1.4649668156884958, 11594.60223391064]
}

retweet_likelihoods = { # -log L so smaller is better
    "simple": 10.942894265991645,
    "lognormal": 10.418007974990289,
    "double_2gammas": 10.470742854792398
}

hashtag_lognormal = get_lognormal_dist(len(hashtag_rank), hashtag_params["lognormal"][0], hashtag_params["lognormal"][1])
hashtag_lognormal_norm = [x/hashtag_lognormal[0] for x in hashtag_lognormal]

retweet_lognormal = get_lognormal_dist(len(retweet_rank), retweet_params["lognormal"][0], retweet_params["lognormal"][1])
retweet_lognormal_norm = [x/retweet_lognormal[0] for x in retweet_lognormal]

fit_fractional_diff = []
for i, H in enumerate(hashtag_lognormal_norm):
    R = retweet_lognormal_norm[i]
    fit_fractional_diff.append(abs(1-H/R)) 

# Plotting --------------------------------------
fig, axs = plt.subplots(1,3)

axs[0].scatter(hashtag_rank, hashtag_freqs_norm)
axs[0].plot(hashtag_lognormal_norm, linestyle='dashed')
axs[0].set_title("Hashtags")

axs[1].scatter(retweet_rank, retweet_freqs_norm)
axs[1].plot(retweet_lognormal_norm, linestyle='dashed')
axs[1].set_title("Retweets")

axs[2].plot(fit_fractional_diff)
axs[2].set_title(r"$|1-\frac{lognorm(H)}{lognorm(R)}$|")
axs[2].set_ylabel("Frequency difference")
axs[2].set_xscale('log')

# common settings
fig.suptitle("Frequency-Rank plots for the JoeBiden_18_09_2022 datasets")
fig.supxlabel("Rank")
fig.supylabel("Frequency")

for ax in [axs[0], axs[1]]:
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(["Data", "Log-normal fit"])
    #ax.set_ylim([0.5,10**6])
    #ax.set_xlim([0.5,10**7])

plt.show()