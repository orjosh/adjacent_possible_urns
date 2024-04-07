import pandas as pd
import os, sys, codecs
import gzip
import pickle
import math
from matplotlib import pyplot as plt
from pathlib import Path
import scipy
import powerlaw

model_fitting_path = "scripts/edugalt_TwitterHashtags_src/"
sys.path.append(model_fitting_path)

from modules_distributor import fit
from modules_fitting_lognormal import pdf_lognormal_disc
from modules_fitting_pow import pdf_power_disc
from general import *

sys.path.append("scripts/")

from analysisfuncs import get_lognormal_dist, get_doublepower_2gamma_dist, get_powerlaw_dist, pdf_lognormal, pdf_powerzipf, loglikelihoodratio

# Data Processing -------------------------------
csv_path_prefix = "twitter_data/retweeted_freq"

filenames = []
dataset_dicts = []
users_lists = []
freq_datasets = []
for p in Path(".").glob(csv_path_prefix + "*"):
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
    print(len(freqs_cutoff))
    print(len(ranks_cutoff))

    lower_freq_datasets.append((ranks_cutoff, freqs_cutoff))

# model fitting ---------------------------------
pickle_filename = "twitterdata_all_retweet_lowerscalingfits.pickle"
do_model_fitting_eduardo = False # this code kept here for replication
if do_model_fitting_eduardo:
    fits = []
    models = ['simple', 'lognormal'] # re add 'double_2gammas'
    nrep = 10 # number of repetitions

    for (rank, freqs) in lower_freq_datasets:
        this_dataset_fits = []
        for model in models:
            res = fit(model = model, counts = freqs, nrep = nrep)
            this_dataset_fits.append(res)
            print(f"\t{model}:\n\t\t{str(res)}")

        fits.append(this_dataset_fits)
        with open("twitter_data/" + pickle_filename, 'wb') as f:
            pickle.dump(fits, f)

with open("twitter_data/" + pickle_filename, 'rb') as f:
    fits = pickle.load(f)

# for i,d in enumerate(freq_datasets):
#     ranks = d[0]
#     freqs = d[1]
#     n = len(freqs)
    
#     power_fit = fits[i][0]
#     lognorm_fit = fits[i][1]

#     power_logpdf = []
#     lognorm_logpdf = []
#     for j in range(n):
#         power_val = pdf_powerzipf(freqs[j], power_fit[0][0])
#         power_logpdf.append(math.log(power_val))

#         lognorm_val = pdf_lognormal(freqs[j], lognorm_fit[0][0], lognorm_fit[0][1])
#         lognorm_logpdf.append(math.log(lognorm_val))
#         print(f"Dataset {i+1}/{len(freq_datasets)}, j = {j+1}/{n}", end='\r')

#     print(loglikelihoodratio(power_logpdf, lognorm_logpdf))
#     print(f"Ratio from fit is {power_fit[1]/lognorm_fit[1]}")

#     plt.plot(power_logpdf)
#     plt.show()


# Plotting --------------------------------------
with open("twitter_data/" + pickle_filename, 'rb') as f:
    fits = pickle.load(f)

powerlaw_params = []
lognorm_params = []
for f in fits:
    powerlaw_params.append(f[0][0])
    lognorm_params.append(f[1][0])

for i,d in enumerate(lower_freq_datasets):
    print(f"{filenames[i]} gamma: {powerlaw_params[i][0]:.2f}")

# for i,d in enumerate(lower_freq_datasets):
#     print(f"Plotting {filenames[i]}")
#     fig, axs = plt.subplots(1,2)

#     rank = d[0]
#     freq = d[1]

#     for ax in axs:
#         ax.set_xscale('log')
#         ax.set_xlim([10**2, (1/2)*10**7])
#         ax.set_xlabel("Rank")
#         ax.set_ylabel("Frequency")

#     gamma = powerlaw_params[i][0]
#     powerlaw_fit = get_powerlaw_dist(len(rank), gamma)
#     const_fac = freq[0] / powerlaw_fit[0] # so that powerlaw[cutoff] = freq[cutoff]
#     powerlaw_fit_norm = [const_fac*x for x in powerlaw_fit]

#     mu = lognorm_params[i][0]
#     sigma = lognorm_params[i][1]
#     lognorm_fit = get_lognormal_dist(len(rank), mu, sigma)
#     const_fac = freq[0] / lognorm_fit[0]
#     lognorm_fit_norm = [const_fac*x for x in lognorm_fit]

#     axs[0].set_yscale('log')
#     axs[0].set_ylim([(1/2)*10**(-7), 1])
#     axs[0].scatter(rank, freq)
#     axs[0].plot(powerlaw_fit, linestyle="dashed")
#     axs[0].plot(lognorm_fit_norm, linestyle='dashed')
#     axs[0].legend(["Data", \
#         r"$r^{-\gamma}$, $\gamma$ = " + f"{gamma:.2f}", \
#             "Lognorm " + r"$\mu$=" + f"{mu:.2f}, " + r"$\sigma=$" + f"{sigma:.2f}"])
#     axs[0].set_title(filenames[i] + " data and fit")

#     logfreq = [math.log10(x) for x in freq]
#     axs[1].scatter(rank, logfreq)
#     axs[1].set_title("log(freq)")

#     plt.savefig(filenames[i] + "_freqrank_tail.png")
#     fig.clear()