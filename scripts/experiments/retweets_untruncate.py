from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sys
sys.path.append("scripts/")
import analysisfuncs as af

datasets, filenames = af.load_all_csvs("twitter_data/", pattern="retweeted_freq*JoeBiden*")

freqs = datasets[0]["freq"].to_list()

np.random.shuffle(freqs)

# These values from MLE fits (see ...fit_params.csv)
gamma = 1.8766848855003384
exp_fac = 4.3372623825939786e-06 # truncated power-law exponential factor

powerlaw = np.random.zipf(gamma, size=len(freqs))

freqs_untrunc = [x*np.exp(exp_fac*x) for x in freqs]

fig, ax = plt.subplots()
fig.set_size_inches(10, 6)

rank = np.linspace(1, len(freqs), len(freqs))
ax.scatter(rank, sorted(freqs_untrunc, reverse=True), label="Data untruncated")
ax.scatter(rank, sorted(powerlaw, reverse=True), label="Powerlaw")
ax.legend()
ax.set_ylabel("Frequency")
ax.set_xlabel("Rank")
ax.set_yscale("log")
ax.set_xscale("log")

fig.savefig(f"{filenames[0]}_untrunc.png")