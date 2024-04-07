import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
import pickle
import sys
sys.path.append("scripts/")
import analysisfuncs as af

# Define constants
RANK_RANGE_LOWER = 10
RANK_RANGE_HIGHER = 1000

FIT_REPS = 10
FIT_MODELS = ['simple', 'lognormal']
FIT_PICKLE_FILENAME = "twitter_data/retweets_upperfreqsfit"

# Load all retweet frequency datasets
datasets, filenames = af.load_all_csvs("twitter_data/", pattern="retweeted_freq_*")

# Get the frequency counts as lists
freqs_lists = {}
for i, df in enumerate(datasets):
    freqs = df["freq"].to_list()
    freq_subset = freqs[RANK_RANGE_LOWER:RANK_RANGE_HIGHER]
    freqs_lists[filenames[i]] = freq_subset
    
# Fit models to data and pickle the results
af.fit_and_pickle(list(freqs_lists.values()), list(freqs_lists.keys()), FIT_MODELS, FIT_REPS, FIT_PICKLE_FILENAME)

# Load fit results and plot
fig, ax = plt.subplots(1,1)
fig.set_size_inches(10,6)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel('Retweet count')
ax.set_xlabel('Rank')

with open(FIT_PICKLE_FILENAME + ".pickle", 'rb') as f:
    fits = pickle.load(f)

# Look at just one for now
data_fit = fits[filenames[0]]
lognorm_params = data_fit["lognormal"]["parameters"]
lognorm_curve = af.get_lognormal_dist(RANK_RANGE_HIGHER - RANK_RANGE_LOWER, lognorm_params[0], lognorm_params[1])

# Find right vertical offset for the fit, which should coincide with the first point of the data
# We want freqs[RANK_RANGE_LOWER] == fit_curve[RANK_RANGE_LOWER]
percent_diff = freqs_lists[filenames[0]][RANK_RANGE_LOWER]/lognorm_curve[0]
lognorm_curve_aligned = [x*percent_diff for x in lognorm_curve]

# Zipfian fit
powerlaw_params = data_fit["simple"]["parameters"]
powerlaw_curve = af.get_powerlaw_dist((RANK_RANGE_LOWER, RANK_RANGE_HIGHER), powerlaw_params[0])
percent_diff = freqs_lists[filenames[0]][0]/powerlaw_curve[0]
powerlaw_curve_scaled = [x*percent_diff for x in powerlaw_curve]


rank = [x + RANK_RANGE_LOWER for x in range(RANK_RANGE_HIGHER-RANK_RANGE_LOWER)]
ax.scatter(rank, freqs_lists[filenames[0]])
ax.scatter(rank, powerlaw_curve_scaled)
ax.scatter(rank, lognorm_curve_aligned)
plt.show()
    