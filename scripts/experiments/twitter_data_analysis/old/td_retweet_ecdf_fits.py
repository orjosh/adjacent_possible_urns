import sys
sys.path.append("scripts/")
import pandas as pd
import pickle
from pathlib import Path
from matplotlib import pyplot as plt
import analysisfuncs as af
import powerlaw_fitting
import powerlaw

FIT_REPS = 10
FIT_MODELS = ['simple', 'lognormal', 'double_2gammas']
FIT_PICKLE_FILENAME = "twitter_data/ecdf_fits"

# Load all retweet frequency datasets
datasets, filenames = af.load_all_csvs("twitter_data/", pattern="retweeted_freq_*")

# Get the frequency counts as lists
ecdf_lists = {}
for i, df in enumerate(datasets):
    freqs = df["freq"].to_list()
    ecdf = list(powerlaw_fitting.get_empirical_cdf(freqs))
    ecdf_lists[filenames[i]] = ecdf
    
# Fit models to data and pickle the results
af.fit_and_pickle(list(ecdf_lists.values()), list(ecdf_lists.keys()), FIT_MODELS, FIT_REPS, FIT_PICKLE_FILENAME)

# Alternative fitting
fit = powerlaw.Fit(datasets[0]["freq"].to_list())
x_min = int(fit.power_law.xmin)
scaling = fit.power_law.alpha

fig, ax = plt.subplots()
fit.plot_ccdf(ax, label="Data")
fit.power_law.plot_ccdf(ax=ax, linestyle="--", label="Power-law")
fit.power_law.plot_ccdf(ax=ax, linestyle=":", label="Lognormal")

ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()
plt.show()

# Load fits and plot the first one
# with open(FIT_PICKLE_FILENAME + ".pickle", 'rb') as f:
#     model_fits = pickle.load(f)

# i = 2

# freqs = datasets[i]["freq"].to_list()
# x_axis = [x for x in range(len(freqs))]

# models = model_fits[filenames[i]]
# dp_fit_params = models['double_2gammas']['parameters']
# gamma1 = float(dp_fit_params[0])
# gamma2 = float(dp_fit_params[1])
# kink = int(dp_fit_params[2])
# print(kink)
# double_power = af.get_doublepower_2gamma_dist(len(freqs), gamma1, gamma2, kink)

# fig, ax = plt.subplots()
# ax.step(freqs, ecdf_lists[filenames[i]])
# ax.plot(x_axis, double_power)
# ax.set_yscale('log')
# ax.set_xscale('log')
# plt.show()