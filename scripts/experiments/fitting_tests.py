import sys
sys.path.append("scripts/")
import powerlaw_fitting
from matplotlib import pyplot as plt
import pandas as pd
import pickle
import math

df = pd.read_csv("twitter_data/hashtags_freq_biden_03_11_2021.csv")
freqs = list(df["freq"])

with open("twitter_data/twitterdata_all_retweet_scalingfits.pickle", 'rb') as f:
    fit_params = pickle.load(f)

mu = fit_params[3][1][0][0]
sigma = fit_params[3][1][0][1]

ecdf = list(powerlaw_fitting.get_empirical_cdf(freqs))

cdf_lognormal = [powerlaw_fitting.cdf_lognormal(x, mu, sigma) for x in range(1,len(ecdf)+1)]

cdf_pow = [powerlaw_fitting.cdf_power(x, 1, 1.11) for x in range(1, len(ecdf)+1)]

fig, ax = plt.subplots()
ax.step(freqs, ecdf)
#ax.scatter(freqs, cdf_lognormal)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel('P(x)')
ax.set_xlabel('Retweet count')

plt.show()
