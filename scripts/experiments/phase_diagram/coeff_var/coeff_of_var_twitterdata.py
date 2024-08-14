import numpy as np
import scipy
from matplotlib import pyplot as plt
import sys
sys.path.append("scripts/")
import analysisfuncs as af

datasets, filenames = af.load_all_csvs("twitter_data/", pattern="retweeted_freq*")

coeffs = []
for i, df in enumerate(datasets):
    print(filenames[i])
    freqs = df["freq"].to_list()
    print(scipy.stats.variation(freqs, ddof=1))
    coeffs.append(scipy.stats.variation(freqs, ddof=1))

fig, ax = plt.subplots()
ax.boxplot(coeffs)
ax.set_ylabel("Coefficient of variation")
ax.set_title("COVs for retweet frequency data")
fig.set_size_inches(10,6)
fig.savefig("retweets_coef_var.png")