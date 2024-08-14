import numpy as np
import scipy
from matplotlib import pyplot as plt
import sys
sys.path.append("scripts/")
import analysisfuncs as af

datasets, filenames = af.load_all_csvs("twitter_data/", pattern="retweeted_freq*")

skews = []
kurts = []

for i, df in enumerate(datasets):
    print(filenames[i])
    freqs = df["freq"].to_list()

    skews.append(scipy.stats.skew(freqs))
    kurts.append(scipy.stats.kurtosis(freqs))
    print(kurts)

fig, axs = plt.subplots(1,2)

axs[0].boxplot(skews)
axs[0].set_ylabel("Skewness")

axs[1].boxplot(kurts)
axs[1].set_ylabel("Kurtosis")

fig.suptitle("Moments for retweet frequency data")
fig.set_size_inches(10,6)
fig.tight_layout()
fig.savefig("retweets_skew_kurt.png")