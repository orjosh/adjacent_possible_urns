import powerlaw
import pandas as pd
import scipy
from matplotlib import pyplot as plt
import sys
sys.path.append("scripts/")
import analysisfuncs as af

datasets, filenames = af.load_all_csvs("twitter_data/", pattern="retweeted_freq*")

# Get the frequency counts as lists
freq_lists = {}
for i, df in enumerate(datasets):
    print(filenames[i])
    freqs = df["freq"].to_list()
    freq_lists[filenames[i]] = freqs
    print(scipy.stats.variation(freqs, ddof=1))

# Fit to the data
all_fits = {}
for i, freqs in enumerate(freq_lists.values()):
    name = filenames[i]
    this_fit = powerlaw.Fit(freqs, discrete=True, xmin_distance='V')
    all_fits[filenames[i]] = this_fit

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 6)

    scaling = this_fit.power_law.alpha
    #powercdf = this_fit.power_law.cdf()
    data, cdf = this_fit.cdf()
    for j, f in enumerate(data):
        cdf[j] = cdf[j]*(f**scaling)

    ccdf = [1 - x for x in cdf]
    #print(ccdf)

    ax.scatter(data, ccdf)

    #this_fit.plot_ccdf(ax=ax, label="Data", linestyle=':', marker='o')

    ax.set_title(name)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r"Retweet frequency $f$")
    ax.set_ylabel(r'$P(F\geq f)\times f^{\gamma}$')
    #ax.set_ylim([10**(-6), 3])
    ax.legend()

    plt.show()
    fig.savefig("figures/" + name + "_scaling_func.png")
    fig.clear()