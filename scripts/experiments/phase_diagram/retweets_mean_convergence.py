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
mu = -35.055238419940984
sigma = 6.810924636886264
exp_fac = 4.3372623825939786e-06 # truncated power-law exponential factor

powerlaw = np.random.zipf(gamma, size=len(freqs))
lognorm = np.random.lognormal(mean=mu, sigma=sigma, size=len(freqs))
freqs_untrunc = [x*np.exp(exp_fac*x) for x in freqs]

running_avg_pow = [1]
running_avg_logn = [1]
running_avg_data = [1] # add a fake entry of 1 to avoid division by zero
running_avg_untrunc = [1]

for i,f in enumerate(freqs):
    avg_data = (running_avg_data[i] * i + f) / (i+1)
    running_avg_data.append(avg_data)

    avg_pow = (running_avg_pow[i] * i + powerlaw[i]) / (i+1)
    running_avg_pow.append(avg_pow)

    # avg_logn = (running_avg_logn[i] * i + lognorm[i]) / (i+1)
    # running_avg_logn.append(avg_logn)

    avg_untrunc = (running_avg_untrunc[i] * i + freqs_untrunc[i]) / (i+1)
    running_avg_untrunc.append(avg_untrunc)

    print(f"{i}/{len(freqs)}")

fig, ax = plt.subplots()
fig.set_size_inches(10, 6)

x_axis = np.linspace(1, len(running_avg_data), len(running_avg_data))
ax.plot(running_avg_data, label="Data (shuffled)", marker=".", linestyle="--")
ax.plot(x_axis, running_avg_untrunc, label="Data untruncated (shuffled)", marker=".", linestyle="--")
ax.plot(x_axis, running_avg_pow, label=r"Power-law $\gamma$="+f"{gamma:1.3g}", marker=".", linestyle="--")
# ax.plot(running_avg_logn, label=r"Lognormal $\mu$="+f"{mu}, " + r"$\sigma$="+f"{sigma}")
ax.legend()
ax.set_ylabel("Average frequency")
ax.set_xlabel("Sample size")
#ax.set_yscale("log")
ax.set_xscale("log")

fig.savefig(f"{filenames[0]}_running_avg.png")