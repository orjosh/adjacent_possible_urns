import sys
import numpy as np
import random
import powerlaw
from matplotlib import pyplot as plt
sys.path.append("./scripts")
from analysisfuncs import power_law_distributed_rng

# This script runs a simple toy model to observe any possible finite size sampling effects that
# may be present in the twitter data

N_USERS = 100
N_EVENTS = 100000
EXPONENT = 1.90 # from data
X_MIN = 100 # from data

degrees = power_law_distributed_rng(N_USERS, X_MIN, EXPONENT)
for i,k in enumerate(degrees):
    degrees[i] = round(k)

all_freqs = []
for deg in degrees:
    freqs = {}
    for i in range(N_EVENTS):
        r = random.randint(1, deg)
        if r in freqs:
            freqs[r] += 1
        else:
            freqs[r] = 1
    
    freqs_vals = list(freqs.values())
    for f in freqs_vals:
        all_freqs.append(f)

#print(sorted(all_freqs, reverse=True))

# Assume no cross-over between users (i.e. none of them share any connections in common)

fig, ax = plt.subplots()
ax.scatter(sorted(all_freqs, reverse=True), range(len(all_freqs)))
ax.set_ylabel('Frequency')
ax.set_xlabel('Rank')
ax.set_xscale('log')
ax.set_yscale('log')
fig.savefig("figures/toy_fss.png")
fig.clear()

fig, ax = plt.subplots()

fit = powerlaw.Fit(sorted(all_freqs, reverse=True))
fit.plot_ccdf(ax=ax, linestyle='--', marker='o', label="Data")
fit.power_law.plot_ccdf(ax=ax, linestyle='--', linewidth='2', label=r"Power-law ($\gamma$=" + f"{fit.power_law.alpha:.2f})")
fit.truncated_power_law.plot_ccdf(ax=ax, linestyle="--", linewidth=2, label="Truncated power-law")

#ax.set_xscale('log')
print(fit.xmin)
ax.set_yscale('log')
ax.set_ylabel(r"$P(X\geq x)$")
ax.set_xlabel("Frequency")
ax.legend()
fig.savefig("figures/toy_ffs_fits.png")

#print(all_freqs)
#print(sum(freqs))
# To self: use power law rng to generate degree distribution for users, then prescribe each
# edge or link an (at first) equal probability of being chosen. This is a "retweet", and the
# other user is not amongst the 10 generated -- they're invisible. Now sample from each
# of the 10 users say 100 'events' i.e. retweets, then see what you get from plotting
# the frequency-rank of retweets.