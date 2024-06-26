import numpy as np
import random
from matplotlib import pyplot as plt
import sys
sys.path.append("scripts/")

# Degree distribution p(k) is normally distributed. The probability that a user of degree k
# picks a specific connection e as one of its events is Zipfian.

N_USERS = 1000
NORM_MEAN = 101
NORM_STDV = 25

N_EVENTS = 1000
POW_SCALING = 1.9

rng = np.random.default_rng(seed=1234)

degs = rng.normal(loc=NORM_MEAN, scale=NORM_STDV, size=N_USERS)

for d in degs:
    freqs = {}
    #print(round(d))
    edge_probs = sorted(rng.zipf(a=POW_SCALING, size=round(d)))

    # init all edge freqs to 0
    for i in range(round(d)):
        freqs[i] = 0
    
    for i in range(N_EVENTS):
        r = random.random()
        last_p = edge_probs[0]
        for j, p in enumerate(edge_probs):
            if r > last_p and r <= p:
                freqs[j] += 1


fig, ax = plt.subplots()
ax.hist(degs)
fig.savefig("normal.png")