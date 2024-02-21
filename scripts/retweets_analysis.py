import pandas as pd
import functools
import scipy.optimize as optimize
import numpy as np
from matplotlib import pyplot as plt

def cdf_power_law(t, x_min, scale):
    if t < x_min or t <= 0:
        return 0
    
    return 1 - -1*(scale-1)/(1-scale) * x_min**(scale-1) * t**(-scale+1)

def ecdf_data(t, lineinfo):
    x = lineinfo.get_xdata()
    y = lineinfo.get_ydata()

    for i in range(len(x)-1):
        #print(i)
        if x[i] <= t <= x[i+1]:
            print(y[i])
            return y[i]

    return y[-1]

def dist(x_min, scale, lineinfo):

    d = 0
    for i,val in enumerate(lineinfo.get_xdata()):
        temp = abs(cdf_power_law(val, x_min, scale) - ecdf_data(val, lineinfo))
        if temp > d:
            d = temp

    return d


df = pd.read_csv("retweeted_freq_biden_JoeBiden_18_09_2022.csv")

retwt_freqs = df["n"]

# ecdf = plt.ecdf(retwt_freqs)

# dist_partial = functools.partial(dist, scale=2, lineinfo=ecdf)
# results = optimize.minimize(dist_partial, 3000, method='Nelder-Mead')
# print(results)

# cdf = [cdf_power_law(x, 1, 1.85) for x in range(len(ecdf.get_xdata()))]

#power = [retwt_freqs[300]*(x+1)**(-1.85) for x in range(10**6)]

# plt.step(ecdf.get_xdata(), ecdf.get_ydata())
# plt.plot(cdf)

plt.scatter(list(range(len(retwt_freqs))), retwt_freqs)
#plt.plot(power)
plt.yscale('log')
plt.xscale('log')
# plt.ylabel('Frequency')
# plt.xlabel('Rank')
plt.show()
