from matplotlib import pyplot as plt
import numpy as np
import scipy

# This script calculates the coefficient of variation of some distributions to see if it is a useful
# measure with which to distinguish distributions from each other

DISTR_SIZE = 100000

rng = np.random.default_rng()

# Gaussian ####################################################################
gaussian_means = np.linspace(start=10,stop=1000,num=50)
gaussian_stddevs = np.linspace(start=1,stop=100,num=50)
#gaussian_results_mat = np.zeros((len(gaussian_means), len(gaussian_stddevs)))
gaussian_results = []

for i,mean in enumerate(gaussian_means):
    for j,stddev in enumerate(gaussian_stddevs):
        print(f"{mean}, {stddev}")
        distribution = rng.normal(loc=mean, scale=stddev, size=DISTR_SIZE)
        coef_var = scipy.stats.variation(distribution, ddof=1)
        #gaussian_results_mat[i][j] = coef_var
        gaussian_results.append(coef_var)

fig, ax = plt.subplots()
# ax.hist(gaussian_results)
# ax.set_xlabel("Coefficient of variation")
# ax.set_ylabel("Count")
ax.boxplot(gaussian_results)
ax.set_ylabel("Coefficient of variation")
ax.set_title(r"COV for Gaussian distributions with $\bar{x}\in [10,1000]$ and $\sigma\in [1,100]$")
fig.set_size_inches(10,6)
fig.savefig("gaussian_coef_var2.png")

# Exponential #########################################
scales = np.linspace(start=1.1,stop=1000,num=2500)
exp_results = []

for x in scales:
    print(f"{x}")
    exp = rng.exponential(scale=x, size=DISTR_SIZE)
    exp_coef_var = scipy.stats.variation(exp, ddof=1)
    exp_results.append(exp_coef_var)

fig, ax = plt.subplots()
# ax.hist(exp_results)
# ax.set_xlabel("Coefficient of variation")
# ax.set_ylabel("Count")
ax.boxplot(exp_results)
ax.set_ylabel("Coefficient of variation")
ax.set_title(r"COV for Exponential distributions $\frac{1}{\beta}e^{-x/\beta}$ with $\beta\in[1.1,1000]$")
fig.set_size_inches(10,6)
fig.savefig("exp_coef_var.png")

# Power (Zipfian) ###############################################################
scales = np.linspace(start=1.01, stop=3.00, num=199)
power_results = []

for x in scales:
    rng = np.random.default_rng()
    power = rng.zipf(a=x, size=DISTR_SIZE)
    power_coef_var = scipy.stats.variation(power, ddof=1)
    power_results.append(power_coef_var)

fig, ax = plt.subplots()
#ax.hist(power_results)
# ax.set_xlabel("Coefficient of variation")
# ax.set_ylabel("Count")
ax.boxplot(power_results)
ax.set_ylabel("Coefficient of variation")
ax.set_title(r"COV for Zipfian distributions $\frac{k^{-\gamma}}{\zeta(\gamma)}$ with $\gamma\in[1.01,3.00]$")
fig.set_size_inches(10,6)
fig.savefig("power_coef_var.png")