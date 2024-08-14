from matplotlib import pyplot as plt
import numpy as np
import scipy

DISTR_SIZE = 100000 # no. points to sample

rng = np.random.default_rng()

do_gaussian = True
do_power = True
do_exp = False

# GAUSSIAN ################################################
if do_gaussian:
    gaussian_means = np.linspace(start=10,stop=1000,num=50)
    gaussian_stddevs = np.linspace(start=1,stop=100,num=50)
    gaussian_skews = []
    gaussian_kurts = []

    for i,mean in enumerate(gaussian_means):
        for j,stddev in enumerate(gaussian_stddevs):
            print(f"{mean}, {stddev}")

            distribution = rng.normal(loc=mean, scale=stddev, size=DISTR_SIZE)

            skew = scipy.stats.skew(distribution)
            gaussian_skews.append(skew)

            kurt = scipy.stats.kurtosis(distribution)
            gaussian_kurts.append(kurt)

    fig, axs = plt.subplots(1,2)

    axs[0].boxplot(gaussian_skews)
    axs[0].set_ylabel("Skewness")

    axs[1].boxplot(gaussian_kurts)
    axs[1].set_ylabel("Kurtosis")

    fig.suptitle(r"Moments for Gaussian distributions with $\bar{x}\in [10,1000]$ and $\sigma\in [1,100]$")
    fig.set_size_inches(10,6)
    fig.tight_layout()
    fig.savefig("gaussian_skew_kurt.png")

# POWER LAW (ZIPFIAN) ################################################
if do_power:
    power_scales = np.linspace(start=1.01, stop=3.00, num=2500)
    power_skews = []
    power_kurts = []

    for i,scaling in enumerate(power_scales):
        print(f"{scaling}")

        distribution = rng.zipf(a=scaling, size=DISTR_SIZE)

        skew = scipy.stats.skew(distribution)
        power_skews.append(skew)

        kurt = scipy.stats.kurtosis(distribution)
        power_kurts.append(kurt)

    fig, axs = plt.subplots(1,2)

    axs[0].boxplot(power_skews)
    axs[0].set_ylabel("Skewness")

    axs[1].boxplot(power_kurts)
    axs[1].set_ylabel("Kurtosis")

    fig.suptitle(r"Moments for Zipfian distributions with $\gamma\in [1.01,3.00]$")
    fig.set_size_inches(10,6)
    fig.tight_layout()
    fig.savefig("power_skew_kurt.png")

# EXPONENTIAL ################################################
if do_exp:
    exp_scales = np.linspace(start=1.1,stop=1000,num=2500)
    exp_skews = []
    exp_kurts = []

    for i,scaling in enumerate(exp_scales):
        print(f"{scaling}")

        distribution = rng.exponential(scale=scaling, size=DISTR_SIZE)

        skew = scipy.stats.skew(distribution)
        exp_skews.append(skew)

        kurt = scipy.stats.kurtosis(distribution)
        exp_kurts.append(kurt)

    fig, axs = plt.subplots(1,2)

    axs[0].boxplot(exp_skews)
    axs[0].set_ylabel("Skewness")

    axs[1].boxplot(exp_kurts)
    axs[1].set_ylabel("Kurtosis")

    fig.suptitle(r"Moments for exponential distributions with $\gamma\in [1.1,1000]$")
    fig.set_size_inches(10,6)
    fig.tight_layout()
    fig.savefig("exp_skew_kurt.png")