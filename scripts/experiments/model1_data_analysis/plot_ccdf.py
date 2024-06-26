import powerlaw
import pickle
from matplotlib import pyplot as plt
import sys
sys.path.append("./scripts")
import analysisfuncs as af

# Load all datasets (sequence of (caller, receiver) events)
datasets, filenames = af.load_all_csvs("generated_data/", pattern="adjpos_orig_wsw_*")

# Calculate freqs as list of receivers
freq_lists = {}
for i, df in enumerate(datasets):
    receivers = df["receiver"].to_list()

    freqs = {}
    for j,x in enumerate(receivers):
        if x not in freqs:
            freqs[x] = 1
        else:
            freqs[x] += 1

    freq_lists[filenames[i]] = list(freqs.values())
    #print(list(freqs.values()))

# Fit to the data
all_fits = {}
for i, freqs in enumerate(freq_lists.values()):
    name = filenames[i]
    this_fit = powerlaw.Fit(freqs)
    all_fits[filenames[i]] = this_fit

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 6)

    this_fit.plot_ccdf(ax=ax, linestyle='--', marker='o', label="Data")
    this_fit.power_law.plot_ccdf(ax=ax, linestyle="--", linewidth=2, label=r"Power-law ($\gamma$=" + f"{this_fit.power_law.alpha:.2f})")
    this_fit.lognormal.plot_ccdf(ax=ax, linestyle="--", linewidth=2, label="Lognormal")
    this_fit.truncated_power_law.plot_ccdf(ax=ax, linestyle="--", linewidth=2, label="Truncated power-law")

    print(f"KS (Trunc.): {this_fit.truncated_power_law.KS()}")
    print(f"Trunc. vs PL: {this_fit.distribution_compare("truncated_power_law", "power_law")}")

    ax.set_title(name)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("In-degree frequency")
    ax.set_ylabel(r'$P(X\geq x)$')
    ax.set_ylim([10**(-5), 5])
    ax.legend()

    plt.show()
    fig.savefig("figures/Tria Model" + name + "_indegfit.png")
    fig.clear()