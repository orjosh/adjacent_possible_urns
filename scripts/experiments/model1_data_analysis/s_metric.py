import sys
sys.path.append("scripts/")
import analysisfuncs as af
import numpy as np
from matplotlib import pyplot as plt
import powerlaw

DATASET_PATH_REGEX = "adjpos_orig_wsw_rho6_nu3.csv"

# Load datasets
datasets, filenames = af.load_all_csvs("generated_data/", pattern=DATASET_PATH_REGEX)

# Calculate degree of each urn in each dataset
for i, df in enumerate(datasets):
    seen_edges = {}

    in_degrees = {}
    out_degrees = {}
    total_degrees = {}

    callers = df["caller"].to_list()
    receivers = df["receiver"].to_list()

    for j,x in enumerate(callers):
        event = (callers[j], receivers[j])
        if event not in seen_edges:
            c = event[0]
            r = event[1]

            if r not in in_degrees:
                in_degrees[r] = 1
            else:
                in_degrees[r] += 1

            if r not in total_degrees:
                total_degrees[r] = 1
            else:
                total_degrees[r] += 1

            if c not in out_degrees:
                out_degrees[c] = 1
            else:
                out_degrees[c] += 1

            if c not in total_degrees:
                total_degrees[c] = 1
            else:
                total_degrees[c] += 1

            seen_edges[event] = 1

    all_ids = list(total_degrees.keys())
    delta_s_values = np.zeros(len(all_ids), dtype=np.int32)
    degs = np.array(list(total_degrees.values()), dtype=np.int32)

    # The following calculates k_i*k_j for all pairs i,j and stores it in a matrix S
    S = np.einsum("i,j->ij", degs, degs)
    
    for j, row in enumerate(S):
        print(f"Calculating delta s {j/len(S)*100:.0f}%", end='\r')
        delta_s_values[j] = max(row) - min(row)

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(10, 6)
    fig.suptitle(filenames[i])

    # Histogram
    logbins = np.geomspace(delta_s_values.min(), delta_s_values.max(), 20)
    axs[0].hist(delta_s_values, bins=logbins, edgecolor = 'black')
    axs[0].set_yscale('log')
    axs[0].set_xscale('log')
    axs[0].set_xlabel(r"$\Delta s$")
    axs[0].set_ylabel("Count")
    axs[0].set_title(r"$\Delta s$" + " counts for each node in the network")

    # powerlaw fits
    fit = powerlaw.Fit(delta_s_values, discrete=True, xmin_distance='V')
    fit.plot_ccdf(ax=axs[1], label="Data", linestyle="--", marker='o', original_data=True)
    fit.power_law.plot_ccdf(ax=axs[1], linestyle="--", linewidth=2, label=r"Power-law ($\gamma$=" + f"{fit.power_law.alpha:.2f})")
    fit.lognormal.plot_ccdf(ax=axs[1], linestyle='--', linewidth=2, label="Lognormal")
    axs[1].set_xlabel(r"$\Delta s$")
    axs[1].set_ylabel(r"$P(X\geq x)$")
    axs[1].legend()

    plt.show()

    R, p = fit.distribution_compare('power_law', 'truncated_power_law')
    print(fit.supported_distributions)
    print(f"({R}, {p})")