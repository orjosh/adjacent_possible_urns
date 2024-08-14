import numpy as np
import scipy
from matplotlib import pyplot as plt
import sys
sys.path.append("scripts/")
import analysisfuncs as af

datasets, filenames = af.load_all_csvs("generated_data/", pattern="adjpos_orig_wsw*")

coeffs = []
for i, df in enumerate(datasets):
    print(filenames[i])
    receivers = df["receiver"].to_list()

    freqs = {}
    for j,x in enumerate(receivers):
        if x not in freqs:
            freqs[x] = 1
        else:
            freqs[x] += 1

    print(scipy.stats.variation(list(freqs.values()), ddof=1))
    coeffs.append(scipy.stats.variation(list(freqs.values()), ddof=1))

fig, ax = plt.subplots()
ax.boxplot(coeffs)
ax.set_ylabel("Coefficient of variation")
ax.set_title("COVs for Adjacent Possible model frequency data")
fig.set_size_inches(10,6)
fig.savefig("adjpos_orig_wsw_coef_var.png")