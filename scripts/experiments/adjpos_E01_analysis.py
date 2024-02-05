import csv
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

sys.path.append("scripts/")

sequence_df = pd.read_csv("adjpos_E01_1.csv")
callers = sequence_df["caller"]
receivers = sequence_df["receiver"]

caller_freqs = callers.value_counts()
receiver_freqs = receivers.value_counts()

combined_freqs = caller_freqs.add(receiver_freqs) # if ID is not in both series, returns NaN
for index, val in combined_freqs.items():
    if pd.isnull(val):
        # since we added receiver_freqs to caller_freqs, missing entries
        # are unique to receiver_freqs
        combined_freqs[index] = receiver_freqs[index]

combined_freqs.sort_values(ascending=False, inplace=True)
#print(combined_freqs)

rank = list(range(len(combined_freqs)))

constant = combined_freqs.iloc[0]
basic_zipf = [constant/r for r in rank]

plt.plot(rank, combined_freqs, 'o', markersize=2)
plt.plot(rank, basic_zipf, '--', color="lightgray")
plt.title("Frequency vs. rank for urn IDs in the event sequence")
plt.ylabel("Frequency")
plt.xlabel("Rank")
plt.xscale('log')
plt.yscale('log')
plt.legend(["Simulation data", "$\\frac{f(1)}{r}$"])
plt.show()