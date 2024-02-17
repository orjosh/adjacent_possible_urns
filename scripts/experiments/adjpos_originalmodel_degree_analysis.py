import networkx as nx
import pandas as pd
import numpy as np
import csv
import pickle
from matplotlib import pyplot as plt
from os.path import exists

experiment_tag = "M1_E1"
data_path = experiment_tag + ".csv"

data = pd.read_csv(data_path)

num_urns = data["num_urns"].iloc[-1]
event_callers = data["caller"]
event_receivers = data["receiver"]

pickle_filename_indegree = experiment_tag + "_in_deg.pickle"
in_degrees = {}
if not exists(pickle_filename_indegree):
    seen_edges = {}
    for i,c in enumerate(event_callers):
        # event (c,r) corresponds to r retweeting c, i.e. in-degree for c
        r = event_receivers.iloc[i]

        e = (c,r)
        if e not in seen_edges:
            if c not in in_degrees:
                in_degrees[c] = 1
            else:
                in_degrees[c] += 1
            
            seen_edges[e] = 1

        print(f"Iteration {i}")

    with open(pickle_filename_indegree, 'wb') as f:
        pickle.dump(in_degrees, f)

pickle_filename_outdegree = experiment_tag + "_out_deg.pickle"
out_degrees = {}
if not exists(pickle_filename_outdegree):
    seen_edges = {}
    for i,c in enumerate(event_callers):
        # event (c,r) corresponds to r retweeting c, i.e. in-degree for c
        r = event_receivers.iloc[i]

        e = (c,r)
        if e not in seen_edges:
            if r not in out_degrees:
                out_degrees[r] = 1
            else:
                out_degrees[r] += 1
            
            seen_edges[e] = 1

        print(f"Iteration {i}")

    with open(pickle_filename_outdegree, 'wb') as f:
        pickle.dump(out_degrees, f)

with open(pickle_filename_indegree, 'rb') as f:
    in_degrees = pickle.load(f)

with open(pickle_filename_outdegree, 'rb') as f:
    out_degrees = pickle.load(f)

in_degrees_values = list(in_degrees.values())
in_degrees_values.sort()

out_degrees_values = list(out_degrees.values())
out_degrees_values.sort()

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

axs[0].hist(in_degrees_values, bins=15, log=True)
axs[1].hist(out_degrees_values, bins=15, log=True)
plt.ylabel("Count")
# axs[0].xlabel("$k_{in}$")
plt.show()