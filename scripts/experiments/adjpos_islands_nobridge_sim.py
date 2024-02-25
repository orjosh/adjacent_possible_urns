import sys
sys.path.append("scripts/")

import csv
import pickle
import pandas as pd
import numpy as np
from os.path import exists
from matplotlib import pyplot as plt
from adjacentpossible import UserUrn
from islands_extension import IslandsMdlExt

experiment_tag = "adjpos_islands_nobridge_run1"

data_path = experiment_tag + ".csv"

if not exists(data_path):
    # Same as M1_E1 except for n_steps
    novelty = 5
    reinforcement = 5
    strategy = "WSW"
    n_steps = 3*10**6
    seed = 4444

    n_starting_urns = 2 + 2*(novelty+1) # 2 main urns with novelty+1 contacts to share each
    starting_urns = []
    for i in range(n_starting_urns):
        u = UserUrn(i+1, {})
        starting_urns.append(u)

    for i in range(3, 3+novelty+1):
        starting_urns[0].add_contact(i)

    for i in range(3+novelty+1, n_starting_urns+1):
        starting_urns[1].add_contact(i)

    # Extension parameters
    prob_core = 0.99
    max_core = 10**5 # Gerlach and Altmann 2013 shows that max_core is ~8000 for English

    model = IslandsMdlExt(prob_core=prob_core, max_core=max_core, rng_seed=1234, \
        novelty_param=novelty, reinforcement_param=reinforcement, strategy=strategy, \
            urns=starting_urns)

    # Run Simulation
    csv_rows = []

    for i in range(n_steps):
        print(f"Step {i+1}/{n_steps}\t{model.n_core}/{model.max_core} core urns")

        model.time_step()
        last_event = model.events[i]
        csv_rows.append((last_event[0], last_event[1], model.n_urns))

    with open(data_path, 'x', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["caller", "receiver", "num_urns"])
        writer.writerows(csv_rows)

# Analysis ------------------------------------------------------------------------------

data = pd.read_csv(data_path)

# No. edges vs. time
num_urns = data["num_urns"].iloc[-1]
event_callers = data["caller"]
event_receivers = data["receiver"]

pickle_filename_freqrank_caller = experiment_tag + "_freqrank_caller.pickle"
pickle_filename_freqrank_receiver = experiment_tag + "_freqrank_receiver.pickle"

edge_count = 0
freqs_caller = {}
if not exists(pickle_filename_freqrank_caller):
    seen_edges = {}
    for i,c in enumerate(event_callers):
        if c not in freqs_caller:
            freqs_caller[c] = 1
        else:
            freqs_caller[c] += 1

        print(f"Iteration {i}")

    with open(pickle_filename_freqrank_caller, 'wb') as f:
        pickle.dump(freqs_caller, f)

edge_count = 0
freqs_receiver = {}
if not exists(pickle_filename_freqrank_receiver):
    seen_edges = {}
    for i,r in enumerate(event_callers):
        if r not in freqs_receiver:
            freqs_receiver[r] = 1
        else:
            freqs_receiver[r] += 1

        print(f"Iteration {i}")

    with open(pickle_filename_freqrank_receiver, 'wb') as f:
        pickle.dump(freqs_receiver, f)

with open(pickle_filename_freqrank_caller, 'rb') as f:
    freqs_caller = pickle.load(f)

with open(pickle_filename_freqrank_receiver, 'rb') as f:
    freqs_receiver = pickle.load(f)

freqs_caller = list(freqs_caller.values())
freqs_caller.sort(reverse=True)
freqs_receiver = list(freqs_receiver.values())
freqs_receiver.sort(reverse=True)

plt.scatter(list(range(len(freqs_caller))), freqs_caller)
plt.scatter(list(range(len(freqs_receiver))), freqs_receiver)
plt.xscale('log')
plt.yscale('log')
plt.ylabel("Frequency")
plt.xlabel("Rank")
plt.title("Frequency vs. rank for the Islands (Bridged) model extension")
plt.legend(["Callers", "Receivers"])
plt.text(10,100, r"$\rho=5$\n$v=5$\n$T=3\times10^6$\n$s=\text{WSW}$$")
plt.show()