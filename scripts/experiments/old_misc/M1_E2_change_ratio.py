import sys
sys.path.append("scripts/")

import csv
import pickle
import pandas as pd
import numpy as np
from os.path import exists
from matplotlib import pyplot as plt
from adjacentpossible import AdjPosModel, UserUrn
from datautils import discrete_power_mle_approx

# In this experiment, we change the ratio R = reinforcement / novelty once a certain number of urns
# have been created. We then plot frequency vs. rank of urn IDs in the event sequence.

experiment_tag = "M1_E2"

data_path = experiment_tag + ".csv"

if not exists(data_path):
    # Model Setup
    novelty = 5
    reinforcement = 5
    strategy = "WSW"
    n_steps = 3*10**6
    seed = 4321

    n_starting_urns = 2 + 2*(novelty+1) # 2 main urns with novelty+1 contacts to share each
    starting_urns = []
    for i in range(n_starting_urns):
        u = UserUrn(i+1, {})
        starting_urns.append(u)

    for i in range(3, 3+novelty+1):
        starting_urns[0].add_contact(i)

    for i in range(3+novelty+1, n_starting_urns+1):
        starting_urns[1].add_contact(i)

    print(f"{starting_urns[0]}")
    print(f"{starting_urns[1]}")
    
    csv_rows = []

    model = AdjPosModel(rng_seed=1234, novelty_param=novelty, reinforcement_param=reinforcement, \
        strategy=strategy, urns=starting_urns)

    # Run Simulation
    for i in range(n_steps):
        print(f"Step {i+1}/{n_steps}\tNo. urns: {model.n_urns}")

        if model.n_urns >= 10**6:
            model.reinforcement_param = 15
        
        model.time_step()
        last_event = model.events[i]
        csv_rows.append((last_event[0], last_event[1], model.n_urns))

    with open(data_path, 'x', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["caller", "receiver", "num_urns"])
        writer.writerows(csv_rows)

# Analysis ----------------------------------------------------------------------------------------

data = pd.read_csv(data_path)

# No. edges vs. time
num_urns = data["num_urns"].iloc[-1]
event_callers = data["caller"]
event_receivers = data["receiver"]

pickle_path = experiment_tag + "_freqrank3.pickle"

frequencies = {}
if not exists(pickle_path):
    seen_edges = {}
    for i,c in enumerate(event_callers):
        r = event_receivers.iloc[i]

        if c not in frequencies:
            frequencies[c] = 1
        else:
            frequencies[c] += 1

        if r not in frequencies:
            frequencies[r] = 1
        else:
            frequencies[r] += 1

        print(f"Iteration {i}")

    with open(pickle_path, 'wb') as f:
        pickle.dump(frequencies, f)

with open(pickle_path, 'rb') as f:
    frequencies = pickle.load(f)

with open("M1_E2_freqrank.pickle", 'rb') as f:
    freqs2 = pickle.load(f)

frequencies = list(frequencies.values())
frequencies.sort(reverse=True)

freqs2 = list(freqs2.values())
freqs2.sort(reverse=True)

plt.scatter(list(range(len(frequencies))), frequencies)
plt.scatter(list(range(len(freqs2))), freqs2)
plt.xscale('log')
plt.yscale('log')
# plt.xlim([10**2, 10**7])
# plt.ylim([10, 10**7])
plt.ylabel("Number of edges")
plt.xlabel("t")
plt.title("Number of edges in the network at timestep $t$")
plt.show()