import sys
sys.path.append("scripts/")

import csv
import pickle
import pandas as pd
import numpy as np
from os.path import exists
from matplotlib import pyplot as plt
from adjacentpossible import AdjPosModel, UserUrn
from analysisfuncs import discrete_power_mle_approx

data_path = "M1_E3.csv"

if not exists(data_path):
    # Model Setup
    novelty = 5
    reinforcement = 5
    strategy = "WSW"
    n_steps = 10**7
    seed = 1234

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

n_edges_t = []
edge_count = 0
if not exists("M1_E1_evst.pickle"):
    seen_edges = {}
    for i,c in enumerate(event_callers):
        # for edges, (c,r) == (r,c); add to dictionary as (larger_id, smaller_id)
        r = event_receivers.iloc[i]

        edge = None
        if c > r:
            edge = (c,r)
        else:
            edge = (r,c)

        if edge not in seen_edges:
            seen_edges[edge] = 1
            edge_count += 1

        print(f"Iteration {i}")
        n_edges_t.append(edge_count)

    with open("M1_E1_evst.pickle", 'wb') as f:
        pickle.dump(n_edges_t, f)

with open("M1_E1_evst.pickle", 'rb') as f:
    n_edges_t = pickle.load(f)

x_min = int(10**3) # TODO determined visually, not robust
exponent, err = discrete_power_mle_approx(n_edges_t, x_min)

mle_power_law = [x**exponent for x in range(len(n_edges_t))]

visual_power_law = [x**0.95 for x in range(len(n_edges_t))]

print(f"{exponent:.5f} +- {err:.5f}")

plt.plot(n_edges_t)
plt.plot(mle_power_law, linestyle="--")
plt.plot(visual_power_law, linestyle="--")
plt.xscale('log')
plt.yscale('log')
plt.xlim([10**2, 10**7])
plt.ylim([10, 10**7])
plt.ylabel("Number of edges")
plt.xlabel("t")
plt.legend(["Simulation", f"MLE estimate ($\gamma$={exponent:.2f})", f"Paper estimate ($\gamma$={0.95})"])
plt.title("Number of edges in the network at timestep $t$")
plt.show()