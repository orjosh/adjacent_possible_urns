import sys
sys.path.append("scripts/")

from adjacentpossible import AdjPosModel, UserUrn
from os.path import exists
from matplotlib import pyplot as plt
import csv
import pickle
import pandas as pd
import numpy as np

def asymmetrical_reinforcement(self, caller, receiver):
    # Receiver gets caller as contact, but only caller is reinforced
    receiver.add_contact(caller.ID)
    for i in range(self.reinforcement_param):
        caller.add_contact(receiver.ID)

        self.urn_sizes[caller.ID] = caller.size
        self.prop_choice.append(caller.ID)

        self.total_size += 1

experiment_tag = "adjpos_asym_reinforcement"
num_repeats = 10
in_degree_data = []
out_degree_data = []

for i in range(num_repeats):
    data_path = experiment_tag + f"_run{i+1}.csv"

    if not exists(data_path):
        # Model Setup
        novelty = 5
        reinforcement = 5
        strategy = "WSW"
        n_steps = 10**6

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

        model = AdjPosModel(novelty_param=novelty, reinforcement_param=reinforcement, \
            strategy=strategy, urns=starting_urns)

        # Run Simulation
        for i in range(n_steps):
            print(f"Step {i+1}/{n_steps}\tNo. urns: {model.n_urns}")

            model.time_step(alt_reinforcement=asymmetrical_reinforcement)
            last_event = model.events[i]
            csv_rows.append((last_event[0], last_event[1], model.n_urns))

        with open(data_path, 'x', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["caller", "receiver", "num_urns"])
            writer.writerows(csv_rows)

        # Analysis ----------------------------------------------------------------------------------------

    data = pd.read_csv(data_path)

    num_urns = data["num_urns"].iloc[-1]
    event_callers = data["caller"]
    event_receivers = data["receiver"]

    in_degrees = {}
    out_degrees = {}
    seen_edges_c = {}
    seen_edges_r = {}
    for i,c in enumerate(event_callers):
        # event (c,r) corresponds to r retweeting c, i.e. in-degree for c
        r = event_receivers.iloc[i]

        e = (c,r)
        if e not in seen_edges_c:
            if c not in in_degrees:
                in_degrees[c] = 1
            else:
                in_degrees[c] += 1
            
            seen_edges_c[e] = 1

        if e not in seen_edges_r:
            if r not in out_degrees:
                out_degrees[r] = 1
            else:
                out_degrees[r] += 1
            
            seen_edges_r[e] = 1

        print(f"Iteration {i}")
    
    in_deg_values = list(in_degrees.values())
    in_deg_values.sort(reverse=True)
    in_degree_data.append(in_deg_values)

    out_deg_values = list(out_degrees.values())
    out_deg_values.sort(reverse=True)
    out_degree_data.append(out_deg_values)


in_deg_avg = []
shortest_len = len(in_degree_data[0])
for i,run in enumerate(in_degree_data):
    if len(run) < shortest_len:
        shortest_len = len(run)

print(shortest_len)
for i in range(shortest_len):
    avg = 0
    for j,run in enumerate(in_degree_data):
        avg += run[i]
    in_deg_avg.append(avg/num_repeats)

out_deg_avg = []
shortest_len = len(out_degree_data[0])
for i,run in enumerate(out_degree_data):
    if len(run) < shortest_len:
        shortest_len = len(run)

for i in range(shortest_len):
    avg = 0
    for j,run in enumerate(out_degree_data):
        avg += run[i]
    out_deg_avg.append(avg/num_repeats)

plt.scatter(list(range(len(in_deg_avg))), in_deg_avg, s=10)
plt.scatter(list(range(len(out_deg_avg))), out_deg_avg, s=10)
plt.yscale('log')
plt.xscale('log')
plt.ylabel('Frequency')
plt.xlabel('Rank')
plt.legend(["Callers", "Receivers"])
plt.title("Frequency vs. Rank with caller-favoured asymmetrical reinforcement")
plt.show()