from matplotlib import pyplot as plt
import pandas as pd
import csv
import math

import sys
sys.path.append("scripts/")

from adjacentpossible import AdjPosModel, UserUrn
import simfuncs
import analysisfuncs
        
# Model parameters
s = "WSW"
n_steps = 5 * 10**6
rho_nu_vals = [10]

# Simulation
data_store_path = "generated_data/adjpos_orig_wsw_rhonueq"

for i,v in enumerate(rho_nu_vals):
    reinforcement = v
    novelty = v

    csv_path = data_store_path + f"{reinforcement}.csv" # rho = nu always

    starting_urns = simfuncs.generate_initial_urns(novelty)

    model = AdjPosModel(novelty_param=novelty, reinforcement_param=reinforcement, \
            strategy=s, urns=starting_urns)

    seen_edges = {}
    n_urns_seen = 0 # different from model.n_urns; does not count empty urns
    seen_urns = {}
    csv_data = []
    avg = 0

    for j in range(n_steps):
        model.time_step()

        most_recent_edge = model.events[j]
        urn_r = most_recent_edge[1]

        n_seen_prev = n_urns_seen

        if urn_r not in seen_urns:
            seen_urns[urn_r] = 1
            n_urns_seen += 1

        if most_recent_edge not in seen_edges:
            # some node's degree has increased by 1 connection
            seen_edges[most_recent_edge] = 1
            degree_sum = avg*n_seen_prev
            avg = (degree_sum + 1)/n_urns_seen

        csv_data.append( (most_recent_edge[0], most_recent_edge[1], model.n_urns, avg) )
        
        print(f"Sim {i}, step {j+1}/{n_steps}, avg: {avg}, seen: {n_urns_seen}/{model.n_urns}")

    with open(csv_path, 'x', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["caller", "receiver", "num_urns", "avg_in_deg"])
        writer.writerows(csv_data)