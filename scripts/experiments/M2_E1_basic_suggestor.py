import sys
sys.path.append("scripts/")

import csv
import pickle
import pandas as pd
import numpy as np
from os.path import exists
from matplotlib import pyplot as plt
from adjacentpossible import AdjPosModel, UserUrn
from suggestor_extension import SuggestorMdlExt
from datautils import discrete_power_mle_approx

data_path = "M2_E1.csv"

if not exists(data_path):
    # Same as M1_E1 except for n_steps
    novelty = 5
    reinforcement = 5
    strategy = "WSW"
    n_steps = 3*10**6
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

    model = AdjPosModel(rng_seed=1234, novelty_param=novelty, reinforcement_param=reinforcement, \
        strategy=strategy, urns=starting_urns)

    # Extension parameters
    prob_suggestion = 0.99
    size_to_suggest = 100 # not really needed, max_suggestable will keep the core vocab finite
    max_suggestable = 10**5

    mdl_extension = SuggestorMdlExt(prob_suggestion, size_to_suggest, max_suggestable)

    # Run Simulation
    csv_rows = []

    for i in range(n_steps):
        print(f"Step {i+1}/{n_steps}\t{mdl_extension.n_suggestable}/{mdl_extension.max_sugg} core urns")

        model.time_step(begin_ext=mdl_extension.do_suggestion_begin_ext, \
            end_ext=mdl_extension.check_suggestable_end_ext)
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

pickle_filename_freqrank = "M2_E1_freqrank.pickle"

frequencies = {}
if not exists(pickle_filename_freqrank):
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

    with open(pickle_filename_freqrank, 'wb') as f:
        pickle.dump(frequencies, f)

with open(pickle_filename_freqrank, 'rb') as f:
    frequencies = pickle.load(f)

frequencies = list(frequencies.values())
frequencies.sort(reverse=True)

plt.scatter(list(range(len(frequencies))), frequencies)
plt.xscale('log')
plt.yscale('log')
plt.ylabel("Frequency")
plt.xlabel("Rank")
plt.title("Frequency vs. rank for the Basic Suggestor extension")
plt.show()