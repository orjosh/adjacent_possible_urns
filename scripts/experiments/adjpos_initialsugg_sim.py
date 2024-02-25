import sys
sys.path.append("scripts/")

from os.path import exists
from matplotlib import pyplot as plt
import pandas as pd
import pickle
import functools
from adjacentpossible import AdjPosModel, UserUrn
from initialsuggestions_extension import InitialSuggestionsMdlExt
import simfuncs
import analysisfuncs

fig, axs = plt.subplots(nrows=2, ncols=3)

n_core_values = [10,50,100,500,1000]
for i,n_core in enumerate(n_core_values):
    experiment_tag = f"adjpos_initialsugg_run{i}"
    data_path = experiment_tag + ".csv"

    # simulation and data storage
    if not exists(data_path):
        novelty = 5
        reinforcement = 5
        strategy = "WSW"
        n_steps = 3*10**6
        seed = i

        starting_urns = simfuncs.generate_initial_urns(novelty)

        model = AdjPosModel(rng_seed=seed, novelty_param=novelty, reinforcement_param=reinforcement, \
            strategy=strategy, urns=starting_urns)

        extension = InitialSuggestionsMdlExt(n_to_suggest=n_core, ls_urns=starting_urns)

        timestep = functools.partial(model.time_step, begin_ext=extension.populate_list_begin_ext, \
            alt_novelty=extension.custom_novelty_step, end_ext=extension.update_suggestions)

        csv_data = simfuncs.run_model(model_instance=model, n_steps=n_steps, custom_timestep=timestep)

        simfuncs.write_data_to_csv(csv_data, data_path)

    # pickling and plotting
    data = pd.read_csv(data_path)
    event_callers = data["caller"]
    event_receivers = data["receiver"]

    pickle_filename_freqrank_caller = experiment_tag + "_freqrank_caller"
    pickle_filename_freqrank_receiver = experiment_tag + "_freqrank_receiver"

    if not exists(pickle_filename_freqrank_caller + ".pickle"):
        simfuncs.pickle_freqrank(event_callers, pickle_filename_freqrank_caller)

    if not exists(pickle_filename_freqrank_receiver + ".pickle"):
        simfuncs.pickle_freqrank(event_receivers, pickle_filename_freqrank_receiver)

    rank_c, freqs_c = simfuncs.get_plottable_freqrank_from_pickle(pickle_filename_freqrank_caller + ".pickle")
    rank_r, freqs_r = simfuncs.get_plottable_freqrank_from_pickle(pickle_filename_freqrank_receiver + ".pickle")

    row_i = 0
    col_i = i
    if i > 2:
        row_i = 1
        col_i -= 3

    axs[row_i, col_i].scatter(rank_c, freqs_c)
    axs[row_i, col_i].scatter(rank_r, freqs_r)
    axs[row_i, col_i].set_xscale('log')
    axs[row_i, col_i].set_yscale('log')
    axs[row_i, col_i].set_title(f"No. core = {n_core}")
    axs[row_i, col_i].legend(["Callers", "Receivers"])

fig.supxlabel("Rank")
fig.supylabel("Frequency")
plt.tight_layout()
plt.show()
