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

n_rows = 2
n_cols = 2
fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols)

n_core_values = [10,50,100,500]
mle_r_exponents = []

mu = 10 # for suggestions
novelty = 5 # normal model novelty parameter
reinforcement = 5
strategy = "WSW"

for i,n_core in enumerate(n_core_values):
    experiment_tag = f"adjpos_initialsuggv3_E2_run{i}"
    data_path = experiment_tag + ".csv"

    # simulation and data storage
    if not exists(data_path):
        
        n_steps = 3*10**6
        seed = i

        starting_urns = simfuncs.generate_initial_urns(novelty)

        model = AdjPosModel(rng_seed=seed, novelty_param=novelty, reinforcement_param=reinforcement, \
            strategy=strategy, urns=starting_urns)

        extension = InitialSuggestionsMdlExt(n_to_suggest=n_core, ls_urns=starting_urns, mu=mu)

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

    row_i = int(i/n_rows)
    col_i = i % n_cols

    # MLE exponent calculations
    x_min_i = round(1.2*n_core)
    # for x in rank_r:
    #     if x >= n_core: # power law transition seems to be at rank (x-axis) == n_core
    #         x_min_i = x
    #         break
    
    freqs_xmin = freqs_r.copy()
    freqs_xmin = freqs_xmin[:x_min_i]
    
    exponent, err = analysisfuncs.continuous_power_mle(freqs_xmin, x_min_i-0.5)
    mle_r_exponents.append(exponent)

    power_law = [x**(-exponent) for x in range(1, len(rank_r))]

    percent_dist = freqs_r[x_min_i]/power_law[x_min_i]

    power_law = [x*percent_dist for x in power_law]

    axs[row_i, col_i].scatter(rank_c, freqs_c)
    axs[row_i, col_i].scatter(rank_r, freqs_r)
    axs[row_i, col_i].plot(power_law, linestyle='dashed', color="green")
    axs[row_i, col_i].set_xscale('log')
    axs[row_i, col_i].set_yscale('log')
    axs[row_i, col_i].set_ylim([1,5*10**5])
    axs[row_i, col_i].set_title(f"No. core = {n_core}, " + r"$\gamma_{\text{est}}$" + f" = {-exponent:.2f}")
    axs[row_i, col_i].legend(["Callers", "Receivers", r"$\gamma_{\text{est}}$ (receivers)"])

for i,n in enumerate(n_core_values):
    print(f"n_core: {n}, gamma_est: {mle_r_exponents[i]}")

fig.supxlabel("Rank")
fig.supylabel("Frequency")
fig.suptitle(r"$\nu$=" + f"{novelty}, " + r"$\rho$=" + f"{reinforcement}, " + r"$s$=" \
    + f"{strategy}, " + r"$\mu$=" + f"{mu}")
plt.tight_layout()
plt.show()
