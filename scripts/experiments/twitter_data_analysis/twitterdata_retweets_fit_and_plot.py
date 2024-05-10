import powerlaw
import pickle
from matplotlib import pyplot as plt
import pandas as pd
import sys
sys.path.append("scripts/")
import analysisfuncs as af

FIT_PICKLE_FILENAME = "twitter_data/retweet_fits_powerlaw.pickle"

# Load all retweet frequency datasets
datasets, filenames = af.load_all_csvs("twitter_data/", pattern="retweeted_freq_*")

# Get the frequency counts as lists
freq_lists = {}
for i, df in enumerate(datasets):
    freqs = df["freq"].to_list()
    freq_lists[filenames[i]] = freqs


# Fit to the data
all_fits = {}
for i, freqs in enumerate(freq_lists.values()):
    name = filenames[i]
    this_fit = powerlaw.Fit(freqs, discrete=True, xmin_distance='V')
    all_fits[filenames[i]] = this_fit

    # fig, ax = plt.subplots()
    # fig.set_size_inches(10, 6)

    # this_fit.plot_ccdf(ax=ax, label="Data", linestyle=':', marker='o')
    # this_fit.power_law.plot_ccdf(ax=ax, linestyle="--", linewidth=2, label="Power-law")
    # this_fit.lognormal.plot_ccdf(ax=ax, linestyle="--", linewidth=2, label="Lognormal")
    # this_fit.truncated_power_law.plot_ccdf(ax=ax, linestyle="--", linewidth=2, label="Truncated power-law")

    # ax.set_title(name)
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.set_xlabel("Retweet frequency")
    # ax.set_ylabel(r'$P(X\geq x)$')
    # ax.set_ylim([10**(-6), 3])
    # ax.legend()

    # plt.show()
    # fig.savefig("figures/td_retweet" + name + ".png")
    # fig.clear()

    model_fit_objs = [this_fit.power_law, this_fit.lognormal, this_fit.truncated_power_law]

    first_params = []
    first_params_names = []
    second_params = []
    second_params_names = []
    for model in model_fit_objs:
        first_params.append(model.parameter1)
        first_params_names.append(model.parameter1_name)
        second_params.append(model.parameter2)
        second_params_names.append(model.parameter2_name)
    
    d = {'param1': first_params, 'param1_name': first_params_names,
        'param2': second_params, 'param2_name': second_params_names}

    df = pd.DataFrame(data=d)
    df.to_csv(f"figures/{name}_fit_params.csv")

    print(name)
    print(f"{this_fit.truncated_power_law.parameter1_name} = {this_fit.truncated_power_law.parameter1}")
    print(f"{this_fit.truncated_power_law.parameter2_name} = {this_fit.truncated_power_law.parameter2}")