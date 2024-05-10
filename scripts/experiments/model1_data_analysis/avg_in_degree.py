from matplotlib import pyplot as plt
import sys
sys.path.append("scripts/")
import analysisfuncs as af

DATASET_PATH_REGEX = "adjpos_orig_wsw_rhonueq*" # other datasets dont have this data

# Load datasets
datasets, filenames = af.load_all_csvs("generated_data/", pattern=DATASET_PATH_REGEX)

for i, df in enumerate(datasets):
    avg_in_deg = df["avg_in_deg"].to_list()

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 6)

    ax.plot(avg_in_deg)
    ax.set_ylabel(r"Average $k_{in}$")
    ax.set_xlabel("Time step")
    ax.set_title(f"{filenames[i]}")

    #ax.set_xscale('log')

    plt.show()