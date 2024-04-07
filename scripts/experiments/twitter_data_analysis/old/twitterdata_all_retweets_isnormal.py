import pandas as pd

csv_path_prefix = "twitter_data/retweeted_freq"

filenames = []
dataset_dicts = []
users_lists = []
freq_datasets = []
for p in Path(".").glob(csv_path_prefix + "*"):
    filename = str(p).split(csv_path_prefix + "_")[1]
    filename = filename.split(".")[0]
    filenames.append(filename)
    print(filename)

    data = pd.read_csv(str(p))
    freqs = data["freq"].to_list()
    users = data["user"].to_list()

    freqs_norm = [x/freqs[0] for x in freqs]
    ranks = [x for x in range(len(freqs_norm))]
    freq_datasets.append((ranks, freqs_norm))

    users_lists.append(set(users))

    dataset_dicts.append(dict(zip(users, freqs)))

lower_freq_datasets = [] # want to look at only ranks >= some number
cutoff = int(500)
for d in freq_datasets:
    ranks = d[0]
    freqs = d[1]
    
    freqs_cutoff = freqs[cutoff:len(freqs)+1] # note: is normalized
    ranks_cutoff = ranks[cutoff:len(ranks)+1]
    print(len(freqs_cutoff))
    print(len(ranks_cutoff))

    lower_freq_datasets.append((ranks_cutoff, freqs_cutoff))