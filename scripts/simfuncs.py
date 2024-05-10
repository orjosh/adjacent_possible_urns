import csv
import pandas as pd
import pickle
from adjacentpossible import UserUrn

def generate_initial_urns(novelty):
    n_starting_urns = 2 + 2*(novelty+1)

    urns = []
    for i in range(n_starting_urns):
        u = UserUrn(i+1, {})
        urns.append(u)

    for i in range(3, 3+novelty+1):
        urns[0].add_contact(i)

    for i in range(3+novelty+1, n_starting_urns+1):
        urns[1].add_contact(i)

    return urns

def run_model(model_instance, n_steps, custom_timestep=None, custom_reinforcement=None, print_msg=None):
    csv_rows = []
    for i in range(n_steps):
        if print_msg:
            print(f"Step {i+1}/{n_steps}" + print_msg)
        else:
            print(f"Step {i+1}/{n_steps}\tNo. urns: {model_instance.n_urns}")

        if custom_timestep:
            custom_timestep(t_step=i)
        elif custom_reinforcement:
            model_instance.time_step(alt_reinforcement=custom_reinforcement)
        else:
            model_instance.time_step()

        last_event = model_instance.events[i]
        csv_rows.append((last_event[0], last_event[1], model_instance.n_urns))

    return csv_rows

def run_model_n_urns(model_instance, n_urns):
    csv_rows = []
    i = 0
    while model_instance.n_urns < n_urns:
        print(f"Step {i+1}, No. urns: {model_instance.n_urns}/{n_urns}")

        model_instance.time_step()

        last_event = model_instance.events[i]
        csv_rows.append((last_event[0], last_event[1], model_instance.n_urns))

        i += 1

    return csv_rows

def write_data_to_csv(csv_data, filepath):
    with open(filepath, 'x', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["caller", "receiver", "num_urns"])
        writer.writerows(csv_data)

def pickle_freqrank(df: pd.DataFrame, save_as):
    """
    Takes a one-dimensional dataframe representing some sequence or database and calculates
    the frequency of each unique element in the sequence, storing it in a dictionary format
    and pickling it.

    Params:     `df` - the one-dimensional dataframe sequence
                `save_as` - filename to use in pickling i.e. will create the file 
                            "{save_as}.pickle" 
    """

    pickle_filename = save_as + ".pickle"
    edge_count = 0
    freqs = {}
    for i,x in enumerate(df):
        if x not in freqs:
            freqs[x] = 1
        else:
            freqs[x] += 1

        print(f"Iteration {i}")

    with open(pickle_filename, 'wb') as f:
        pickle.dump(freqs, f)

def get_plottable_freqrank_from_pickle(filepath):
    with open(filepath, 'rb') as f:
        freqs = pickle.load(f)
        freqs = list(freqs.values())
        freqs.sort(reverse=True)
        
        return list(range(len(freqs))), freqs
