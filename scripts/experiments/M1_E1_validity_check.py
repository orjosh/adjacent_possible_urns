import sys
sys.path.append("scripts/")
import csv
from os.path import exists
from adjacentpossible import AdjPosModel, UserUrn

if not exists("M1_E1.csv"):
    # Model Setup
    novelty = 5
    reinforcement = 5
    strategy = "WSW"
    n_steps = 10**6
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
        csv_rows.append(model.events[i])

    filename = "M1_E1.csv"
    with open(filename, 'x', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["caller", "receiver"])
        writer.writerows(csv_rows)
    
