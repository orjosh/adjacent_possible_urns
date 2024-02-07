import csv
import sys

sys.path.append("scripts/") # assumes repo cwd

from adjacentpossible import UserUrn, AdjPosModel

# In this sub-experiment, I change the ratio R = reinforcement / novelty
# half-way through the sequence

urn1 = UserUrn(1, {})
urn2 = UserUrn(2, {})
urn3 = UserUrn(3, {})
urn4 = UserUrn(4, {})
urn5 = UserUrn(5, {})
urn6 = UserUrn(6, {})
urn7 = UserUrn(7, {})
urn8 = UserUrn(8, {})

urn1.add_contact(urn2.ID)
urn1.add_contact(urn4.ID)
urn1.add_contact(urn6.ID)
urn1.add_contact(urn8.ID)

urn2.add_contact(urn1.ID)
urn2.add_contact(urn3.ID)
urn2.add_contact(urn5.ID)
urn2.add_contact(urn7.ID)

starting_urns = [urn1, urn2, urn3, urn4, urn5, urn6, urn7, urn8]

strat = "WSW"
novelty = 1
reinforcement = 2
model = AdjPosModel(novelty_param=novelty, reinforcement_param=reinforcement, \
    strategy=strat, urns=starting_urns)

n_steps = int(10**6)
t_change_ratio = int(10**4)
csv_rows = []
for i in range(n_steps):
    print(f"Step {i+1}/{n_steps}\tR={model.reinforcement_param/model.novelty_param}\tChange at {t_change_ratio}")
    #print(f"Urns: {model.urn_sizes}")
    if i == t_change_ratio:
        model.novelty_param = 3
    model.time_step()
    csv_rows.append((model.events[i][0], model.events[i][1], model.n_urns))

filename = "adjpos_E01_2.csv"
with open(filename, 'x', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["caller", "receiver", "n_distinct"])
    writer.writerows(csv_rows)
