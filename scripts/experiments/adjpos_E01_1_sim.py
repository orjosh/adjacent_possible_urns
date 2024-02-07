import csv
import sys

sys.path.append("scripts/") # assumes repo cwd

from adjacentpossible import UserUrn, AdjPosModel

# We start with six urns; two main urns who are connected to each other and which
# have two additional 'novel' urns each, unknown to the other.

urn1 = UserUrn(1, {})
urn2 = UserUrn(2, {})
urn3 = UserUrn(3, {})
urn4 = UserUrn(4, {})
urn5 = UserUrn(5, {})
urn6 = UserUrn(6, {})

urn1.add_contact(urn2.ID)
urn1.add_contact(urn4.ID)
urn1.add_contact(urn6.ID)

urn2.add_contact(urn1.ID)
urn2.add_contact(urn3.ID)
urn2.add_contact(urn5.ID)

starting_urns = [urn1, urn2, urn3, urn4, urn5, urn6]

strat = "WSW"
novelty = 1
reinforcement = 2
model = AdjPosModel(novelty_param=novelty, reinforcement_param=reinforcement, \
    strategy=strat, urns=starting_urns)

n_steps = 1000000
for i in range(n_steps):
    model.time_step()
    print(f"Step {i+1}/{n_steps}")

filename = "adjpos_E01_1.csv"
with open(filename, 'x', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["caller", "receiver"])
    writer.writerows(model.events)
