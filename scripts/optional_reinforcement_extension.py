import sys
sys.path.append("scripts/")

import random
from adjacentpossible import AdjPosModel, UserUrn

class OptnReinforcementMdlExt:
    """
    This model extension adds a "core vocabulary" to the network in order to mimic the model
    described in Gerlach and Altmann PRX 2013 (DOI 10.1103/PhysRevX.3.021006). In this
    extension, urns have a chance upon a novel interaction to execute the regular
    reinforcement step where p copies of urn i are placed in urn j and vice versa.
    These urns are "core" urns. Otherwise, only a single copy of the IDs are placed
    in each urn (i.e. p = 1).
    """
    def __init__(self, prob_core, max_core):
        self.n_core = 0
        self.prob_core = prob_core
        self.max_core = max_core

    def do_optional_reinforcement(self, model: AdjPosModel, caller: UserUrn, receiver: UserUrn):
        # Called after the normal reinforcement step
        r = random.random()
        if r < self.prob_core and self.n_core < self.max_core:
            model.do_reinforcement(caller, receiver)
            self.n_core += 1
        else:
            caller.add_contact(receiver.ID)
            receiver.add_contact(caller.ID)

            model.urn_sizes[caller.ID] = caller.size
            model.prop_choice.append(caller.ID)

            model.urn_sizes[receiver.ID] = receiver.size
            model.prop_choice.append(receiver.ID)

            model.total_size += 2
