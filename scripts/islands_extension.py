import random
import sys
sys.path.append("scripts/")

from adjacentpossible import AdjPosModel, UserUrn

class IslandsMdlExt(AdjPosModel):
    """
    This model extension adds a core/non-core 'vocabulary' aspect to the network generation.

    A new non-core urn is defined as the standard creation of new urns; a caller chosen
    proportionally picks an empty receiver from its contents who then generates v+1 new
    urns. 

    A new core urn is defined as both the caller and receiver being new urns. Both generate
    V+1 new urns to share with each other. In this way, "islands" or isolated subgraphs can
    appear in the network, unlike in the original model. These subgraphs can still become
    connected in the future through the standard selection process.
    """
    def __init__(self, prob_core, max_core, rng_seed=None, novelty_param=1, reinforcement_param=2, \
        strategy="WSW", urns=[]):
        self.prob_core = prob_core
        self.max_core = max_core
        self.n_core = 0
        
        self.novelty_param = novelty_param
        self.reinforcement_param = reinforcement_param
        self.strategy = strategy
        self.events = []
        self.urn_sizes = {}
        self.total_size = 0
        self.urns = urns
        self.n_urns = len(urns)
        self.interaction_lookup = {}
        self.prop_choice = [] # for performance in _get_calling_urn()

        for u in self.urns:
            self.urn_sizes[u.ID] = u.size
            self.total_size += u.size
            for i in range(u.size):
                self.prop_choice.append(u.ID)

    def time_step(self):
        r = random.random()
        if r < self.prob_core and self.n_core < self.max_core:
            caller = UserUrn(self.n_urns+1, {})
            self.urns.append(caller)
            self.n_urns += 1

            receiver = UserUrn(self.n_urns+1, {})
            self.urns.append(receiver)
            self.n_urns += 1

            # custom novelty step
            for i in range(self.novelty_param + 1):
                new_urn_c = UserUrn(self.n_urns+1, {})
                self.urns.append(new_urn_c)
                self.urn_sizes[new_urn_c.ID] = 0
                self.n_urns += 1
                caller.add_contact(new_urn_c.ID)
                self.prop_choice.append(caller.ID)
                self.total_size += 1

                new_urn_r = UserUrn(self.n_urns+1, {})
                self.urns.append(new_urn_r)
                self.urn_sizes[new_urn_r.ID] = 0
                self.n_urns += 1
                receiver.add_contact(new_urn_r.ID)
                self.prop_choice.append(receiver.ID)
                self.total_size += 1

            self.urn_sizes[caller.ID] = caller.size
            self.urn_sizes[receiver.ID] = receiver.size

            self.do_reinforcement(caller, receiver)
            
            if self.strategy == "WSW":
                caller_contacts_before = caller.contacts.copy()
                caller_size_before = caller.size
                receiver_contacts_before = receiver.contacts.copy()
                receiver_size_before = receiver.size
                self._do_strat_WSW(caller, receiver, caller_contacts_before, caller_size_before)
                self._do_strat_WSW(receiver, caller, receiver_contacts_before, receiver_size_before)

            self.events.append((caller.ID, receiver.ID))
            self.n_core += 1
        else:
            super().time_step()