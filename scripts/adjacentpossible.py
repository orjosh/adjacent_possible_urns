from typing import Dict
from dataclasses import dataclass
import random
import numpy as np
from datautils import choose_proportional

@dataclass
class UserUrn:
    # NOTE comparison methods are based on the list of contacts for ease of sorting,
    # so to check if two urns are the same urn use urn1.ID == urn2.ID
    ID: int
    contacts: Dict # (K,V) = (ID, count)
    n_contacts: int = 0

    def add_contact(self, urn):
        if self.contacts is []:
            self.contacts = {urn.ID: 1}
        elif urn.ID in self.contacts:
            self.contacts[urn.ID] += 1
        else:
            self.contacts[urn.ID] = 1

        self.n_contacts += 1

    def extract_prop(self):
        freqs = []
        ids = []
        for k, v in self.contacts:
            ids.append(k)
            freqs.append(v)

        return choose_proportional(ids, freqs, self.n_contacts)

    def __eq__(self, urn):
        return len(self.contacts) == len(urn.contacts)

    def __lt__(self, urn):
        return len(self.contacts) < len(urn.contacts)

    def __gt__(self, urn):
        return len(self.contacts) > len(urn.contacts)

class AdjPosModel:
    def __init__(self, rng_seed=None, novelty_param=0, reinforcement_param=1, strategy="WSW", urns=None):
        self.novelty_param = novelty_param
        self.reinforcement_param = reinforcement_param
        self.strategy = strategy
        self.events = []
        self.interaction_mat = np.zeros((len(urns), len(urns)))
        if urns is None:
            self.urns = []
        else:
            self.urns = urns
        self.n_urns = len(urns)

        random.seed(rng_seed)

    def _get_calling_urn(self):
        urn_sizes = []
        for u in self.urns:
            urn_sizes.append(u.n_contacts)

        return choose_proportional(self.urns, urn_sizes, self.n_urns)

    def time_step(self):
        """
        Performs a single time step of the model according to the set strategy (WSW, by default).
        NOTE: Expects at least one urn to have a non-empty contacts dict
        """
        # get caller and receiver, store event
        caller = self._get_calling_urn()
        unique_contact_ids = list(caller.contacts.keys())
        unique_contact_counts = list(caller.contacts.values())

        for i, x in enumerate(unique_contact_counts):
            # we do this so that empty urns can be selected as receivers too
            unique_contact_counts[i] += 1

        receiver_id = choose_proportional(unique_contact_ids, unique_contact_counts, caller.n_contacts)
        receiver = None
        for urn in self.urns:
            if urn.ID == receiver_id:
                receiver = urn
        self.events.append((caller.ID, receiver.ID))
        #print(self.events)

        # reinforcement step
        receiver_is_empty = False
        caller.contacts[receiver.ID] += self.reinforcement_param
        if len(receiver.contacts) == 0: # receiver may be empty
            receiver_is_empty = True
            receiver.contacts[caller.ID] = self.reinforcement_param
        else:
            receiver.contacts[caller.ID] += self.reinforcement_param

        # create v + 1 new nodes if receiver was empty
        if receiver_is_empty:
            for i in range(self.novelty_param + 1):
                new_urn = UserUrn(len(self.urns) + 1, {})
                self.urns.append(new_urn)
                self.n_urns += 1
                receiver.contacts[new_urn.ID] = 1

        # novelty step, strategy called here
        if (caller.ID, receiver.ID) not in self.events \
            and (receiver.ID, caller.ID) not in self.events: # TODO not efficient, replace with interaction matrix version
            # novelty only occurs if urns never interacted before
            if self.strategy == "WSW":
                # TODO ids and counts of caller previously calculated (unique_contact_...), no point re-calculating?
                self._do_WSW(caller, receiver)
            else:
                raise ValueError(f"'{self.strategy}' is not a valid strategy.")

    def _do_WSW(self, caller, receiver):
        caller_ids = list(caller.contacts.keys())
        caller_counts = list(caller.contacts.values())
        receiver_ids = list(receiver.contacts.keys())
        receiver_counts = list(receiver.contacts.values())

        # choose v + 1 unique IDs proportionally from caller, add to receiver
        already_chosen = []

        if(len(caller_ids) - 1 < self.novelty_param + 1):
            # - 1 because caller_ids contains the receiver too
            raise ValueError(f"Caller ID {caller.ID} doesn't contain enough contacts to meet novelty requirement.")
        
        for i in range(self.novelty_param + 1):
            drawn_id = receiver.ID
            while drawn_id == receiver.ID or drawn_id in already_chosen \
                or drawn_id in receiver_ids:
                # must pick a unique ID each time, and also can't give an
                # urn its own ID
                drawn_id = choose_proportional(caller_ids, caller_counts, caller.n_contacts)
            if drawn_id in receiver.contacts:
                receiver.contacts[drawn_id] += 1
            else:
                receiver.contacts[drawn_id] = 1
            already_chosen.append(drawn_id)

        # vice versa
        already_chosen = []

        # TODO does receiver_ids necessarily contain caller? Do we need that -1?
        if(len(receiver_ids) - 1 < self.novelty_param + 1):
            raise ValueError(f"Receiver ID {receiver.ID} doesn't contain enough contacts to meet novelty requirement.")

        for i in range(self.novelty_param + 1):
            drawn_id = caller.ID
            while drawn_id == caller.ID or drawn_id in already_chosen \
                or drawn_id in caller_ids:
                drawn_id = choose_proportional(receiver_ids, receiver_counts, receiver.n_contacts)
            if drawn_id in caller.contacts:
                caller.contacts[drawn_id] += 1
            else:
                caller.contacts[drawn_id] = 1
            already_chosen.append(drawn_id)
