from typing import Dict
from dataclasses import dataclass
import random
import numpy as np
from math import floor
from datautils import choose_proportional, choose_proportional_dict

@dataclass
class UserUrn:
    # NOTE comparison methods are based on the list of contacts for ease of sorting,
    # so to check if two urns are the same urn use urn1.ID == urn2.ID
    ID: int
    contacts: Dict # (K,V) = (ID, count)
    n_contacts: int = 0
    size: int = 0 # number of 'balls', different from unique n_contacts

    def add_contact(self, urn_id):
        if not self.contacts:
            self.contacts = {urn_id: 1}
            self.n_contacts += 1
        elif urn_id in self.contacts:
            # not a new contact
            self.contacts[urn_id] += 1
        else:
            self.contacts[urn_id] = 1
            self.n_contacts += 1
        self.size += 1

    def extract_prop(self):
        return choose_proportional_dict(self.contacts, self.size)

    def __eq__(self, urn):
        return len(self.contacts) == len(urn.contacts)

    def __lt__(self, urn):
        return len(self.contacts) < len(urn.contacts)

    def __gt__(self, urn):
        return len(self.contacts) > len(urn.contacts)

class AdjPosModel:
    def __init__(self, rng_seed=None, novelty_param=1, reinforcement_param=2, strategy="WSW", urns=[]):
        self.novelty_param = novelty_param
        self.reinforcement_param = reinforcement_param
        self.strategy = strategy
        self.events = []
        self.urn_sizes = {}
        self.total_size = 0
        self.urns = urns
        for u in self.urns:
            self.urn_sizes[u.ID] = u.size
            self.total_size += u.size
        self.n_urns = len(urns)
        self.interaction_lookup = {}

        random.seed(rng_seed)

    def _get_calling_urn(self):
        caller_id = choose_proportional_dict(self.urn_sizes, self.total_size)
        return self.urns[caller_id-1]

    def _do_strat_WSW(self, urn_a, urn_b):
        # NOTE we copy all these things as they are now, so that novel urns which
        # are about to be added cannot be chosen accidentally
        urn_a_contacts = urn_a.contacts.copy()
        #print(f"Urn A: {urn_a}")
        #print(f"Urn B: {urn_b}")

        # Choose v+1 unique IDs from urn A, add to urn B
        urn_a_contacts.pop(urn_b.ID)
        n_unique_left = urn_a.n_contacts - 1 # because we removed B's ID which we can't share to B
        num_iter = self.novelty_param + 1
        #print(f"{n_unique_left} vs. {num_iter}")
        if(n_unique_left < num_iter):
            #print(f"{n_unique_left} vs. {num_iter}")
            num_iter -= 1
        
        for i in range(num_iter):
            #print(f"iter {i+1}/{num_iter}\nLeft: {urn_a_contacts} (n={n_unique_left})")
            drawn_id = choose_proportional_dict(urn_a_contacts, n_unique_left)

            # otherwise,
            #print(f"drew {drawn_id}")
            urn_b.add_contact(drawn_id)
            self.urn_sizes[urn_b.ID] = urn_b.size
            self.total_size += 1
            urn_a_contacts.pop(drawn_id)
            n_unique_left -= 1

    def _do_reinforcement(self, caller, receiver):
        for i in range(self.reinforcement_param):
            caller.add_contact(receiver.ID)
            receiver.add_contact(caller.ID)
            self.urn_sizes[caller.ID] = caller.size
            self.urn_sizes[receiver.ID] = receiver.size
            self.total_size += 2

    def _do_novelty(self, caller, receiver):
        """
        Performs the novelty step.

        Returns `True` if this is a first-time interaction and thus whether the strategy step
        should take place.
        """
        # create v + 1 new nodes if receiver was empty
        if receiver.n_contacts == 0:
            for i in range(self.novelty_param + 1):
                new_urn = UserUrn(len(self.urns) + 1, {})
                self.urns.append(new_urn)
                self.urn_sizes[new_urn.ID] = 0
                self.n_urns += 1
                receiver.add_contact(new_urn.ID)
                self.urn_sizes[receiver.ID] = receiver.size
                self.total_size += 1

        # check if first-time interaction
        lower = None
        higher = None
        if caller.ID < receiver.ID:
            lower = caller.ID
            higher = receiver.ID
        else:
            lower = receiver.ID
            higher = caller.ID

        if self.interaction_lookup.get((lower, higher)) is None:
            self.interaction_lookup[(lower, higher)] = 1
            return True

    def time_step(self, start_extension=None, end_extension=None):
        """
        Performs a single time step of the model according to the set strategy (WSW, by default).
        NOTE: Expects at least one urn to have a non-empty contacts dict.

        Model extensions can be added through the `start_extension` and `end_extension` parameters,
        which are function objects that are called at the start (before reinforcement, novelty,
        and strategy steps) and end (after reinforcement, novelty, and strategy steps)
        respectively.
        """
        # choose caller and receiver
        caller = self._get_calling_urn()
        receiver_id = caller.extract_prop()
        receiver = None
        for urn in self.urns:
            if urn.ID == receiver_id:
                receiver = urn
                break

        #print(f"C: {caller}, R: {receiver}")
        is_first_interaction = self._do_novelty(caller, receiver)
        self._do_reinforcement(caller, receiver) 
        if is_first_interaction:
            if self.strategy == "WSW":
                self._do_strat_WSW(caller, receiver)
                self._do_strat_WSW(receiver, caller)

        # store event
        self.events.append((caller.ID, receiver.ID))
