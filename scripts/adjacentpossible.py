from typing import Dict
from dataclasses import dataclass
import random
import numpy as np
from math import floor
from datautils import choose_proportional_dict

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

class AdjPosModel:
    def __init__(self, rng_seed=None, novelty_param=1, reinforcement_param=2, strategy="WSW", urns=[]):
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

        # print(self.prop_choice)
        # print(f"ID 1 is {self.prop_choice.count(1)}/{len(self.prop_choice)}")
        # print(f"Or {self.prop_choice.count(1)/len(self.prop_choice):.2f}")

        random.seed(rng_seed)

    def _get_calling_urn(self):
        r = random.randint(0, self.total_size-1)
        return self.prop_choice[r]

    def _do_strat_WSW(self, urn_a, urn_b, urn_a_contacts, urn_a_size):
        # Choose v+1 unique IDs from urn A, add to urn B
        urn_a_size -= urn_a_contacts[urn_b.ID]
        urn_a_contacts.pop(urn_b.ID)
        num_iter = self.novelty_param + 1
        if(urn_a.n_contacts - 1 < num_iter): # may only contain v IDs, not including urn B's ID
            num_iter -= 1
        
        for i in range(num_iter):
            # print(f"iter {i+1}/{num_iter}\nHave: {urn_a_contacts} of size {urn_a_size} (vs. {sum(list(urn_a_contacts.values()))})")
            drawn_id = choose_proportional_dict(urn_a_contacts, urn_a_size)

            # otherwise,
            urn_b.add_contact(drawn_id)
            self.urn_sizes[urn_b.ID] = urn_b.size
            self.prop_choice.append(urn_b.ID)
            self.total_size += 1

            urn_a_size -= urn_a_contacts[drawn_id]
            urn_a_contacts.pop(drawn_id)

    def do_reinforcement(self, caller, receiver):
        for i in range(self.reinforcement_param):
            caller.add_contact(receiver.ID)
            receiver.add_contact(caller.ID)

            self.urn_sizes[caller.ID] = caller.size
            self.prop_choice.append(caller.ID)

            self.urn_sizes[receiver.ID] = receiver.size
            self.prop_choice.append(receiver.ID)

            self.total_size += 2

    def do_novelty(self, caller, receiver):
        """
        Performs the novelty step.

        Returns `True` if this is a first-time interaction and thus whether the strategy step
        should take place.
        """
        # create v + 1 new nodes if receiver was empty
        if receiver.n_contacts == 0:
            for i in range(self.novelty_param + 1):
                new_urn = UserUrn(self.n_urns+1, {})
                self.urns.append(new_urn)
                self.urn_sizes[new_urn.ID] = 0
                self.n_urns += 1
                receiver.add_contact(new_urn.ID)
                self.urn_sizes[receiver.ID] = receiver.size
                self.prop_choice.append(receiver.ID)
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

    def time_step(self, begin_ext=None, mid_ext=None, strat_ext=None, end_ext=None, alt_reinforcement=None):
        """
        Performs a single time step of the model according to the set strategy (WSW, by default).
        NOTE: Expects at least one urn to have a non-empty contacts dict.

        Model extensions can be added through the various `..._ext` parameters, which take a
        function object to call at various points throughout the model evolution process, as
        follows:
            - `begin_ext`:  Before novelty, reinforcement, and strategy steps (but the caller
                            has been chosen).
            - `mid_ext`:    Called after novelty and reinforcement steps but before the strategy
                            step. Unlike `strat_ext` this extension is always called.
            - `strat_ext`:  Same as `mid_ext` but is only called if the strategy is to be called
                            too (i.e. if this is a first-time interaction).
            - `end_ext`:    After novelty, reinforcement, and strategy steps.
        """
        # choose caller and receiver
        caller_id = self._get_calling_urn()
        caller = self.urns[caller_id-1]
        receiver_id = caller.extract_prop()
        receiver = self.urns[receiver_id-1]

        if begin_ext:
            caller_id, receiver_id = begin_ext(caller, receiver)
            caller = self.urns[caller_id-1]
            receiver = self.urns[receiver_id-1]

        is_first_interaction = self.do_novelty(caller, receiver)

        if alt_reinforcement:
            alt_reinforcement(self, caller, receiver)
        else:
            self.do_reinforcement(caller, receiver)

        if mid_ext:
            caller_id, receiver_id = mid_ext(self, caller, receiver)
            caller = self.urns[caller_id-1]
            receiver = self.urns[receiver_id-1]

        if is_first_interaction:
            if strat_ext:
                caller_id, receiver_id = strat_ext(self, caller, receiver)
                caller = self.urns[caller_id-1]
                receiver = self.urns[receiver_id-1]

            if self.strategy == "WSW":
                caller_contacts_before = caller.contacts.copy()
                caller_size_before = caller.size
                receiver_contacts_before = receiver.contacts.copy()
                receiver_size_before = receiver.size

                self._do_strat_WSW(caller, receiver, caller_contacts_before, caller_size_before)
                self._do_strat_WSW(receiver, caller, receiver_contacts_before, receiver_size_before)

        # store event
        self.events.append((caller.ID, receiver.ID))

        if end_ext:
            end_ext(caller, receiver)
