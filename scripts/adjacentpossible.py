from typing import Dict
from dataclasses import dataclass
import random
import numpy as np
from math import floor
from datautils import choose_proportional, choose_proportional_lazy

@dataclass
class UserUrn:
    # NOTE comparison methods are based on the list of contacts for ease of sorting,
    # so to check if two urns are the same urn use urn1.ID == urn2.ID
    ID: int
    contacts: Dict # (K,V) = (ID, count)
    n_contacts: int = 0

    def add_contact(self, urn_id):
        # print(f"Adding {urn_id} to {self.ID}")
        if not self.contacts:
            # print("empty")
            self.contacts = {urn_id: 1}
            self.n_contacts += 1
        elif urn_id in self.contacts:
            # not a new contact
            self.contacts[urn_id] += 1
        else:
            # print(f"{urn_id} not in {self.contacts}")
            self.contacts[urn_id] = 1
            self.n_contacts += 1
            # print(f"is now {self.contacts}")

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

    def _get_calling_urn(self, lazy=False):
        urn_sizes = []
        for u in self.urns:
            urn_sizes.append(u.n_contacts) # NOTE: don't +1 to these; shouldn't select empty urns

        return choose_proportional(self.urns, urn_sizes, self.n_urns)

    def time_step(self, lazy=False):
        """
        Performs a single time step of the model according to the set strategy (WSW, by default).
        NOTE: Expects at least one urn to have a non-empty contacts dict
        """
        # get caller and receiver, store event
        caller = self._get_calling_urn(lazy)
        # print(f"CALLER IS {caller}")
        unique_contact_ids = list(caller.contacts.keys())
        unique_contact_counts = list(caller.contacts.values())

        receiver_id = choose_proportional(unique_contact_ids, unique_contact_counts, caller.n_contacts)
        receiver = None
        for urn in self.urns:
            if urn.ID == receiver_id:
                receiver = urn
                break

        # reinforcement step
        receiver_is_empty = False
        if receiver.n_contacts == 0: # so we can do novelty step
            receiver_is_empty = True

        for i in range(self.reinforcement_param):
            caller.add_contact(receiver.ID)
            receiver.add_contact(caller.ID)

        # create v + 1 new nodes if receiver was empty
        if receiver_is_empty:
            for i in range(self.novelty_param + 1):
                new_urn = UserUrn(len(self.urns) + 1, {})
                self.urns.append(new_urn)
                self.n_urns += 1
                receiver.add_contact(new_urn.ID)
                # print(f"Added {new_urn.ID} to {receiver.ID}")

        # novelty step, strategy called here
        # print(f"looking for ({caller.ID}, {receiver.ID}) in {self.events}")
        if (caller.ID, receiver.ID) not in self.events \
            and (receiver.ID, caller.ID) not in self.events: # TODO not efficient, replace with interaction matrix version
            # novelty only occurs if urns never interacted before
            # print(f"{caller.ID} and {receiver.ID} never interacted bvefore")
            if self.strategy == "WSW":
                self._do_WSW(caller, receiver)
            else:
                raise ValueError(f"'{self.strategy}' is not a valid strategy.")

        self.events.append((caller.ID, receiver.ID))

    def _do_WSW(self, caller, receiver):
        # print(f"WSW with C {caller.ID} R {receiver.ID}")
        # print(f"\tC: {caller.contacts}\n\tR: {receiver.contacts}")
        # NOTE we copy all these things as they are now, so that novel urns which
        # are about to be added cannot be chosen accidentally
        caller_ids = list(caller.contacts.keys())
        receiver_ids = list(receiver.contacts.keys())

        caller_contacts = caller.contacts.copy()
        caller_len = caller.n_contacts
        receiver_contacts = receiver.contacts.copy()
        receiver_len = receiver.n_contacts

        # Choose v+1 unique IDs from caller, add to receiver
        num_iter = self.novelty_param + 1
        novel_options = list(set(caller_contacts.keys()) - set(receiver_contacts.keys())) # complement
        # print(f"C - R = {novel_options}")
        if len(novel_options) == 1:
            # TODO hacky fix, is something broken? Without this, sometimes the only novelty to give an urn is itself
            num_iter = 0
        elif len(novel_options) - 1 < self.novelty_param + 1:
            # Supplementary notes say that we pick only v distinct IDs if urn i
            # would otherwise have to give a copy of urn j to urn j itself
            num_iter -= 1
        

        for i in range(num_iter):
            drawn_id = receiver.ID
            while drawn_id is receiver.ID or drawn_id in list(receiver_contacts.keys()):
                # print("C drawing")
                # print(f"\t iter {i}/{num_iter}")
                # print(f"\tR ({receiver.ID}): {receiver_ids}\n\tC ({caller.ID}): {caller_ids}")
                # print(f"\tComplement: {list(set(caller_ids) - set(receiver_ids))}")
                drawn_id = choose_proportional(list(caller_contacts.keys()), \
                    list(caller_contacts.values()), caller_len)
            # otherwise,
            receiver.add_contact(drawn_id)
            caller_contacts.pop(drawn_id)
            caller_len -= 1

        # The same but from receiver, adding to caller
        caller_contacts = caller.contacts.copy()
        num_iter = self.novelty_param + 1
        novel_options = list(set(receiver_contacts.keys()) - set(caller_contacts.keys())) # complement of both sets
        # print(f"R - C = {novel_options}")
        if len(novel_options) == 1:
            # TODO hacky fix, see above
            num_iter = 0
        elif len(novel_options) - 1 < self.novelty_param + 1:
            # Supplementary notes say that we pick only v distinct IDs if urn i
            # would otherwise have to give a copy of urn j to urn j itself
            num_iter -= 1
        

        for i in range(num_iter):
            drawn_id = receiver.ID
            while drawn_id is caller.ID or drawn_id in list(caller_contacts.keys()):
                # print("R drawing")
                # print(f"\t iter {i}/{num_iter}")
                # print(f"\tR ({receiver.ID}): {receiver_ids}\n\tC ({caller.ID}): {caller_ids}")
                # print(f"\tComplement: {novel_options}")
                drawn_id = choose_proportional(list(receiver_contacts.keys()), \
                    list(receiver_contacts.values()), receiver_len)
            # otherwise,
            caller.add_contact(drawn_id)
            receiver_contacts.pop(drawn_id)
            receiver_len -= 1

    def _do_WSW_old(self, caller, receiver):
        # print(f"WSW with C {caller} R {receiver}")
        # NOTE we copy all these things as they are now, so that novel urns which
        # are about to be added cannot be chosen accidentally
        caller_ids = list(caller.contacts.keys())
        caller_counts = list(caller.contacts.values())
        caller_len = caller.n_contacts
        receiver_ids = list(receiver.contacts.keys())
        receiver_counts = list(receiver.contacts.values())
        receiver_len = receiver.n_contacts

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
                print(f"C{i}: {caller_ids} with sizes {caller_counts} and length {caller_len}")
                print(f"R has {receiver_ids}, already chose {already_chosen}")
                drawn_id = choose_proportional(caller_ids, caller_counts, caller_len)
                print(f"Drew {drawn_id}")

            receiver.add_contact(drawn_id)
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
                print(f"R{i}: {receiver_ids} with sizes {receiver_counts} and length {receiver_len}")
                drawn_id = choose_proportional(receiver_ids, receiver_counts, receiver_len)

            caller.add_contact(drawn_id)
            already_chosen.append(drawn_id)
