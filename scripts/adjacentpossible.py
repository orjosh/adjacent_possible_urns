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
        return choose_proportional(list(self.contacts.keys()), list(self.contacts.values()), self.size)

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

    def time_step(self):
        """
        Performs a single time step of the model according to the set strategy (WSW, by default).
        NOTE: Expects at least one urn to have a non-empty contacts dict
        """
        # get caller and receiver, store event
        caller = self._get_calling_urn()
        receiver_id = caller.extract_prop()
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
            self.urn_sizes[caller.ID] = caller.size
            self.urn_sizes[receiver.ID] = receiver.size
            self.total_size += 2

        # create v + 1 new nodes if receiver was empty
        if receiver_is_empty:
            for i in range(self.novelty_param + 1):
                new_urn = UserUrn(len(self.urns) + 1, {})
                self.urns.append(new_urn)
                self.urn_sizes[new_urn.ID] = 0
                self.n_urns += 1
                receiver.add_contact(new_urn.ID)
                self.urn_sizes[receiver.ID] = receiver.size
                self.total_size += 1

        # novelty step, strategy called here
        lower = None
        higher = None
        if caller.ID < receiver.ID:
            lower = caller.ID
            higher = receiver.ID
        else:
            lower = receiver.ID
            higher = caller.ID

        if self.interaction_lookup.get((lower, higher)) is None:
            # novelty only occurs if urns never interacted before
            if self.strategy == "WSW":
                self._do_WSW(caller, receiver)
                self._do_WSW(receiver, caller)
            else:
                raise ValueError(f"'{self.strategy}' is not a valid strategy.")

        # store event
        self.events.append((caller.ID, receiver.ID))

        # also store in lookup table (for performance)
        self.interaction_lookup[(lower, higher)] = 1

    def _do_WSW(self, urn_a, urn_b):
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

    def _do_WSW_old1(self, caller, receiver):
        # NOTE we copy all these things as they are now, so that novel urns which
        # are about to be added cannot be chosen accidentally
        caller_ids = list(caller.contacts.keys())
        receiver_ids = list(receiver.contacts.keys())

        caller_contacts = caller.contacts.copy()
        caller_len = caller.n_contacts
        receiver_contacts = receiver.contacts.copy()
        receiver_len = receiver.n_contacts

        # Choose v+1 unique IDs from caller, add to receiver
        #print(f"Caller is {caller}")
        #print(f"Receiver is {receiver}")
        num_iter = self.novelty_param + 1
        novel_ids = list(set(caller_contacts.keys()) - set(receiver_contacts.keys())) # complement
        num_novel = len(novel_ids)   
        if num_novel == 1:
            # TODO hacky fix, is something broken? Without this, sometimes the only novelty to give an urn is itself
            num_iter = 0
        elif num_novel - 1 < self.novelty_param + 1:
            # Supplementary notes say that we pick only v distinct IDs if urn i
            # would otherwise have to give a copy of urn j to urn j itself
            num_iter -= 1
        
        novel_counts = []
        for u in novel_ids:
            print(f"Have {u}, has count {caller_contacts[u]} in {caller}")
            novel_counts.append(caller_contacts[u])

        novel_options = dict(zip(novel_ids, novel_counts))       
        
        for i in range(num_iter):
            print(f"iter {i+1}/{num_iter}")
            print(f"Options: {novel_options}")
            drawn_id = receiver.ID
            while drawn_id is receiver.ID or drawn_id in list(receiver_contacts.keys()):
                drawn_id = choose_proportional_dict(novel_options, num_novel)
                #print(f"Drew {drawn_id}")
            # otherwise,
            receiver.add_contact(drawn_id)
            self.urn_sizes[receiver.ID] = receiver.size
            self.total_size += 1
            novel_options.pop(drawn_id)
            num_novel -= 1

        # The same but from receiver, adding to caller
        caller_contacts = caller.contacts.copy()
        num_iter = self.novelty_param + 1
        novel_ids = list(set(receiver_contacts.keys()) - set(caller_contacts.keys())) # complement
        num_novel = len(novel_ids)   
        if num_novel == 1:
            # TODO hacky fix, is something broken? Without this, sometimes the only novelty to give an urn is itself
            num_iter = 0
        elif num_novel - 1 < self.novelty_param + 1:
            # Supplementary notes say that we pick only v distinct IDs if urn i
            # would otherwise have to give a copy of urn j to urn j itself
            num_iter -= 1
        
        novel_sizes = []
        for u in novel_ids:
            novel_sizes.append(self.urn_sizes[u])

        novel_options = dict(zip(novel_ids, novel_sizes))
        
        for i in range(num_iter):
            print(f"iter {i+1}/{num_iter}")
            drawn_id = receiver.ID
            while drawn_id is caller.ID or drawn_id in list(caller_contacts.keys()):
                drawn_id = choose_proportional_dict(novel_options, num_novel)
                #print(f"Drew {drawn_id}")
            # otherwise,
            #print(f"(R) DREW {drawn_id}\t iter {i+1}/{num_iter}")
            caller.add_contact(drawn_id)
            self.urn_sizes[caller.ID] = caller.size
            self.total_size += 1
            novel_options.pop(drawn_id)
            num_novel -= 1

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
                #print(f"C{i}: {caller_ids} with sizes {caller_counts} and length {caller_len}")
                #print(f"R has {receiver_ids}, already chose {already_chosen}")
                drawn_id = choose_proportional(caller_ids, caller_counts, caller_len)
                #print(f"Drew {drawn_id}")

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
