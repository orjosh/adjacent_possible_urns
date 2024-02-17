import sys
sys.path.append("scripts/")

import random
from adjacentpossible import AdjPosModel, UserUrn

class SuggestorMdlExt:
    """
    This model extension adds a "core vocabulary" to the network in order to mimic the model
    described in Gerlach and Altmann PRX 2013 (DOI 10.1103/PhysRevX.3.021006). The core 
    vocabulary consists of the largest urns (most connected users) in the network. At each 
    evolution step, there is a chance that a calling urn will select one of these largest 
    urns as a receiver instead of drawing one proportionally from its own contents/connections.

    NOTE: This extension does NOT incorporate the update rule for p_new described in the paper.
    It is currently just an attempt to add the idea of a 'core vocabulary'.
    """
    def __init__(self, prob_to_suggest=0.5, size_threshold=1, max_suggestable=1):
        self.suggestable = {}
        self.n_suggestable = 0
        self.prob_sugg = prob_to_suggest
        self.size_threshold = size_threshold
        self.max_sugg = max_suggestable

        self.caller_size_bef = 0
        self.receiver_size_bef = 0

    # def get_sizes_before_begin_ext(self, caller: UserUrn, receiver: UserUrn):
    #     self.caller_size_bef = caller.size
    #     self.receiver_size_bef = receiver.size

    def do_suggestion_begin_ext(self, caller: UserUrn, receiver: UserUrn):
        caller_id = caller.ID
        receiver_id = receiver.ID

        # randomly choose whether or not to perform the suggestion
        r = random.random()
        if r < self.prob_sugg and self.n_suggestable > 0:
            # pick a random urn from the list of suggestions
            r = random.randint(0, self.n_suggestable-1)
            receiver_id = list(self.suggestable.keys())[r]
        return caller_id, receiver_id

    def check_suggestable_end_ext(self, caller: UserUrn, receiver: UserUrn):
        print(self.suggestable)
        if self.n_suggestable == self.max_sugg:
            return
        
        if caller.size >= self.size_threshold \
            and caller.ID not in self.suggestable:
            self.suggestable[caller.ID] = 1
            self.n_suggestable += 1
        
        if receiver.size >= self.size_threshold \
            and receiver.ID not in self.suggestable:
            self.suggestable[receiver.ID] = 1
            self.n_suggestable += 1