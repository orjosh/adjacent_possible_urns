import sys
sys.path.append("scripts/")

import random
from adjacentpossible import AdjPosModel, UserUrn

class ScoreboardQueue:
    """
    Data structure similar to a queue which stores only the n "best" elements
    in order, based on some numerical metric.
    """
    def __init__(self, size):
        self.n = 0
        self.max_size = size
        self.id_size_dict = {}
        self.scoreboard = [] # tuples of (id, score)

    def update_size(self, element_id, new_size):
        new_item = False
        if element_id not in self.id_size_dict:
            self.n += 1
            new_item = True
        
        # print(f"Trying to update {element_id} of size {new_size}")
        self.id_size_dict[element_id] = new_size
        self.sort()

        if new_item and self.n > self.max_size:
            # We have too many, need to prune the smallest element
            last_id = self.scoreboard[-1][0]
            self.id_size_dict.pop(last_id)
            self.scoreboard.pop(-1)
            self.n -= 1

    def sort(self):
        size_dict_items = self.id_size_dict.items()
        # print(f"Was {self.scoreboard}")
        self.scoreboard = sorted(size_dict_items, key=lambda x: x[1], reverse=True)
        # print(f"Now {self.scoreboard}")

    def get_scoreboard_as_list(self):
        result = []
        for x in self.scoreboard:
            result.append(x[0])

        return result


class InitialSuggestionsMdlExt:
    """
    This model extension makes it so that all empty urns will start off with the IDs of the
    nu + 1 largest urns in the network in addition to any regular procedures.
    """
    def __init__(self, n_to_suggest, ls_urns):
        self.suggestions = ScoreboardQueue(n_to_suggest)

        scoreboard = []
        for u in ls_urns:
            self.suggestions.update_size(u.ID, u.size)
            # self.suggestions.id_size_dict[u.ID] = u.size
        self.suggestions.sort()

    def populate_list_begin_ext(self, model: AdjPosModel, caller, receiver):
        if self.suggestions.n < self.suggestions.max_size:
            for u in model.urns:
                self.suggestions.update_size(u.ID, u.size)

        return caller.ID, receiver.ID

    def custom_novelty_step(self,  model: AdjPosModel, caller, receiver):
        """
        Returns `True` if this is a first-time interaction and thus whether the strategy step
        should take place.
        """
        if receiver.n_contacts == 0:
            # standard novelty procedure
            for i in range(model.novelty_param + 1):
                new_urn = UserUrn(model.n_urns+1, {})
                model.urns.append(new_urn)
                model.urn_sizes[new_urn.ID] = 0
                model.n_urns += 1
                receiver.add_contact(new_urn.ID)
                model.urn_sizes[receiver.ID] = receiver.size
                model.prop_choice.append(receiver.ID)
                model.total_size += 1

            # new part
            suggested_ids = self.suggestions.get_scoreboard_as_list()
            for u in suggested_ids:
                # receiver.contacts[u] = 1 # NOTE import distinction here, not increasing n_contacts, just exposing her to u
                # print(f"Adding {u} to {receiver.ID}\n{receiver}")
                receiver.add_contact(u)
                model.urn_sizes[receiver.ID] = receiver.size
                model.prop_choice.append(receiver.ID)
                model.total_size += 1

        # check if first-time interaction
        lower = None
        higher = None
        if caller.ID < receiver.ID:
            lower = caller.ID
            higher = receiver.ID
        else:
            lower = receiver.ID
            higher = caller.ID

        if model.interaction_lookup.get((lower, higher)) is None:
            model.interaction_lookup[(lower, higher)] = 1
            return True
    
    def update_suggestions(self, caller, receiver):
        self.suggestions.update_size(caller.ID, caller.size)
        self.suggestions.update_size(receiver.ID, receiver.size)

        # print(f"\tSuggestions: {self.suggestions.get_scoreboard_as_list()}")
        print(f"Scoreboard is of size {len(self.suggestions.scoreboard)}")
        print(f"1st: {self.suggestions.scoreboard[0]}, 2nd: {self.suggestions.scoreboard[1]}")