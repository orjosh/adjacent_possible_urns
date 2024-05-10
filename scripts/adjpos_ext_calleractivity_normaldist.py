import sys
sys.path.append("scripts/")
from adjacentpossible import AdjPosModel, UserUrn

class AdjPosCANDModel(AdjPosModel):
    """
    CAND short for Caller Activity Normally Distributed. As it says on the tin, this model takes the
    original Adj Pos model and modifies it so that the caller is not chosen proportionally but instead
    each urn has an "activity" level which determines their chance of being called upon. In this case,
    the order of magnitude of the activity is normally distributed. Receivers are still chosen
    proportionally from the calling urn. 
    """
    def __init__(self, rng_seed=None, novelty_param=1, reinforcement_param=2, strategy="WSW", urns=[]):
        super().__init__(rng_seed, novelty_param, reinforcement_param, strategy, urns)

        self.activity_levels = {}

    # Modified to not update self.prop_choice which is not needed anymore
    def _add_contact_update_size(self, urn1, urn2_id):
        urn1.add_contact(urn2_id)
        super().urn_sizes[urn1.ID] = urn1.size
        super().total_size += 1

    def _get_calling_urn(self):