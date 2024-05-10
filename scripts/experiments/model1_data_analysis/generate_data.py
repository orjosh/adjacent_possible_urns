import sys
sys.path.append("scripts/")
from functools import partial
from adjacentpossible import AdjPosModel
import simfuncs as sim

# Hypothesis: Novelty dictates the addition of new urns, and reinforcement dictates the skewness
# of the interaction frequencies (i.e. higher reinforcement means larger spread / larger difference
# in interaction counts). These are partially at-odds since a higher novelty means more urns to
# pick from and thus any given urn has a smaller chance of being drawn. Therefore, I expect
# gamma to increase with rho/nu.

#DESIRED_N_URNS = int(3 * 10**6)
N_STEPS = int(5*10**6)
REINFORCEMENT = 4
NOVELTY = 12

CSV_SAVE_PATH = "generated_data/adjpos_orig_wsw_rho_change_2.csv"

# def changing_rho_timestep(model_instance, t_step, rho_vals, step_start):
#     rho = rho_vals[0]

#     for i, t in enumerate(step_start):
#         if t_step >= t:
#             rho = rho_vals[i]

#     model_instance.reinforcement_param = rho
#     print(f"rho = {model_instance.reinforcement_param}")
#     model_instance.time_step()

def degree_based_reinforcement_step(model, caller, receiver):
    # Reinforcement is equal to node's degree + rho, i.e. in interaction (A, B)
    # A will get deg(B) + rho copies of B and vice versa
    for i in range(model.reinforcement_param + receiver.n_contacts):
        model._add_contact_update_size(caller, receiver.ID)

    for i in range(model.reinforcement_param + caller.n_contacts):
        model._add_contact_update_size(receiver, caller.ID)

starting_urns = sim.generate_initial_urns(NOVELTY)
model = AdjPosModel(novelty_param=NOVELTY, reinforcement_param=REINFORCEMENT, urns=starting_urns)

csv_data = sim.run_model(model, N_STEPS, custom_reinforcement=degree_based_reinforcement_step)

sim.write_data_to_csv(csv_data, CSV_SAVE_PATH)