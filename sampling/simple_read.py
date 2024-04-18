import numpy as np

with open('./data_sb_controller.npy', 'rb') as f:
    state = np.load(f)
    reference = np.load(f)
    input = np.load(f)
    external_disturbance = np.load(f)



# HOW DID I SAVED THING??
# temp_state =  np.zeros((1, 25))
# temp_state[0, 0] = state_current["position"][0]
# temp_state[0, 1] = state_current["position"][1]
# temp_state[0, 2] = state_current["position"][2] 
# temp_state[0, 3:6] = state_current["linear_velocity"][0:3] 
# temp_state[0, 6] = state_current["orientation"][0] 
# temp_state[0, 7] = state_current["orientation"][1] 
# temp_state[0, 8] = state_current["orientation"][2]
# temp_state[0, 9] = state_current["angular_velocity"][0]
# temp_state[0, 10] = state_current["angular_velocity"][1]
# temp_state[0, 11] = state_current["angular_velocity"][2]
# temp_state[0, 12:15] = state_current["foot_FL"]
# temp_state[0, 15:18] = state_current["foot_FR"]
# temp_state[0, 18:21] = state_current["foot_RL"]
# temp_state[0, 21:24] = state_current["foot_RR"]
# temp_state[0, 24] = pgg.step_freq
# data_state.append(copy.deepcopy(temp_state))


# temp_ref = np.zeros((1, 22))
# temp_ref[0, 0] = reference_state["ref_position"][2]
# temp_ref[0, 1:4] = reference_state["ref_linear_velocity"][0:3]
# temp_ref[0, 4] = reference_state["ref_orientation"][0]
# temp_ref[0, 5] = reference_state["ref_orientation"][1]
# temp_ref[0, 6] = reference_state["ref_angular_velocity"][0]
# temp_ref[0, 7] = reference_state["ref_angular_velocity"][1]
# temp_ref[0, 8] = reference_state["ref_angular_velocity"][2]
# temp_ref[0, 9:12] = reference_state["ref_FL_foot"]
# temp_ref[0, 12:15] = reference_state["ref_FR_foot"]
# temp_ref[0, 15:18] = reference_state["ref_RL_foot"]
# temp_ref[0, 18:21] = reference_state["ref_RR_foot"]
# temp_ref[0, 21] = 1.5 #nominal step frequency
# data_reference.append(copy.deepcopy(temp_ref))


# temp_input = np.zeros((1, 24))
# temp_input[0, 0:3] = nmpc_GRFs[0:3]
# temp_input[0, 3:6] = nmpc_GRFs[3:6]
# temp_input[0, 6:9] = nmpc_GRFs[6:9]
# temp_input[0, 9:12] = nmpc_GRFs[9:12]
# temp_input[0, 12:15] = tau_FL
# temp_input[0, 15:18] = tau_FR
# temp_input[0, 18:21] = tau_RL
# temp_input[0, 21:24] = tau_RR
# data_input.append(copy.deepcopy(temp_input))


# temp_disturbance = np.zeros((1, 7))
# temp_disturbance[0, 0:6] = disturbance_wrench
# temp_disturbance[0, 6] = start_disturbance_boolean
# data_external_disturbance.append(copy.deepcopy(temp_disturbance))