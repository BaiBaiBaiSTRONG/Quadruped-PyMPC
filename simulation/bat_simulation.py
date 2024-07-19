# Description: This script is used to simulate the full model of the robot in mujoco

# Authors:
# - Giulio Turrisi

import os
import time

# TODO: Ugly hack so people dont have to run the python command specifying the working directory.
#  we should remove this before the final release.
import sys

# Add the parent directory of this script to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import numpy as np

import config as cfg

# Parameters for both MPC and simulation
from helpers.foothold_reference_generator import FootholdReferenceGenerator
from helpers.periodic_gait_generator import PeriodicGaitGenerator
from helpers.srb_inertia_computation import SrbInertiaComputation
from helpers.swing_trajectory_controller import SwingTrajectoryController
from helpers.terrain_estimator import TerrainEstimator
from simulation.quadruped_env import QuadrupedEnv
from utils.math_utils import skew
from utils.mujoco_utils.visual import plot_swing_mujoco, render_vector
from utils.quadruped_utils import GaitType, LegsAttr, estimate_terrain_slope


np.set_printoptions(precision=3, suppress=True)


# Main simulation loop ------------------------------------------------------------------
def get_gait_params(gait_type: str) -> [GaitType, float, float]:
    if gait_type == "trot":
        step_frequency = 2.5
        duty_factor = 0.65
        gait_type = GaitType.TROT
    elif gait_type == "crawl":
        step_frequency = 0.7
        duty_factor = 0.9
        gait_type = GaitType.BACKDIAGONALCRAWL
    elif gait_type == "pace":
        step_frequency = 2
        duty_factor = 0.7
        gait_type = GaitType.PACE
    elif gait_type == "bound":
        step_frequency = 4
        duty_factor = 0.65
        gait_type = GaitType.BOUNDING
    else:
        step_frequency = 2
        duty_factor = 0.65
        gait_type = GaitType.FULL_STANCE
        # print("FULL STANCE")
    return gait_type, duty_factor, step_frequency








import torch

def inverse_conjugate_euler_xyz_rate_matrix(euler_xyz_angle: torch.Tensor) -> torch.Tensor:
    """
    Given euler angles in  the XYZ convention (ie. roll pitch yaw), return the inverse conjugate euler rate matrix.

    Note
        Follow the convention of 'Representing Attitude: Euler Angles, Unit Quaternions, and Rotation Vectors'
        The inverse_conjugate_euler_xyz_rate_matrix is given by eq. 79
        https://www.astro.rug.nl/software/kapteyn-beta/_downloads/attitude.pdf

    
    Args
        euler_xyz_angle (torch.tensor): XYZ euler angle of shape(bacth, 3)

    Return
        inverse_conjugate_euler_xyz_rate_matrix (torch.tensor): inverse conjugate XYZ euler rate matrix of shape(batch, 3, 3)
    """
    # Extract Roll Pitch Yaw
    roll  = euler_xyz_angle[:, 0] # shape(batch)
    pitch = euler_xyz_angle[:, 1] # shape(batch)
    yaw   = euler_xyz_angle[:, 2] # shape(batch)

    # Compute intermediary variables
    cos_roll  = torch.cos(roll)   # shape(batch)
    sin_roll  = torch.sin(roll)   # shape(batch)
    cos_pitch = torch.cos(pitch)  # shape(batch)
    sin_pitch = torch.sin(pitch)  # shape(batch)
    tan_pitch = torch.tan(pitch)  # shape(batch)

    # Check for singularities: pitch close to +/- 90 degrees (or +/- pi/2 radians)
    assert not torch.any(torch.abs(cos_pitch) < 1e-6), "Numerical instability likely due to pitch angle near +/- 90 degrees."

    # Create the matrix of # shape(batch, 3, 3)
    inverse_conjugate_euler_xyz_rate_matrix = torch.zeros((euler_xyz_angle.shape[0], 3, 3), dtype=euler_xyz_angle.dtype, device=euler_xyz_angle.device)

    # Fill the matrix with element given in eq. 79
    inverse_conjugate_euler_xyz_rate_matrix[:, 0, 0] = 1                    # shape(batch)
    inverse_conjugate_euler_xyz_rate_matrix[:, 0, 1] = sin_roll * tan_pitch # shape(batch)
    inverse_conjugate_euler_xyz_rate_matrix[:, 0, 2] = cos_roll * tan_pitch # shape(batch)

    inverse_conjugate_euler_xyz_rate_matrix[:, 1, 0] = 0                    # shape(batch)
    inverse_conjugate_euler_xyz_rate_matrix[:, 1, 1] = cos_roll             # shape(batch)
    inverse_conjugate_euler_xyz_rate_matrix[:, 1, 2] = -sin_roll            # shape(batch)

    inverse_conjugate_euler_xyz_rate_matrix[:, 2, 0] = 0                    # shape(batch)
    inverse_conjugate_euler_xyz_rate_matrix[:, 2, 1] = sin_roll / cos_pitch # shape(batch)
    inverse_conjugate_euler_xyz_rate_matrix[:, 2, 2] = cos_roll / cos_pitch # shape(batch)

    return inverse_conjugate_euler_xyz_rate_matrix


def rotation_matrix_from_w_to_b(euler_xyz_angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrix to transform value from wolrd frame orientation to base frame oriention
    given euler angles (Roll, Pitch, Yaw) in the XYZ convention : [roll, pitch, yaw].T -> SO(3)

    Apply the three successive rotation : 
    R_xyz(roll, pitch, yaw) = R_x(roll)*R_y(pitch)*R_z(yaw)

    Note 
        Follow the convention of 'Representing Attitude: Euler Angles, Unit Quaternions, and Rotation Vectors'
        The rotation matrix is given by eq. 67
        https://www.astro.rug.nl/software/kapteyn-beta/_downloads/attitude.pdf       

    Arg
        euler_xyz_angle (torch.tensor): XYZ euler angle of shape(bacth, 3)

    Return
        rotation_matrix_from_w_to_b (torch.Tensor): Rotation matrix that rotate from w to b of shape(batch, 3, 3)
    """

    # Extract Roll Pitch Yaw
    roll  = euler_xyz_angle[:, 0] # shape(batch)
    pitch = euler_xyz_angle[:, 1] # shape(batch)
    yaw   = euler_xyz_angle[:, 2] # shape(batch)

    # Compute intermediary variables
    cos_roll  = torch.cos(roll)   # shape(batch)
    sin_roll  = torch.sin(roll)   # shape(batch)
    cos_pitch = torch.cos(pitch)  # shape(batch)
    sin_pitch = torch.sin(pitch)  # shape(batch)
    cos_yaw   = torch.cos(yaw)  # shape(batch)
    sin_yaw   = torch.sin(yaw)  # shape(batch)

    # Create the matrix of # shape(batch, 3, 3)
    rotation_matrix_from_w_to_b = torch.zeros((euler_xyz_angle.shape[0], 3, 3), dtype=euler_xyz_angle.dtype, device=euler_xyz_angle.device)

    # Fill the matrix with element
    rotation_matrix_from_w_to_b[:, 0, 0] = cos_pitch*cos_yaw                                # shape(batch)
    rotation_matrix_from_w_to_b[:, 0, 1] = cos_pitch*sin_yaw                                # shape(batch)
    rotation_matrix_from_w_to_b[:, 0, 2] = -sin_pitch                                       # shape(batch

    rotation_matrix_from_w_to_b[:, 1, 0] = sin_roll*sin_pitch*cos_yaw - cos_roll*sin_yaw    # shape(batch)
    rotation_matrix_from_w_to_b[:, 1, 1] = sin_roll*sin_pitch*sin_yaw + cos_roll*cos_yaw    # shape(batch)
    rotation_matrix_from_w_to_b[:, 1, 2] = sin_roll*cos_pitch                               # shape(batch

    rotation_matrix_from_w_to_b[:, 2, 0] = cos_roll*sin_pitch*cos_yaw + sin_roll*sin_yaw    # shape(batch)
    rotation_matrix_from_w_to_b[:, 2, 1] = cos_roll*sin_pitch*sin_yaw - sin_roll*cos_yaw    # shape(batch)
    rotation_matrix_from_w_to_b[:, 2, 2] = cos_roll*cos_pitch                               # shape(batch)

    return rotation_matrix_from_w_to_b

# ---------------------------------- Optimizer --------------------------------
class SamplingOptimizer():
    """ Model Based optimizer based on the centroidal model """

    def __init__(self, device, num_legs, optimizerCfg):
        """ 
        Args :
            device                 : 'cuda' or 'cpu' 
            num_legs               : Number of legs of the robot. Used for variable dimension definition
            num_samples            : Number of samples used for the sampling optimizer
            sampling_horizon       : Number of time steps the sampling optimizer is going to predict
            discretization_time    : Duration of a time step
            interpolation_F_method : Method to reconstruct GRF action from provided GRF warm start :
                                    can be 'discrete' (one action per time step is provided) or 'cubic spine' (a set of parameters for every time step)
            interpolation_p_method : Method to reconstruct foot touch down position action from provided warm start :
                                    can be 'discrete' (one action per time step is provided) or 'cubic spine' (a set of parameters for every time step)
        """

        # General variables
        self.device = device
        self.num_legs = num_legs
        
        # Save Config
        self.cfg = optimizerCfg

        # Optimizer configuration
        self.num_samples = optimizerCfg.num_samples
        self.sampling_horizon = optimizerCfg.prevision_horizon
        self.dt = optimizerCfg.discretization_time    
        self.num_optimizer_iterations = optimizerCfg.num_optimizer_iterations

        # Define Interpolation method for GRF and interfer GRF input size 
        if   optimizerCfg.parametrization_F == 'cubic spline' : 
            self.interpolation_F=self.compute_cubic_spline
            self.F_param = 4
        elif optimizerCfg.parametrization_F == 'discrete'     :
            self.interpolation_F=self.compute_discrete
            self.F_param = self.sampling_horizon
        else : raise NotImplementedError('Request interpolation method is not implemented yet')

        # Define Interpolation method for foot touch down position and interfer foot touch down position input size 
        if   optimizerCfg.parametrization_p == 'cubic spline' : 
            self.interpolation_p=self.compute_cubic_spline
            self.p_param = 4
        elif optimizerCfg.parametrization_p == 'discrete'     : 
            self.interpolation_p=self.compute_discrete
            self.p_param = self.sampling_horizon
        else : raise NotImplementedError('Request interpolation method is not implemented yet')

        # Input and State dimension for centroidal model : hardcoded because the centroidal model is hardcoded
        self.state_dim = 24 # CoM_pos(3) + lin_vel(3) + CoM_pose(3) + ang_vel(3) + foot_pos(12) 
        self.input_dim = 12 # GRF(12) (foot touch down pos is state and an input)


        # TODO Get these value properly
        self.mu = optimizerCfg.mu
        self.gravity_lw = torch.tensor((0.0, 0.0, -9.81), device=self.device) # shape(3) #self._env.sim.cfg.gravity
        # self.robot_mass = 24.64
        self.robot_mass = 20.6380
        self.robot_inertia = torch.tensor([[ 0.2310941359705289,   -0.0014987128245817424, -0.021400468992761768 ], # shape (3,3)
                                           [-0.0014987128245817424, 1.4485084687476608,     0.0004641447134275615],
                                           [-0.021400468992761768,  0.0004641447134275615,  1.503217877350808    ]],device=self.device)
        self.inv_robot_inertia = torch.linalg.inv(self.robot_inertia) # shape(3,3)
        self.F_z_min = 0
        self.F_z_max = self.robot_mass*9.81

        # Boolean to enable variable optimization or not
        self.optimize_f = optimizerCfg.optimize_f
        self.optimize_d = optimizerCfg.optimize_d
        self.optimize_p = optimizerCfg.optimize_p
        self.optimize_F = optimizerCfg.optimize_F

        # Sampling law
        if   optimizerCfg.sampling_law == 'normal' : self.sampling_law = self.normal_sampling
        elif optimizerCfg.sampling_law == 'uniform': self.sampling_law = self.uniform_sampling

        # Wether to force clip the sample to std
        self.clip_sample = optimizerCfg.clip_sample

        # How much of the previous solution is used to generate samples compare to the provided guess (in [0,1])
        self.propotion_previous_solution = optimizerCfg.propotion_previous_solution

        # Define the height reference for the tracking
        self.height_ref = optimizerCfg.height_ref

        # Define Variance for the sampling law
        self.std_f = torch.tensor((0.05), device=device)
        self.std_d = torch.tensor((0.05), device=device)
        self.std_p = torch.tensor((0.02), device=device)
        self.std_F = torch.tensor((5.00), device=device)

        # State weight - shape(state_dim)
        self.Q_vec = torch.zeros(self.state_dim, device=self.device)
        self.Q_vec[0]  = 0.0        #com_x
        self.Q_vec[1]  = 0.0        #com_y
        self.Q_vec[2]  = 111500     #com_z
        self.Q_vec[3]  = 5000       #com_vel_x
        self.Q_vec[4]  = 5000       #com_vel_y
        self.Q_vec[5]  = 200        #com_vel_z
        self.Q_vec[6]  = 11200      #base_angle_roll
        self.Q_vec[7]  = 11200      #base_angle_pitch
        self.Q_vec[8]  = 0.0        #base_angle_yaw
        self.Q_vec[9]  = 20         #base_angle_rates_x
        self.Q_vec[10] = 20         #base_angle_rates_y
        self.Q_vec[11] = 600        #base_angle_rates_z

        # Input weight - shape(input_dim)
        self.R_vec = torch.zeros(self.input_dim, device=self.device)

        # Initialize the best solution
        self.f_best = 1.4*torch.ones( (1,self.num_legs),                 device=device)
        self.d_best = 0.6*torch.ones( (1,self.num_legs),                 device=device)
        self.p_best =     torch.zeros((1,self.num_legs,3,self.p_param ), device=device)
        self.F_best =     torch.zeros((1,self.num_legs,3,self.F_param ), device=device)
        self.F_best[:,:,2,:] = 100.0

        # For plotting
        self.live_plot = True
        self.robot_height_list = [0.0]
        self.cost_list = [0.0]


    def reset(self):
        # Reset the best solution
        self.f_best = 1.5*torch.ones( (1,self.num_legs),                 device=self.device)
        self.d_best = 0.6*torch.ones( (1,self.num_legs),                 device=self.device)
        self.p_best =     torch.zeros((1,self.num_legs,3,self.p_param ), device=self.device)
        self.F_best =     torch.zeros((1,self.num_legs,3,self.F_param ), device=self.device) 
        self.F_best[:,:,2,:] = 50.0


    def optimize_latent_variable(self, state_current,ref_state, f:torch.Tensor, d:torch.Tensor, p_lw:torch.Tensor, F_lw:torch.Tensor, phase:torch.Tensor, c_prev:torch.Tensor, height_map) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Given latent variable f,d,F,p, returns f*,d*,F*,p*, optimized with a sampling optimization 
        
        Args :
            f      (Tensor): Leg frequency                of shape(batch_size, num_leg)
            d      (Tensor): Leg duty cycle               of shape(batch_size, num_leg)
            p_lw   (Tensor): Foot touch down position     of shape(batch_size, num_leg, 3, p_param)
            F_lw   (Tensor): ground Reaction Forces       of shape(batch_size, num_leg, 3, F_param)
            phase  (Tensor): Current feet phase           of shape(batch_size, num_leg)
            c_prev (Tensor): Contact sequence determined at previous iteration of shape (batch_size, num_leg)
            height_map (Tr): Height map arround the robot of shape(x, y)

        Returns :
            f_star    (Tensor): Leg frequency                of shape(batch_size, num_leg)
            d_star    (Tensor): Leg duty cycle               of shape(batch_size, num_leg)
            p_star_lw (Tensor): Foot touch down position     of shape(batch_size, num_leg, 3, p_param)
            F_star_lw (Tensor): ground Reaction Forces       of shape(batch_size, num_leg, 3, F_param)
        """
        print()

        for i in range(self.num_optimizer_iterations):
            print(f'\niteration {i}')
            # --- Step 1 : Generate the samples and bound them to valid input
            f_samples, d_samples, p_lw_samples, F_lw_samples = self.generate_samples(iter=i, f=f, d=d, p_lw=p_lw, F_lw=F_lw, height_map=height_map)

            # --- Step 2 : Given f and d samples -> generate the contact sequence for the samples
            c_samples, new_phase = self.gait_generator(f_samples=f_samples, d_samples=d_samples, phase=phase.squeeze(0), sampling_horizon=self.sampling_horizon, dt=self.dt)

            # F_lw_samples[:,:,2,:] = F_lw_samples[:,:,2,:] + (c_samples.unsqueeze(-1)) * ((self.robot_mass*9.81) / torch.sum(c_samples, dim=1).unsqueeze(1))

            # --- Step 2 : prepare the variables : convert from torch.Tensor to Jax
            initial_state, reference_seq_state, reference_seq_input_samples, action_param_samples = self.prepare_variable_for_compute_rollout(state_current=state_current, ref_state=ref_state, c_samples=c_samples, p_lw_samples=p_lw_samples, F_lw_samples=F_lw_samples, feet_in_contact=c_prev[0,])

            # --- Step 3 : Compute the rollouts to find the rollout cost : can't used named argument with VMAP...
            cost_samples = self.compute_rollout( initial_state, reference_seq_state, reference_seq_input_samples, action_param_samples, c_samples)

            # --- Step 4 : Given the samples cost, find the best control action
            f_star, d_star, p0_star_lw, F0_star_lw = self.find_best_actions(cost_samples, f_samples, d_samples, c_samples, p_lw_samples, F_lw_samples, c_prev)


        return f_star, d_star, p0_star_lw, F0_star_lw # p_star_lw, F_star_lw


    def prepare_variable_for_compute_rollout(self, state_current,ref_state, c_samples:torch.Tensor, p_lw_samples:torch.Tensor, F_lw_samples:torch.Tensor, feet_in_contact:torch.Tensor) -> tuple[dict, dict, dict, dict]:
        """ Helper function to modify the embedded state, reference and action to be used with the 'compute_rollout' function

        Note :
            Initial state and reference can be retrieved only with the environment
            _w   : World frame (inertial frame)
            _lw  : World frame centered at the environment center -> local world frame
            _b   : Base frame (attached to robot's base)
            _h   : Horizontal frame -> Base frame position for xy, world frame for z, roll, pitch, base frame for yaw
            _bw  : Base/world frame -> Base frame position, world frame rotation

        Args :
            env  (ManagerBasedRLEnv): Environment manager to retrieve all necessary simulation variable
            c_samples       (t.bool): Foot contact sequence sample                                                      of shape(num_samples, num_legs, sampling_horizon)
            p_lw_samples    (Tensor): Foot touch down position                                                          of shape(num_samples, num_legs, 3, p_param)
            F_lw_samples    (Tensor): ground Reaction Forces                                                            of shape(num_samples, num_legs, 3, F_param)
            feet_in_contact (Tensor): Feet in contact, determined by prevous solution                                   of shape(num_legs)

        
        Return :
            initial_state         (dict): Dictionnary containing the current robot's state
                pos_com_lw      (Tensor): CoM position in local world frame                             of shape(3)
                lin_com_vel_lw  (Tensor): CoM linear velocity in local world frame                      of shape(3)
                euler_xyz_angle (Tensor): CoM orientation (wrt. to l. world frame) as XYZ euler angle   of shape(3)
                ang_vel_com_b   (Tensor): CoM angular velocity as roll pitch yaw                        of shape(3)
                p_lw            (Tensor): Feet position in local world frame                            of shape(num_legs, 3)

            reference_seq_state   (dict): Dictionnary containing the robot's reference state along the prediction horizon
                pos_com_lw      (Tensor): CoM position in local world frame                             of shape(3, sampling_horizon)
                lin_com_vel_lw  (Tensor): CoM linear velocity in local world frame                      of shape(3, sampling_horizon)
                euler_xyz_angle (Tensor): CoM orientation (wrt. to l. world frame) as XYZ euler angle   of shape(3, sampling_horizon)
                ang_vel_com_b   (Tensor): CoM angular velocity as roll pitch yaw                        of shape(3, sampling_horizon)
                p_lw            (Tensor): Feet position in local world frame                            of shape(num_legs, 3, sampling_horizon)                

            reference_seq_input_samples (dict) 
                F_lw            (Tensor): Reference GRF sequence samples along the prediction horizon   of shape(num_sample, num_legs, 3, sampling_horizon)  

            action_param_samples  (dict): Dictionnary containing the robot's actions along the prediction horizon 
                p_lw            (Tensor): Foot touch down position in local world frame                 of shape(num_samples, num_legs, 3, p_param)
                F_lw            (Tensor): GRF parameters in local world frame                           of shape(num_samples, num_legs, 3, F_param)
        """
        initial_state = {}
        reference_seq_state = {}
        reference_seq_input_samples = {}
        action_param_samples = {}

        data_type = torch.float32


        # ----- Step 1 : Retrieve the initial state
        p_lw = torch.stack((torch.from_numpy(state_current['foot_FL']).to(dtype=data_type, device=self.device),
                            torch.from_numpy(state_current['foot_FR']).to(dtype=data_type, device=self.device),
                            torch.from_numpy(state_current['foot_RL']).to(dtype=data_type, device=self.device),
                            torch.from_numpy(state_current['foot_RR']).to(dtype=data_type, device=self.device)), dim=0)


        # Prepare the state (at time t)
        initial_state['pos_com_lw']      = torch.from_numpy(state_current['position']).to(dtype=data_type, device=self.device)       # shape(3)              # CoM position in local world frame, height is propriocetive
        initial_state['lin_com_vel_lw']  = torch.from_numpy(state_current['linear_velocity']).to(dtype=data_type, device=self.device)   # shape(3)              # Linear Velocity in local world frame
        initial_state['euler_xyz_angle'] = torch.from_numpy(state_current['orientation']).to(dtype=data_type, device=self.device)  # shape(3)              # Euler angle in XYZ convention
        initial_state['ang_vel_com_b']   = torch.from_numpy(state_current['angular_velocity']).to(dtype=data_type, device=self.device)    # shape(3)              # Angular velocity in base frame
        initial_state['p_lw']            = p_lw             # shape(num_legs, 3)    # Foot position in local world frame


        # ----- Step 2 : Retrieve the robot's reference along the integration horizon
        # Compute the gravity compensation GRF along the horizon : of shape (num_samples, num_legs, 3, sampling_horizon)
        num_leg_contact_seq_samples = (torch.sum(c_samples, dim=1)).clamp(min=1) # Compute the number of leg in contact, clamp by minimum 1 to avoid division by zero. shape(num_samples, sampling_horizon)
        gravity_compensation_F_samples = torch.zeros((self.num_samples, self.num_legs, 3, self.sampling_horizon), device=self.device) # shape (num_samples, num_legs, 3, sampling_horizon)
        gravity_compensation_F_samples[:,:,2,:] =  c_samples * ((self.robot_mass * 9.81)/num_leg_contact_seq_samples).unsqueeze(1)    # shape (num_samples, num_legs, sampling_horizon)
        
        p_ref_seq_lw = torch.stack((torch.from_numpy(state_current['foot_FL']).to(dtype=data_type, device=self.device),
                                    torch.from_numpy(state_current['foot_FR']).to(dtype=data_type, device=self.device),
                                    torch.from_numpy(state_current['foot_RL']).to(dtype=data_type, device=self.device),
                                    torch.from_numpy(state_current['foot_RR']).to(dtype=data_type, device=self.device)), dim=0).unsqueeze(-1).expand(self.num_legs, 3, self.sampling_horizon)

        # Prepare the reference sequence (at time t, t+dt, etc.)
        reference_seq_state['pos_com_lw']      = torch.from_numpy(ref_state['ref_position']).to(dtype=data_type, device=self.device).unsqueeze(-1).expand(3, self.sampling_horizon)       # shape(3, sampling_horizon)           # CoM position reference in local world frame
        reference_seq_state['lin_com_vel_lw']  = torch.from_numpy(ref_state['ref_linear_velocity']).to(dtype=data_type, device=self.device).unsqueeze(-1).expand(3, self.sampling_horizon)    # shape(3, sampling_horizon)           # Linear Velocity reference in local world frame
        reference_seq_state['euler_xyz_angle'] = torch.from_numpy(ref_state['ref_orientation']).to(dtype=data_type, device=self.device).unsqueeze(-1).expand(3, self.sampling_horizon)  # shape(3, sampling_horizon)           # Euler angle reference in XYZ convention
        reference_seq_state['ang_vel_com_b']   = torch.from_numpy(ref_state['ref_angular_velocity']).to(dtype=data_type, device=self.device).unsqueeze(-1).expand(3, self.sampling_horizon)    # shape(3, sampling_horizon)           # Angular velocity reference in base frame
        reference_seq_state['p_lw']            = p_ref_seq_lw            # shape(num_legs, 3, sampling_horizon) # Foot position in local world frame



        # Prepare the input sequence reference for the samples
        reference_seq_input_samples['F_lw'] = gravity_compensation_F_samples # shape(num_samples, num_legs, 3, sampling_horizon) # gravity compensation per samples along the horizon


        # ----- Step 3 : Retrieve the actions and prepare them with the correct method
        action_param_samples['p_lw'] = p_lw_samples # Foot touch down position samples    # shape(num_samples, num_legs, 3 p_param)
        action_param_samples['F_lw'] = F_lw_samples # Ground Reaction Forces samples      # shape(num_samples, num_legs, 3 F_param)


        return initial_state, reference_seq_state, reference_seq_input_samples, action_param_samples


    def find_best_actions(self,cost_samples, f_samples: torch.Tensor, d_samples: torch.Tensor, c_samples: torch.Tensor, p_lw_samples: torch.Tensor, F_lw_samples: torch.Tensor, c_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
        """ Given action samples and associated cost, filter invalid values and retrieves the best cost and associated actions
        
        Args 
            cost_samples (Tensor): Associated trajectory cost          of shape(num_samples)
            f_samples    (Tensor): Leg frequency samples               of shape(num_samples, num_leg)
            d_samples    (Tensor): Leg duty cycle samples              of shape(num_samples, num_leg)
            c_samples    (Tensor): Leg contact sequence samples        of shape(num_samples, num_legs, time_horizon)
            p_lw_samples (Tensor): Foot touch down position samples    of shape(num_samples, num_leg, 3, p_param)
            F_lw_samples (Tensor): Ground Reaction forces samples      of shape(num_samples, num_leg, 3, F_param)
             
        Returns
            f_star    (Tensor): Best leg frequency                     of shape(1, num_leg)
            d_star    (Tensor): Best leg duty cycle                    of shape(1, num_leg)
            p0_star_lw (Tensor): Best foot touch down position         of shape(1, num_leg, 3)
            F0_star_lw (Tensor): Best ground Reaction Forces           of shape(1, num_leg, 3)  
        """

        # Saturate the cost in case of NaN or inf
        cost_samples[torch.isnan(cost_samples) | torch.isinf(cost_samples)] = 1e10

        # Take the best found control parameters
        best_index = torch.argmin(cost_samples)
        best_cost = cost_samples[best_index] # cost_samples.take(best_index)

        print('cost sample ',cost_samples)
        print('Best cost :', best_cost, ', best index :', best_index)

        if self.live_plot:
            self.cost_list.append(best_cost.cpu().numpy())
            if len(self.cost_list) > 100 : self.cost_list.pop(0)
            np.savetxt('live_variable/cost.csv', [self.cost_list], delimiter=',', fmt='%.3f')

        # Retrieve best sample, given the best index
        f_star = f_samples[best_index].unsqueeze(0)           # shape(1, num_leg)
        d_star = d_samples[best_index].unsqueeze(0)           # shape(1, num_leg)
        c_star = c_samples[best_index].unsqueeze(0)           # shape(1, num_leg, sampling_horizon)
        p_star_lw = p_lw_samples[best_index].unsqueeze(0)     # shape(1, num_leg, 3, p_param)
        F_star_lw = F_lw_samples[best_index].unsqueeze(0)     # shape(1, num_leg, 3, F_param)

        if self.live_plot:
            np.savetxt("live_variable/F_best_FL.csv", F_star_lw[0,0,:,:].cpu().numpy(), delimiter=",")
            np.savetxt("live_variable/F_best_FR.csv", F_star_lw[0,1,:,:].cpu().numpy(), delimiter=",")
            np.savetxt("live_variable/F_best_RL.csv", F_star_lw[0,2,:,:].cpu().numpy(), delimiter=",")
            np.savetxt("live_variable/F_best_RR.csv", F_star_lw[0,3,:,:].cpu().numpy(), delimiter=",")

        # Update previous best solution
        self.f_best, self.d_best, self.p_best, self.F_best = f_star, d_star, p_star_lw, F_star_lw
        self.f_best, self.d_best, self.p_best, self.F_best = self.shift_actions(f=self.f_best, d=self.d_best, p=self.p_best, F=self.F_best)

        # reset the delta of the actions if contact ended (ie. started swing phase)
        lift_off_mask = ((c_prev[:,:] == 1) * (c_star[:,:,0] == 0)) # shape (1,num_legs) # /!\ c_prev is incremented with sim_dt, while c_star with mpc_dt : Thus, 
        self.F_best[lift_off_mask] = 0.0
        self.F_best[lift_off_mask,:,2] = (self.robot_mass*9.81)/2 #(torch.sum(c_star[:,:,0]+1).clamp(min=1))
        print('lift off mask', lift_off_mask)

        # # Retrive action to be applied at next time step
        # # p : Foot touch Down
        # if   self.cfg.parametrization_p == 'cubic spline':
        #     p0_star_lw = p_star_lw[...,1]
        # elif self.cfg.parametrization_p == 'discrete':
        #     p0_star_lw = p_star_lw[...,0]

        # # F : GRF
        # if   self.cfg.parametrization_F == 'cubic spline':
        #     F0_star_lw = F_star_lw[...,1]
        # elif self.cfg.parametrization_F == 'discrete':
        #     F0_star_lw = F_star_lw[...,0]

        # Retrive action to be applied at next time step
        p0_star_lw = self.interpolation_p(parameters=p_star_lw, step=0, horizon=self.sampling_horizon) # shape(1, num_legs, 3)
        F0_star_lw = self.interpolation_F(parameters=F_star_lw, step=0, horizon=self.sampling_horizon) # shape(1, num_legs, 3)

        if self.optimize_f : print('f - cum. diff. : %3.2f' % torch.sum(torch.abs(f_star - f_samples[0,...])))
        if self.optimize_d : print('d - cum. diff. : %3.2f' % torch.sum(torch.abs(d_star - d_samples[0,...])))
        if self.optimize_p : print('p - cum. diff. : %3.2f' % torch.sum(torch.abs(p_star_lw - p_lw_samples[0,...])))
        if self.optimize_F : print('F - cum. diff. : %5.1f' % torch.sum(torch.abs(F_star_lw - F_lw_samples[0,...])))

        # Add gravity compensation
        # F0_star_lw -= c_star[:,:,0].unsqueeze(-1) * self.gravity_lw.unsqueeze(0).unsqueeze(0) * self.robot_mass / torch.sum(c_star[:,:,0].unsqueeze(0).unsqueeze(-1)) # shape(1, num_legs, 3)

        return f_star, d_star, p0_star_lw, F0_star_lw


    def shift_actions(self, f: torch.Tensor, d: torch.Tensor, p: torch.Tensor,F: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Shift the actions from one time step and copy the last action
        """
        f[...,0:-1] = f[...,1:].clone()
        d[...,0:-1] = d[...,1:].clone()
        p[...,0:-1] = p[...,1:].clone()
        F[...,0:-1] = F[...,1:].clone()

        return f, d, p, F


    def generate_samples(self, iter:int, f:torch.Tensor, d:torch.Tensor, p_lw:torch.Tensor, F_lw:torch.Tensor, height_map:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Given action (f,d,p,F), generate action sequence samples (f_samples, d_samples, p_samples, F_samples)
        If multiple action sequence are provided (because several policies are blended together), generate samples
        from these polices with equal proportions. TODO
        
        Args :
            f    (Tensor): Leg frequency                                of shape(batch_size, num_leg)
            d    (Tensor): Leg duty cycle                               of shape(batch_size, num_leg)
            p_lw (Tensor): Foot touch down position                     of shape(batch_size, num_leg, 3, p_param)
            F_lw (Tensor): ground Reaction Forces                       of shape(batch_size, num_leg, 3, F_param)
            height_map   (torch.Tensor): Height map arround the robot   of shape(x, y)
            
        Returns :
            f_samples    (Tensor) : Leg frequency samples               of shape(num_samples, num_leg)
            d_samples    (Tensor) : Leg duty cycle samples              of shape(num_samples, num_leg)
            p_lw_samples (Tensor) : Foot touch down position samples    of shape(num_samples, num_leg, 3, p_param)
            F_lw_samples (Tensor) : Ground Reaction forces samples      of shape(num_samples, num_leg, 3, F_param)
        """
        # Define how much samples from the RL or from the previous solution we're going to sample
        if iter == 0:
            num_samples_previous_best = int(self.num_samples * self.propotion_previous_solution)
            num_samples_RL = self.num_samples - num_samples_previous_best
        else :
            num_samples_previous_best = self.num_samples
            num_samples_RL = 0

        # Samples from the previous best solution
        f_samples_best    = self.sampling_law(num_samples=num_samples_previous_best, mean=self.f_best[0], std=self.std_f, clip=self.clip_sample)
        d_samples_best    = self.sampling_law(num_samples=num_samples_previous_best, mean=self.d_best[0], std=self.std_d, clip=self.clip_sample)
        p_lw_samples_best = self.sampling_law(num_samples=num_samples_previous_best, mean=self.p_best[0], std=self.std_p, clip=self.clip_sample)
        F_lw_samples_best = self.sampling_law(num_samples=num_samples_previous_best, mean=self.F_best[0], std=self.std_F, clip=self.clip_sample)

        # Samples from the provided guess
        f_samples_rl    = self.sampling_law(num_samples=num_samples_RL, mean=f[0],    std=self.std_f, clip=self.clip_sample)
        d_samples_rl    = self.sampling_law(num_samples=num_samples_RL, mean=d[0],    std=self.std_d, clip=self.clip_sample)
        p_lw_samples_rl = self.sampling_law(num_samples=num_samples_RL, mean=p_lw[0], std=self.std_p, clip=self.clip_sample)
        F_lw_samples_rl = self.sampling_law(num_samples=num_samples_RL, mean=F_lw[0], std=self.std_F, clip=self.clip_sample)

        # Concatenate the samples
        f_samples    = torch.cat((f_samples_rl,    f_samples_best),    dim=0)
        d_samples    = torch.cat((d_samples_rl,    d_samples_best),    dim=0)
        p_lw_samples = torch.cat((p_lw_samples_rl, p_lw_samples_best), dim=0)
        F_lw_samples = torch.cat((F_lw_samples_rl, F_lw_samples_best), dim=0)

        # Clamp the input to valid range
        f_samples, d_samples, p_lw_samples, F_lw_samples = self.enforce_valid_input(f_samples=f_samples, d_samples=d_samples, p_lw_samples=p_lw_samples, F_lw_samples=F_lw_samples, height_map=height_map)

        # Set the foot height to the nominal foot height # TODO change
        p_lw_samples[:,:,2,:] = p_lw[0,:,2,:]

        # Put the RL actions as the first samples
        f_samples[0,:]        = f[0,:]
        d_samples[0,:]        = d[0,:]
        p_lw_samples[0,:,:,:] = p_lw[0,:,:,:]
        F_lw_samples[0,:,:,:] = F_lw[0,:,:,:]

        # Put the Previous best actions as the second samples
        f_samples[1,:]        = self.f_best[0,:]
        d_samples[1,:]        = self.d_best[0,:]
        p_lw_samples[1,:,:,:] = self.p_best[0,:,:,:]
        F_lw_samples[1,:,:,:] = self.F_best[0,:,:,:]

        # If optimization is set to false, samples are feed with initial guess
        if not self.optimize_f : f_samples[:,:]        = f
        if not self.optimize_d : d_samples[:,:]        = d
        if not self.optimize_p : p_lw_samples[:,:,:,:] = p_lw
        if not self.optimize_F : F_lw_samples[:,:,:,:] = F_lw

        return f_samples, d_samples, p_lw_samples, F_lw_samples


    def normal_sampling(self, num_samples:int, mean:torch.Tensor, std:torch.Tensor|None=None, seed:int|None=None, clip=False) -> torch.Tensor:
        """ Normal sampling law given mean and std -> return a samples
        
        Args :
            mean     (Tensor): Mean of normal sampling law          of shape(num_dim1, num_dim2, etc.)
            std      (Tensor): Standard dev of normal sampling law  of shape(num_dim1, num_dim2, etc.)
            num_samples (int): Number of samples to generate
            seed        (int): seed to generate random numbers

        Return :
            samples  (Tensor): Samples generated with mean and std  of shape(num_sammple, num_dim1, num_dim2, etc.)
        """

        # Seed if provided
        if seed : 
            torch.manual_seed(seed)

        if std is None :
            std = torch.ones_like(mean)

        # Sample from a normal law with the provided parameters
        if clip == True :
            samples = mean + (std * torch.randn((num_samples,) + mean.shape, device=self.device)).clamp(min=-2*std, max=2*std)
        else :
            samples = mean + (std * torch.randn((num_samples,) + mean.shape, device=self.device))

        return samples
    

    def uniform_sampling(self, num_samples:int, mean:torch.Tensor, std:torch.Tensor|None=None, seed:int|None=None, clip=False) -> torch.Tensor:
        """ Normal sampling law given mean and std -> return a samples
        
        Args :
            mean     (Tensor): Mean of normal sampling law          of shape(num_dim1, num_dim2, etc.)
            std      (Tensor): Standard dev of normal sampling law  of shape(num_dim1, num_dim2, etc.)
            num_samples (int): Number of samples to generate
            seed        (int): seed to generate random numbers

        Return :
            samples  (Tensor): Samples generated with mean and std  of shape(num_sammple, num_dim1, num_dim2, etc.)
        """

        # Seed if provided
        if seed : 
            torch.manual_seed(seed)

        if std is None :
            std = torch.ones_like(mean)

        # Sample from a uniform law with the provided parameters
        samples = mean + (std * torch.empty((num_samples,) + mean.shape, device=self.device).uniform_(-1.0, 1.0))

        return samples


    def gait_generator(self, f_samples: torch.Tensor, d_samples: torch.Tensor, phase: torch.Tensor, sampling_horizon: int, dt) -> tuple[torch.Tensor, torch.Tensor]:
        """ Implement a gait generator that return a contact sequence given a leg frequency and a leg duty cycle
        Increment phase by dt*f 
        restart if needed
        return contact : 1 if phase < duty cyle, 0 otherwise  
        c == 1 : Leg is in contact (stance)
        c == 0 : Leg is in swing

        Note:
            No properties used, no for loop : purely functional -> made to be jitted
            parallel_rollout : this is optional, it will work without the parallel rollout dimension

        Args:
            - f_samples     (Tensor): Leg frequency samples                 of shape(num_samples, num_legs)
            - d_samples     (Tensor): Stepping duty cycle samples in [0,1]  of shape(num_samples, num_legs)
            - phase         (Tensor): phase of leg samples in [0,1]         of shape(num_legs)
            - sampling_horizon (int): Time horizon for the contact sequence

        Returns:
            - c_samples     (t.bool): Foot contact sequence samples         of shape(num_samples, num_legs, sampling_horizon)
            - phase_samples (Tensor): The phase samples updated by 1 dt     of shape(num_samples, num_legs)
        """
        
        # Increment phase of f*dt: new_phases[0] : incremented of 1 step, new_phases[1] incremented of 2 steps, etc. without a for loop.
        # new_phases = phase + f*dt*[1,2,...,sampling_horizon]
        #            (1, num_legs, 1)                  +  (samples, legs, 1)      * (1, 1, sampling_horizon) -> shape(samples, legs, sampling_horizon)
        new_phases_samples = phase.unsqueeze(0).unsqueeze(-1) + (f_samples.unsqueeze(-1) * torch.linspace(start=1, end=sampling_horizon, steps=sampling_horizon, device=self.device).unsqueeze(0).unsqueeze(1)*dt)

        # Make the phases circular (like sine) (% is modulo operation)
        new_phases_samples = new_phases_samples%1

        # Save first phase -> shape(num_samples, num_legs)
        new_phase_samples = new_phases_samples[..., 0]

        # Make comparaison to return discret contat sequence : c = 1 if phase < d, 0 otherwise
        #(samples, legs, sampling_horizon) <= (samples, legs, 1) -> shape(num_samples, num_legs, sampling_horizon)
        c_samples = new_phases_samples <= d_samples.unsqueeze(-1)

        return c_samples, new_phase_samples


    def compute_rollout(self, initial_state: dict, reference_seq_state: dict, reference_seq_input_samples: dict, action_param_samples: dict, c_samples: torch.Tensor) -> torch.Tensor:
        """
        Calculate cost of rollouts of given action sequence samples 

        Args :
            initial_state         (dict): Dictionnary containing the current robot's state
                pos_com_lw      (Tensor): CoM position in local world frame                             of shape(3)
                lin_com_vel_lw  (Tensor): CoM linear velocity in local world frame                      of shape(3)
                euler_xyz_angle (Tensor): CoM orientation (wrt. to l. world frame) as XYZ euler angle   of shape(3)
                ang_vel_com_b   (Tensor): CoM angular velocity as roll pitch yaw                        of shape(3)
                p_lw            (Tensor): Feet position in local world frame                            of shape(num_legs, 3)

            reference_seq_state   (dict): Dictionnary containing the robot's reference state along the prediction horizon
                pos_com_lw      (Tensor): CoM position in local world frame                             of shape(3, sampling_horizon)
                lin_com_vel_lw  (Tensor): CoM linear velocity in local world frame                      of shape(3, sampling_horizon)
                euler_xyz_angle (Tensor): CoM orientation (wrt. to l. world frame) as XYZ euler angle   of shape(3, sampling_horizon)
                ang_vel_com_b   (Tensor): CoM angular velocity as roll pitch yaw                        of shape(3, sampling_horizon)
                p_lw            (Tensor): Feet position in local world frame                            of shape(num_legs, 3, sampling_horizon)                

            reference_seq_input_samples (dict) 
                F_lw            (Tensor): Reference GRF sequence samples along the prediction horizon   of shape(num_sample, num_legs, 3, sampling_horizon)  

            action_param_samples  (dict): Dictionnary containing the robot's actions along the prediction horizon 
                p_lw            (Tensor): Foot touch down position in local world frame                 of shape(num_samples, num_legs, 3, p_param)
                F_lw            (Tensor): GRF parameters in local world frame                           of shape(num_samples, num_legs, 3, F_param)

            c_samples       (t.bool): Foot contact sequence sample                                                      of shape(num_samples, num_legs, sampling_horizon)

        Return 
            cost_samples        (Tensor): Cost associated at each sample (along the prediction horizon) of shape(num_samples)
        """
        cost_samples = torch.zeros(self.num_samples, device=self.device)
        state = {}
        state['pos_com_lw']      = initial_state['pos_com_lw'].unsqueeze(0).expand(self.num_samples, 3)
        state['lin_com_vel_lw']  = initial_state['lin_com_vel_lw'].unsqueeze(0).expand(self.num_samples, 3)
        state['euler_xyz_angle'] = initial_state['euler_xyz_angle'].unsqueeze(0).expand(self.num_samples, 3)
        state['ang_vel_com_b']   = initial_state['ang_vel_com_b'].unsqueeze(0).expand(self.num_samples, 3)
        state['p_lw']            = initial_state['p_lw'].unsqueeze(0).expand(self.num_samples, self.num_legs, 3)
        input = {}

        for i in range(self.sampling_horizon):
            # --- Step 1 : prepare the inputs
            # Find the current action given the actions parameters
            input['p_lw'] = self.interpolation_p(parameters=action_param_samples['p_lw'], step=i, horizon=self.sampling_horizon)# shape(num_samples, num_legs, 3)
            input['F_lw'] = self.interpolation_F(parameters=action_param_samples['F_lw'], step=i, horizon=self.sampling_horizon)# shape(num_samples, num_legs, 3)
            contact = c_samples[:,:,i]                                                                                          # shape(num_samples, num_legs)

            # Add gravity compensation
            # input['F_lw'] -= contact.unsqueeze(-1) * (self.gravity_lw.unsqueeze(0).unsqueeze(0) * self.robot_mass) / torch.sum(contact, dim=1).unsqueeze(-1).unsqueeze(-1)     

            # Enforce force constraints (Friction cone constraints)
            # input['F_lw'] = self.enforce_friction_cone_constraints_torch(F=input['F_lw'], mu=self.mu)                           # shape(num_samples, num_legs, 3)


            # --- Step 2 : Step the model
            new_state = self.centroidal_model_step(state=state, input=input, contact=contact)
            state = new_state


            # --- Step 3 : compute the step cost
            state_vector     = torch.cat([vector.view(self.num_samples, -1) for vector in state.values()], dim=1)               # Shape: (num_samples, state_dim)
            ref_state_vector = torch.cat([vector.select(-1, i).view(-1) for vector in reference_seq_state.values()], dim=0)     # Shape: (state_dim)
            
            # Compute the state cost
            state_error = state_vector - ref_state_vector.unsqueeze(0)                                                          # shape (num_samples, state_dim)
            state_cost  = torch.sum(self.Q_vec.unsqueeze(0) * (state_error ** 2), dim=1)                                        # Shape (num_samples)

            # Compute the input cost
            input_error = (input['F_lw'] - reference_seq_input_samples['F_lw'][:,:,:,i]).flatten(1,2)                           # shape (num_samples, action_dim)
            input_cost  = torch.sum(self.R_vec.unsqueeze(0) * (input_error ** 2), dim=1)                                        # Shape (num_samples)

            step_cost = state_cost #+ input_cost                                                                                # shape(num_samples)

            # Update the trajectory cost
            cost_samples += step_cost                                                                                           # shape(num_samples)

        return cost_samples


    def compute_cubic_spline(self, parameters, step, horizon):
        """ Given a set of spline parameters, and the point in the trajectory return the function value 
        
        Args :
            parameters (Tensor): Spline action parameter      of shape(batch, num_legs, 3, spline_param)              
            step          (int): The point in the curve in [0, horizon]
            horizon       (int): The length of the curve
            
        Returns : 
            actions    (Tensor): Discrete action              of shape(batch, num_legs, 3)
        """
        # Find the point in the curve q in [0,1]
        tau = step/(horizon)        
        q = (tau - 0.0)/(1.0-0.0)
        
        # Compute the spline interpolation parameters
        a =  2*q*q*q - 3*q*q     + 1
        b =    q*q*q - 2*q*q + q
        c = -2*q*q*q + 3*q*q
        d =    q*q*q -   q*q

        # Compute intermediary parameters 
        phi_1 = 0.5*(parameters[...,2]  - parameters[...,0]) # shape (batch, num_legs, 3)
        phi_2 = 0.5*(parameters[...,3]  - parameters[...,1]) # shape (batch, num_legs, 3)

        # Compute the spline
        actions = a*parameters[...,1] + b*phi_1 + c*parameters[...,2]  + d*phi_2 # shape (batch, num_legs, 3)

        return actions


    def compute_discrete(self, parameters, step, horizon):
        """ If actions are discrete actions, no interpolation are required.
        This function simply return the action at the right time step

        Args :
            parameters (Tensor): Discrete action parameter    of shape(batch, num_legs, 3, sampling_horizon)
            step          (int): The current step index along horizon
            horizon       (int): Not used : here for compatibility

        Returns :
            actions    (Tensor): Discrete action              of shape(batch, num_legs, 3)
        """

        actions = parameters[:,:,:,step]
        return actions
     

    def enforce_valid_input(self, f_samples: torch.Tensor, d_samples: torch.Tensor, p_lw_samples: torch.Tensor, F_lw_samples: torch.Tensor, height_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Enforce the input f, d, p_lw, F_lw to valid ranges. Ie. clip
            - f to [0,3] [Hz]
            - d to [0,1]
            - p_lw can't be clipped because already in lw frame
            - F_lw -> F_lw_z to [0, +inf]
        Moreover, ensure additionnal constraints
            - p_z on the ground (not implemented yet) TODO Implement
            - F_lw : Friction cone constraints

        Args
            f_samples    (torch.Tensor): Leg frequency samples               of shape(num_samples, num_leg)
            d_samples    (torch.Tensor): Leg duty cycle samples              of shape(num_samples, num_leg)
            p_lw_samples (torch.Tensor): Foot touch down position samples    of shape(num_samples, num_leg, 3, p_param)
            F_lw_samples (torch.Tensor): Ground Reaction forces samples      of shape(num_samples, num_leg, 3, F_param)
            height_map   (torch.Tensor): Height map arround the robot        of shape(x, y)

        Return
            f_samples    (torch.Tensor): Clipped Leg frequency samples               of shape(num_samples, num_leg)
            d_samples    (torch.Tensor): Clipped Leg duty cycle samples              of shape(num_samples, num_leg)
            p_lw_samples (torch.Tensor): Clipped Foot touch down position samples    of shape(num_samples, num_leg, 3, p_param)
            F_lw_samples (torch.Tensor): Clipped Ground Reaction forces samples      of shape(num_samples, num_leg, 3, F_param)
        """
        # --- Step 1 : Clip Action to valid range
        # Clip f
        f_samples = f_samples.clamp(min=0, max=3)

        # Clip d
        d_samples = d_samples.clamp(min=0, max=1)

        # Clip p - is already in lw frame... can't clip it in this frame
        # ...

        # Clip F
        F_lw_samples[:,:,2,:] = F_lw_samples[:,:,2,:].clamp(min=self.F_z_min, max=self.F_z_max) # TODO This clip also spline param 0 and 3, which may be restrictive


        # --- Step 2 : Add Constraints
        # p :  Ensure p on the ground TODO Implement
        # p_lw_samples[:,:,2,:] = 0.0*torch.ones_like(p_lw_samples[:,:,2,:])
      
        # F : Ensure Friction Cone constraints (Bounding spline means quasi bounding trajectory)
        F_lw_samples = self.enforce_friction_cone_constraints_torch(F=F_lw_samples, mu=self.mu)


        return f_samples, d_samples, p_lw_samples, F_lw_samples


    def enforce_friction_cone_constraints_torch(self, F:torch.Tensor, mu:float) -> torch.Tensor:
        """ Enforce the friction cone constraints
        ||F_xy|| < F_z*mu
        Args :
            F (torch.Tensor): The GRF                                    of shape(num_samples, num_legs, 3,(optinally F_param))

        Returns :
            F (torch.Tensor): The GRF with enforced friction constraints of shape(num_samples, num_legs, 3,(optinally F_param))
        """

        F_x = F[:,:,0].unsqueeze(2)
        F_y = F[:,:,1].unsqueeze(2)
        F_z = F[:,:,2].unsqueeze(2).clamp(min=self.F_z_min, max=self.F_z_max)

        # Angle between vec_x and vec_F_xy
        alpha = torch.atan2(F[:,:,1], F[:,:,0]).unsqueeze(2) # atan2(y,x) = arctan(y/x)

        # Compute the maximal Force in the xy plane
        F_xy_max = mu*F_z

        # Clipped the violation for the x and y component (unsqueeze to avoid to loose that dimension) : To use clamp_max -> need to remove the sign...
        F_x_clipped =  F_x.sign()*(torch.abs(F_x).clamp_max(torch.abs(torch.cos(alpha)*F_xy_max)))
        F_y_clipped =  F_y.sign()*(torch.abs(F_y).clamp_max(torch.abs(torch.sin(alpha)*F_xy_max)))

        # Reconstruct the vector
        F = torch.cat((F_x_clipped, F_y_clipped, F_z), dim=2)

        return F


    def from_zero_twopi_to_minuspi_pluspi(self, roll, pitch, yaw):
        """ Change the function space from [0, 2pi[ to ]-pi, pi] 
        
        Args :
            roll  (Tensor): roll in [0, 2pi[    shape(x)
            pitch (Tensor): roll in [0, 2pi[    shape(x)
            yaw   (Tensor): roll in [0, 2pi[    shape(x)
        
        Returns :   
            roll  (Tensor): roll in ]-pi, pi]   shape(x)
            pitch (Tensor): roll in ]-pi, pi]   shape(x)
            yaw   (Tensor): roll in ]-pi, pi]   shape(x)    
        """

        # Apply the transformation
        roll  = ((roll  - torch.pi) % (2*torch.pi)) - torch.pi
        pitch = ((pitch - torch.pi) % (2*torch.pi)) - torch.pi 
        yaw   = ((yaw   - torch.pi) % (2*torch.pi)) - torch.pi

        return roll, pitch, yaw
    

    def centroidal_model_step(self, state, input, contact):
        """
        TODO

        Note 
            Model used is described in 'Model Predictive Control With Environment Adaptation for Legged Locomotion'
            ttps://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9564053
            
        Args : 
            state     (dict): Dictionnary containing the robot's state
                pos_com_lw      (tensor): CoM position in local world frame                             of shape(num_samples, 3)
                lin_com_vel_lw  (tensor): CoM linear velocity in local world frame                      of shape(num_samples, 3)
                euler_xyz_angle (tensor): CoM orientation (wrt. to l. world frame) as XYZ euler angle   of shape(num_samples, 3)
                ang_vel_com_b   (tensor): CoM angular velocity as roll pitch yaw                        of shape(num_samples, 3)
                p_lw            (tensor): Feet position in local world frame                            of shape(num_samples, num_legs, 3)

            input     (dict): Dictionnary containing the robot's input
                p_lw            (tensor): Feet touch down position in local world frame                 of shape(num_samples, num_legs, 3)
                F_lw            (tensor): Ground Reaction Forces in local world frame                   of shape(num_samples, num_legs, 3)

            contact (tensor): Foot contact status (stance=1, swing=0)                                   of shape(num_samples, num_legs)

        Return :
            new_state (dict): Dictionnary containing the robot's updated state after one iteration
                pos_com_lw      (tensor): CoM position in local world frame                             of shape(num_samples, 3)
                lin_com_vel_lw  (tensor): CoM linear velocity in local world frame                      of shape(num_samples, 3)
                euler_xyz_angle (tensor): CoM orientation (wrt. to l. world frame) as XYZ euler angle   of shape(num_samples, 3)
                ang_vel_com_b   (tensor): CoM angular velocity as roll pitch yaw                        of shape(num_samples, 3)
                p_lw            (tensor): Feet position in local world frame                            of shape(num_samples, num_legs, 3)
        """
        new_state = {}

        # --- Step 1 : Compute linear velocity
        lin_com_vel_lw = state['lin_com_vel_lw']     # shape (num_samples, 3)


        # --- Step 2 : Compute linear acceleration  as sum of forces divide by mass
        linear_com_acc_lw = (torch.sum(input['F_lw'] * contact.unsqueeze(-1), dim=1) / self.robot_mass) + self.gravity_lw.unsqueeze(0) # shape (num_samples, 3)


        # --- Step 3 : Compute angular velocity (as euler rate to increment euler angles) : Given by eq. 27 in 'Note' paper
        # euler_xyz_rate = inverse(conjugate(euler_xyz_rate_matrix) * omega_b
        euler_xyz_rate = torch.bmm(inverse_conjugate_euler_xyz_rate_matrix(state['euler_xyz_angle']), state['ang_vel_com_b'].unsqueeze(-1)).squeeze(-1 ) # shape (s,3,3)*(s,3,1)->(s,3,1) -> (num_samples, 3)


        # --- Step 4 : Compute angular acceleration : Given by eq. 4 in 'Note' paper
        # Compute the sum of the moment induced by GRF (1. Moment=dist_from_F cross F, 2. Keep moment only for feet in contact, 3. sum over 4 legs)
        sum_of_GRF_moment_lw = torch.sum(contact.unsqueeze(-1) * torch.cross((state['p_lw'] - state['pos_com_lw'].unsqueeze(1)), input['F_lw'], dim=-1), dim=1) # shape sum{(s,l,1)*[(s,l,3)^(s,l,3)], dim=l} -> (num_sample, 3)

        # Transform the sum of moment from local world frame to base frame
        sum_of_GRF_moment_b  = torch.bmm(rotation_matrix_from_w_to_b(state['euler_xyz_angle']), sum_of_GRF_moment_lw.unsqueeze(-1)).squeeze(-1) # shape(num_samples, 3)

        # Compute intermediary variable : w_b^(I*w_b) : unsqueeze robot inertia for batched operation, squeeze and unsqueeze matmul to perform dot product (1,3,3)*(s,3,1)->(s,3,1)->(s,3)
        omega_b_cross_inertia_dot_omega_b = torch.cross(state['ang_vel_com_b'], torch.matmul(self.robot_inertia.unsqueeze(0), state['ang_vel_com_b'].unsqueeze(-1)).squeeze(-1), dim=-1) # shape(num_samples, 3)

        # Finally compute the angular acc : unsqueeze robot inertia for batched operation, squeeze and unsqueeze matmul to perform dot product (1,3,3)*(s,3,1)->(s,3,1)->(s,3)
        ang_acc_com_b = torch.matmul(self.inv_robot_inertia.unsqueeze(0), (sum_of_GRF_moment_b - (omega_b_cross_inertia_dot_omega_b)).unsqueeze(-1)).squeeze(-1) # shape(num_samples, 3)


        # --- Step 5 : Perform forward integration of the model (as simple forward euler)
        new_state['pos_com_lw']      = state['pos_com_lw']      + self.dt*lin_com_vel_lw
        new_state['lin_com_vel_lw']  = state['lin_com_vel_lw']  + self.dt*linear_com_acc_lw
        new_state['euler_xyz_angle'] = state['euler_xyz_angle'] + self.dt*euler_xyz_rate
        new_state['ang_vel_com_b']   = state['ang_vel_com_b']   + self.dt*ang_acc_com_b
        new_state['p_lw']            = state['p_lw']
        # new_state['p_lw']            = state['p_lw']*contact.unsqueeze(-1) + input['p_lw']*contact.unsqueeze(-1)

        return new_state
    



class OptimizerCfg():
    """ Config class for the optimizer """
    def __init__(self):
    
        self.optimizerType:str = 'sampling'
        """ Different type of optimizer. For now, only 'sampling' is implemented """

        self.prevision_horizon: int = cfg.mpc_params['horizon'] # 15
        """ Prevision horizon for predictive optimization (in number of time steps) """

        self.discretization_time: float = cfg.mpc_params['dt']#0.01 # 0.04
        """ Duration of a time step in seconds for the predicitve optimization """

        self.num_samples: int = cfg.mpc_params['num_parallel_computations'] #5000
        """ Number of samples used if the optimizerType is 'sampling' """

        self.parametrization_F = 'cubic spline'
        """ Define how F, Ground Reaction Forces, are encoded : can be 'discrete' or 'cubic spline', this modify F_param """

        self.parametrization_p = 'cubic spline'
        """ Define how p, foot touch down position, are encoded : can be 'discrete' or 'cubic spline', this modify p_param  """

        self.height_ref: float = 0.35 #0.38
        """ Height reference for the optimization, defined as mean distance between legs in contact and base """

        self.mu : float = cfg.mpc_params['mu']
        """ Coefficient of friction imposed for the friction cone constraints """

        self.optimize_f: bool = False
        """ If enabled, leg frequency will be optimized"""

        self.optimize_d: bool = False
        """ If enabled, duty cycle will be optimized"""

        self.optimize_p: bool = False
        """ If enabled, Foot step will be optimized"""

        self.optimize_F: bool = True
        """ If enabled, Ground Reaction Forces will be optimized"""

        self.propotion_previous_solution: float = 1.0
        """ Proportion of the previous solution that will be used to generate samples"""

        self.num_optimizer_iterations: int = cfg.mpc_params['num_sampling_iterations'] #1
        """ Number of time the sampling optiizer will iterate """

        self.sampling_law = 'normal'
        """ Sampling law to sample from in ['normal', 'uniform'] """

        self.clip_sample: bool = False 
        """ Wether to clip or not the samples to a range of the standard deviation """

        self.debug_apply_action = 'full stance' #None
        """ Wether to deactivate f,d,and p from RL and change that with another static gait"""













if __name__ == '__main__':

    robot_name = cfg.robot
    hip_height = cfg.hip_height
    robot_leg_joints = cfg.robot_leg_joints
    robot_feet_geom_names = cfg.robot_feet_geom_names
    scene_name = cfg.simulation_params['scene']
    simulation_dt = cfg.simulation_params['dt']

    state_observables = ('base_pos', 'base_lin_vel', 'base_ori_quat_wxyz', 'base_ang_vel',
                         'qpos_js', 'qvel_js', 'tau_applied',
                         'feet_pos_base', 'feet_vel_base', 'contact_state', 'contact_forces_base',)

    # Create the quadruped robot environment. _______________________________________________________________________
    env = QuadrupedEnv(robot=robot_name,
                       hip_height=hip_height,
                       legs_joint_names=LegsAttr(**robot_leg_joints),
                       scene=scene_name,
                       sim_dt=simulation_dt,
                       base_lin_vel_range=(-0.0 * hip_height, 0.0 * hip_height),#(-4.0 * hip_height, 4.0 * hip_height),
                       base_ang_vel_range=(-0.0, 0.0),#(-np.pi * 3 / 4, np.pi * 3 / 4),
                       ground_friction_coeff_range=(0.6,0.6),#(0.3, 1.5),
                       base_vel_command_type="human",  # "forward", "random", "human"
                       feet_geom_name=LegsAttr(**robot_feet_geom_names),  # Geom/Frame id of feet
                       state_obs_names=state_observables,
                       )
    env.reset()
    env.render()  # Pass in the first render call any mujoco.viewer.KeyCallbackType
    mass = np.sum(env.mjModel.body_mass)

    MAX_STEPS = 2000 if env.base_vel_command_type != "human" else 20000

    #   ... __________________________________________________________________________________________________
    mpc_frequency = cfg.simulation_params['mpc_frequency']
    mpc_dt = cfg.mpc_params['dt']
    horizon = cfg.mpc_params['horizon']


    if cfg.mpc_params['type'] == 'sampling':

        num_parallel_computations = cfg.mpc_params['num_parallel_computations']  # num_samples
        iteration = cfg.mpc_params['num_sampling_iterations']

        device = "cuda:0"
        num_legs = 4
        optimizerCfg = OptimizerCfg()

        controller = SamplingOptimizer(device=device,num_legs=num_legs, optimizerCfg=optimizerCfg)

        f_fake = torch.empty((1,num_legs),device=device)
        d_fake = torch.empty((1,num_legs),device=device)
        p_fake = torch.empty((1,num_legs, 3, 4),device=device)
        F_fake = torch.empty((1,num_legs, 3, 4),device=device)
        height_map = torch.empty((1,num_legs),device=device)
        phase = torch.zeros((1,num_legs), device=device)
        c_prev = torch.ones((1,4), device=device)


    # Periodic gait generator _________________________________________________________________________
    gait_name = cfg.simulation_params['gait']
    gait_type, duty_factor, step_frequency = get_gait_params(gait_name)
    # Given the possibility to use nonuniform discretization, 
    # we generate a contact sequence two times longer and with a dt half of the one of the mpc
    pgg = PeriodicGaitGenerator(duty_factor=duty_factor, step_freq=step_frequency, gait_type=gait_type,
                                horizon=horizon * 2, contact_sequence_dt=mpc_dt/2.)
    contact_sequence = pgg.compute_contact_sequence()
    nominal_sample_freq = step_frequency
    
    # Create the foothold reference generator
    stance_time = (1 / step_frequency) * duty_factor
    frg = FootholdReferenceGenerator(stance_time=stance_time, hip_height=cfg.hip_height, lift_off_positions=env.feet_pos(frame='world'))


    # Create swing trajectory generator
    step_height = cfg.simulation_params['step_height']
    swing_period = (1 - duty_factor) * (1 / step_frequency)  # + 0.07
    position_gain_fb = cfg.simulation_params['swing_position_gain_fb']
    velocity_gain_fb = cfg.simulation_params['swing_velocity_gain_fb']
    swing_generator = cfg.simulation_params['swing_generator']
    stc = SwingTrajectoryController(step_height=step_height, swing_period=swing_period,
                                    position_gain_fb=position_gain_fb, velocity_gain_fb=velocity_gain_fb,
                                    generator=swing_generator)

    

    # Terrain estimator
    terrain_computation = TerrainEstimator()

    # Online computation of the inertia parameter
    srb_inertia_computation = SrbInertiaComputation()  # TODO: This seems to be unsused.

    # Initialization of variables used in the main control loop
    # ____________________________________________________________
    # Set the reference for the state
    ref_pose = np.array([0, 0, cfg.hip_height])
    ref_base_lin_vel, ref_base_ang_vel = env.target_base_vel()
    ref_orientation = np.array([0.0, 0.0, 0.0])
    # # SET REFERENCE AS DICTIONARY
    # TODO: I would suggest to create a DataClass for "BaseConfig" used in the PotatoModel controllers.
    ref_state = {}

    # Starting contact sequence
    previous_contact = np.array([1, 1, 1, 1])
    previous_contact_mpc = np.array([1, 1, 1, 1])
    current_contact = np.array([1, 1, 1, 1])

    nmpc_GRFs = np.zeros((12,))
    nmpc_wrenches = np.zeros((6,))
    nmpc_footholds = np.zeros((12,))

    # Jacobian matrices
    jac_feet_prev = LegsAttr(*[np.zeros((3, env.mjModel.nv)) for _ in range(4)])
    jac_feet_dot = LegsAttr(*[np.zeros((3, env.mjModel.nv)) for _ in range(4)])
    # Torque vector
    tau = LegsAttr(*[np.zeros((env.mjModel.nv, 1)) for _ in range(4)])
    # State
    state_current, state_prev = {}, {}
    feet_pos = None
    feet_traj_geom_ids, feet_GRF_geom_ids = None, LegsAttr(FL=-1, FR=-1, RL=-1, RR=-1)
    legs_order = ["FL", "FR", "RL", "RR"]

    RENDER_FREQ = 30  # Hz
    last_render_time = time.time()

    height_list = [0.0]
    height_ref_list = [0.0]
    cost_list = [0.0]
    FL_foot_list = [np.array([0.0, 0.0, 0.0])]
    FR_foot_list = [np.array([0.0, 0.0, 0.0])]
    RL_foot_list = [np.array([0.0, 0.0, 0.0])]
    RR_foot_list = [np.array([0.0, 0.0, 0.0])]
    FL_foot_ref_list = [np.array([0.0, 0.0, 0.0])]
    FR_foot_ref_list = [np.array([0.0, 0.0, 0.0])]
    RL_foot_ref_list = [np.array([0.0, 0.0, 0.0])]
    RR_foot_ref_list = [np.array([0.0, 0.0, 0.0])]
    des_foot_pos = LegsAttr(*[np.zeros(3) for _ in range(4)])

    while True:
        step_start = time.time()
        # breakpoint()

        # Update the robot state --------------------------------
        feet_pos = env.feet_pos(frame='world')
        hip_pos = env.hip_positions(frame='world')

        state_current = dict(
            position=env.base_pos,
            linear_velocity=env.base_lin_vel,
            orientation=env.base_ori_euler_xyz,
            angular_velocity=env.base_ang_vel,
            foot_FL=feet_pos.FL,
            foot_FR=feet_pos.FR,
            foot_RL=feet_pos.RL,
            foot_RR=feet_pos.RR
            )
        height_list.append(state_current['position'][2])
        if len(height_list) > 100 : height_list.pop(0)
        np.savetxt('live_variable/height.csv', [height_list], delimiter=',', fmt='%.3f')

        FL_foot_list.append(state_current['foot_FL'])
        if len(FL_foot_list) > 100 : FL_foot_list.pop(0)
        np.savetxt('live_variable/FL_foot.csv', FL_foot_list, delimiter=',', fmt='%.3f')
        FR_foot_list.append(state_current['foot_FR'])
        if len(FR_foot_list) > 100 : FR_foot_list.pop(0)
        np.savetxt('live_variable/FR_foot.csv', FR_foot_list, delimiter=',', fmt='%.3f')
        RL_foot_list.append(state_current['foot_RL'])
        if len(RL_foot_list) > 100 : RL_foot_list.pop(0)
        np.savetxt('live_variable/RL_foot.csv', RL_foot_list, delimiter=',', fmt='%.3f')
        RR_foot_list.append(state_current['foot_RR'])
        if len(RR_foot_list) > 100 : RR_foot_list.pop(0)
        np.savetxt('live_variable/RR_foot.csv', RR_foot_list, delimiter=',', fmt='%.3f')

        # Update target base velocity
        ref_base_lin_vel, ref_base_ang_vel = env.target_base_vel()
        # -------------------------------------------------------

        # Update the desired contact sequence ---------------------------
        pgg.run(simulation_dt, pgg.step_freq)
        contact_sequence = pgg.compute_contact_sequence()

        # in the case of nonuniform discretization, we need to subsample the contact sequence
        if (cfg.mpc_params['use_nonuniform_discretization']):
            dt_fine_grained = cfg.mpc_params['dt_fine_grained']
            horizon_fine_grained = cfg.mpc_params['horizon_fine_grained']
            contact_sequence = pgg.sample_contact_sequence(contact_sequence, mpc_dt, dt_fine_grained, horizon_fine_grained)
        

        previous_contact = current_contact
        current_contact = np.array([contact_sequence[0][0],
                                    contact_sequence[1][0],
                                    contact_sequence[2][0],
                                    contact_sequence[3][0]])

        # Compute the reference for the footholds ---------------------------------------------------
        frg.update_lift_off_positions(previous_contact, current_contact, feet_pos, legs_order)
        ref_feet_pos = frg.compute_footholds_reference(
            com_position=env.base_pos,
            base_ori_euler_xyz=env.base_ori_euler_xyz,
            base_xy_lin_vel=env.base_lin_vel[0:2],
            ref_base_xy_lin_vel=ref_base_lin_vel[0:2],
            hips_position=hip_pos)

        # Estimate the terrain slope and elevation -------------------------------------------------------
        terrain_roll, \
            terrain_pitch, \
            terrain_height = terrain_computation.compute_terrain_estimation(
            base_position=env.base_pos,
            yaw=env.base_ori_euler_xyz[2],
            feet_pos=frg.lift_off_positions,
            current_contact=current_contact)

        ref_pos = np.array([0, 0, cfg.hip_height])
        ref_pos[2] = cfg.simulation_params['ref_z'] + terrain_height

        # print('Ref Height : ', ref_pos[2])
        height_ref_list.append(ref_pos[2])
        if len(height_ref_list) > 100 : height_ref_list.pop(0)
        np.savetxt('live_variable/height_ref.csv', [height_ref_list], delimiter=',', fmt='%.3f')


        # Update state reference ------------------------------------------------------------------------
        ref_state |= dict(ref_foot_FL=ref_feet_pos.FL.reshape((1, 3)),
                          ref_foot_FR=ref_feet_pos.FR.reshape((1, 3)),
                          ref_foot_RL=ref_feet_pos.RL.reshape((1, 3)),
                          ref_foot_RR=ref_feet_pos.RR.reshape((1, 3)),
                          # Also update the reference base linear velocity and
                          ref_linear_velocity=ref_base_lin_vel,
                          ref_angular_velocity=ref_base_ang_vel,
                          ref_orientation=np.array([terrain_roll, terrain_pitch, 0.0]),
                          ref_position=ref_pos
                          )
        # -------------------------------------------------------------------------------------------------


        # TODO: this should be hidden inside the controller forward/get_action method
        # Solve OCP ---------------------------------------------------------------------------------------
        if env.step_num % round(1 / (mpc_frequency * simulation_dt)) == 0:

            # We can recompute the inertia of the single rigid body model

            # or use the fixed one in cfg.py
            if(cfg.simulation_params['use_inertia_recomputation']):
                # TODO: d.qpos is not defined
                #inertia = srb_inertia_computation.compute_inertia(d.qpos)
                inertia = env.get_base_inertia().flatten()  # Reflected inertia of base at qpos, in world frame
            else:
                inertia = cfg.inertia.flatten()

            # If we use sampling
            if (cfg.mpc_params['type'] == 'sampling'):

                time_start = time.time()

                f_star, d_star, p0_star_lw, F0_star_lw = controller.optimize_latent_variable(state_current=state_current,ref_state=ref_state, f=f_fake, d=d_fake, p_lw=p_fake, F_lw=F_fake, phase=phase, c_prev=c_prev, height_map=height_map)
                
                # update phase and contact sequence
                f_samples = torch.tensor(([[1.4, 1.4, 1.4, 1.4]]), device=controller.device)
                d_samples = torch.tensor(([[1.0, 1.0, 1.0, 1.0]]), device=controller.device)
                c_samples2, new_phase_samples = controller.gait_generator( f_samples, d_samples, phase.squeeze(0), 1, simulation_dt)
                phase = new_phase_samples
                # c_prev=c_samples2.squeeze(-1)
                c_prev = torch.from_numpy(current_contact).to(dtype=torch.bool,device=controller.device).unsqueeze(0)

                nmpc_footholds = ref_feet_pos

                nmpc_GRFs = np.array(F0_star_lw.flatten(0,-1).cpu().numpy())

                previous_contact_mpc = current_contact

                print('sampling time : ', time.time() - time_start)

        


    
           # TODO: Indexing should not be hardcoded. Env should provide indexing of leg actuator dimensions.
            nmpc_GRFs = LegsAttr(FL=nmpc_GRFs[0:3] * current_contact[0],
                                 FR=nmpc_GRFs[3:6] * current_contact[1],
                                 RL=nmpc_GRFs[6:9] * current_contact[2],
                                 RR=nmpc_GRFs[9:12] * current_contact[3])

        # -------------------------------------------------------------------------------------------------

        # Compute Stance Torque ---------------------------------------------------------------------------
        feet_jac = env.feet_jacobians(frame='world', return_rot_jac=False)
        # Compute feet velocities
        feet_vel = LegsAttr(**{leg_name: feet_jac[leg_name] @ env.mjData.qvel for leg_name in legs_order})
        # Compute jacobian derivatives of the contact points
        jac_feet_dot = (feet_jac - jac_feet_prev) / simulation_dt  # Finite difference approximation
        jac_feet_prev = feet_jac  # Update previous Jacobians
        # Compute the torque with the contact jacobian (-J.T @ f)   J: R^nv -> R^3,   f: R^3
        tau.FL = -np.matmul(feet_jac.FL[:, env.legs_qvel_idx.FL].T, nmpc_GRFs.FL)
        tau.FR = -np.matmul(feet_jac.FR[:, env.legs_qvel_idx.FR].T, nmpc_GRFs.FR)
        tau.RL = -np.matmul(feet_jac.RL[:, env.legs_qvel_idx.RL].T, nmpc_GRFs.RL)
        tau.RR = -np.matmul(feet_jac.RR[:, env.legs_qvel_idx.RR].T, nmpc_GRFs.RR)
        # ---------------------------------------------------------------------------------------------------

        # Compute Swing Torque ------------------------------------------------------------------------------
        # TODO: Move contact sequence to labels FL, FR, RL, RR instead of a fixed indexing.
        # The swing controller is in the end-effector space. For its computation,
        # we save for simplicity joints position and velocities
        qpos, qvel = env.mjData.qpos, env.mjData.qvel
        # centrifugal, coriolis, gravity
        legs_mass_matrix = env.legs_mass_matrix()
        legs_qfrc_bias = env.legs_qfrc_bias()

        
        stc.update_swing_time(current_contact, legs_order, simulation_dt)

        for leg_id, leg_name in enumerate(legs_order):
            if current_contact[leg_id] == 0:  # If in swing phase, compute the swing trajectory tracking control.
                tau[leg_name], des_foot_pos[leg_name], _ = stc.compute_swing_control(
                    leg_id=leg_id,
                    q_dot=qvel[env.legs_qvel_idx[leg_name]],
                    J=feet_jac[leg_name][:, env.legs_qvel_idx[leg_name]],
                    J_dot=jac_feet_dot[leg_name][:, env.legs_qvel_idx[leg_name]],
                    lift_off=frg.lift_off_positions[leg_name],
                    touch_down=nmpc_footholds[leg_name],
                    foot_pos=feet_pos[leg_name],
                    foot_vel=feet_vel[leg_name],
                    h=legs_qfrc_bias[leg_name],
                    mass_matrix=legs_mass_matrix[leg_name]
                    )
        # ---------------------------------------------------------------------------------------------------
        # Set control and mujoco step ----------------------------------------------------------------------
        action = np.zeros(env.mjModel.nu)
        action[env.legs_tau_idx.FL] = tau.FL
        action[env.legs_tau_idx.FR] = tau.FR
        action[env.legs_tau_idx.RL] = tau.RL
        action[env.legs_tau_idx.RR] = tau.RR
        action_noise = np.random.normal(0, 2, size=env.mjModel.nu)

        state, reward, is_terminated, is_truncated, info = env.step(action=action + action_noise)

        state_obs = env.extract_obs_from_state(state)
        feet_contact_state, _, feet_GRF = env.feet_contact_state(ground_reaction_forces=True)

        # Render only at a certain frequency
        if time.time() - last_render_time > 1.0 / RENDER_FREQ or env.step_num == 1:
            feet_traj_geom_ids = plot_swing_mujoco(viewer=env.viewer,
                                                   swing_traj_controller=stc,
                                                   swing_period=swing_period,
                                                   swing_time=LegsAttr(FL=stc.swing_time[0],
                                                                       FR=stc.swing_time[1],
                                                                       RL=stc.swing_time[2],
                                                                       RR=stc.swing_time[3]),
                                                   lift_off_positions=frg.lift_off_positions,
                                                   nmpc_footholds=nmpc_footholds,
                                                   ref_feet_pos=ref_feet_pos,
                                                   geom_ids=feet_traj_geom_ids)
            for leg_id, leg_name in enumerate(legs_order):
                feet_GRF_geom_ids[leg_name] = render_vector(env.viewer,
                                                            vector=feet_GRF[leg_name],
                                                            pos=feet_pos[leg_name],
                                                            scale=np.linalg.norm(feet_GRF[leg_name]) * 0.005,
                                                            color=np.array([0, 1, 0, .5]),
                                                            geom_id=feet_GRF_geom_ids[leg_name])

            env.render()
            last_render_time = time.time()

        if env.step_num > MAX_STEPS or is_terminated or is_truncated:
            if is_terminated:
                print("Environment terminated")
            env.reset()
            pgg.reset()
            frg.lift_off_positions = env.feet_pos(frame='world')
            current_contact = np.array([0, 0, 0, 0])
            previous_contact = np.asarray(current_contact)
            z_foot_mean = 0.0
        # print("loop time: ", time.time() - step_start)

        # Save reference value for the leg - plotting 
        if current_contact[0] == 1: # Stance
            FL_foot_ref_list.append(ref_state['ref_foot_FL'][0,:])
        else : # Swing
            FL_foot_ref_list.append(des_foot_pos['FL'])

        if current_contact[1] == 1: # Stance
            FR_foot_ref_list.append(ref_state['ref_foot_FR'][0,:])
        else : # Swing
            FR_foot_ref_list.append(des_foot_pos['FR'])

        if current_contact[2] == 1: # Stance
            RL_foot_ref_list.append(ref_state['ref_foot_RL'][0,:])
        else : # Swing
            RL_foot_ref_list.append(des_foot_pos['RL'])

        if current_contact[3] == 1: # Stance
            RR_foot_ref_list.append(ref_state['ref_foot_RR'][0,:])
        else : # Swing
            RR_foot_ref_list.append(des_foot_pos['RR'])


        if len(FL_foot_ref_list) > 100 : FL_foot_ref_list.pop(0)
        np.savetxt('live_variable/FL_foot_ref.csv', FL_foot_ref_list, delimiter=',', fmt='%.3f')
        if len(FR_foot_ref_list) > 100 : FR_foot_ref_list.pop(0)
        np.savetxt('live_variable/FR_foot_ref.csv', FR_foot_ref_list, delimiter=',', fmt='%.3f')
        if len(RL_foot_ref_list) > 100 : RL_foot_ref_list.pop(0)
        np.savetxt('live_variable/RL_foot_ref.csv', RL_foot_ref_list, delimiter=',', fmt='%.3f')
        if len(RR_foot_ref_list) > 100 : RR_foot_ref_list.pop(0)
        np.savetxt('live_variable/RR_foot_ref.csv', RR_foot_ref_list, delimiter=',', fmt='%.3f')

        pass

