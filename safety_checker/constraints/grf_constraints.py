import numpy as np
from typing import Dict, List, Union, Optional
from ..core.barrier_functions import BarrierFunction, BarrierConfig
from ..core.safety_level import ConstraintStatus, ConstraintPriority

class GRFConstraint:
    """Base class for ground reaction force constraints"""
    def __init__(self, config: Dict):
        self.config = config
        self.enabled = config.get('enabled', True)
        self.name = "grf_constraint"
        self._init_barrier_functions()

    def _init_barrier_functions(self):
        """Initialize barrier functions"""
        raise NotImplementedError

    def check(self, state_dict: Dict) -> ConstraintStatus:
        """Check constraint satisfaction"""
        raise NotImplementedError

class ContactForceConstraint(GRFConstraint):
    """Constraint on contact force magnitude"""
    def __init__(self, config: Dict):
        super().__init__(config)
        self.name = "contact_force"
        self.num_legs = 4

    def _init_barrier_functions(self):
        self.max_force_config = BarrierConfig(
            threshold=self.config['thresholds']['force_max'],
            warning_threshold=self.config['thresholds']['warning'],
            danger_threshold=self.config['thresholds']['danger'],
            emergency_threshold=self.config['thresholds']['emergency'],
            weight=self.config['weights'][0]
        )
        self.max_force_barrier = BarrierFunction(self.max_force_config)
        
        self.min_force_config = BarrierConfig(
            threshold=self.config['thresholds'].get('force_min', 0.0),
            warning_threshold=self.config['thresholds']['warning'],
            danger_threshold=self.config['thresholds']['danger'],
            emergency_threshold=self.config['thresholds']['emergency'],
            weight=self.config['weights'][0]
        )
        self.min_force_barrier = BarrierFunction(self.min_force_config)

    def check(self, state_dict: Dict) -> List[ConstraintStatus]:
        if not self.enabled:
            return [ConstraintStatus(
                name=f"{self.name}_leg_{i}",
                enabled=False,
                priority=ConstraintPriority.HIGH,
                value=0.0,
                threshold=float('inf'),
                violation=0.0,
                is_violated=False
            ) for i in range(self.num_legs)]

        contact_forces = state_dict['contact_forces_base']  # Shape: (4, 3)
        contact_state = state_dict['contact_state']  # Shape: (4,)
        
        statuses = []
        for i in range(self.num_legs):
            if contact_state[i]:
                force_norm = np.linalg.norm(contact_forces[i])
                
                # Check both minimum and maximum force constraints
                max_value = self.max_force_barrier.evaluate(force_norm)
                min_value = self.min_force_barrier.evaluate(force_norm)
                
                value = min(max_value, min_value)
                threshold = self.max_force_config.threshold
            else:
                # No constraint when foot is not in contact
                value = float('inf')
                threshold = float('inf')

            statuses.append(ConstraintStatus(
                name=f"{self.name}_leg_{i}",
                enabled=True,
                priority=ConstraintPriority.HIGH,
                value=value,
                threshold=threshold,
                violation=max(0, threshold - value),
                is_violated=value < 0
            ))

        return statuses

class FrictionConeConstraint(GRFConstraint):
    """摩擦锥约束"""
    def __init__(self, config: Dict):
        self.name = "friction_cone"
        self.num_legs = 4
        self.friction_coef = config['friction_coefficient']
        super().__init__(config)

    def _init_barrier_functions(self):
        self.barrier_config = BarrierConfig(
            threshold=np.arctan(self.friction_coef),  # 摩擦锥角度
            warning_threshold=self.config['thresholds']['warning'],
            danger_threshold=self.config['thresholds']['danger'],
            emergency_threshold=self.config['thresholds']['emergency'],
            weight=self.config['weights'][0]
        )
        self.barrier = BarrierFunction(self.barrier_config)

    def check(self, state_dict: Dict) -> List[ConstraintStatus]:
        if not self.enabled:
            return [ConstraintStatus(
                name=f"{self.name}_leg_{i}",
                enabled=False,
                priority=ConstraintPriority.HIGH,
                value=0.0,
                threshold=float('inf'),
                violation=0.0,
                is_violated=False
            ) for i in range(self.num_legs)]

        # 确保接触力数据格式正确
        contact_forces = np.array(state_dict['contact_forces_base'])
        if len(contact_forces.shape) == 1:
            # 如果是一维数组，假设它是按[fx1,fy1,fz1,fx2,fy2,fz2,...]排列
            contact_forces = contact_forces.reshape(-1, 3)
            
        contact_state = np.array(state_dict['contact_state'])
        if len(contact_state.shape) == 0:
            contact_state = np.array([contact_state])

        statuses = []
        for i in range(self.num_legs):
            try:
                if contact_state[i]:
                    force = contact_forces[i]
                    # 计算力与垂直方向的夹角
                    horizontal_force = np.linalg.norm(force[:2])  # xy平面上的力
                    vertical_force = force[2]  # z方向的力
                    force_angle = np.arctan2(horizontal_force, vertical_force)
                    value = self.barrier.evaluate(force_angle)
                    threshold = self.barrier_config.threshold
                else:
                    value = float('inf')
                    threshold = float('inf')

                statuses.append(ConstraintStatus(
                    name=f"{self.name}_leg_{i}",
                    enabled=True,
                    priority=ConstraintPriority.HIGH,
                    value=value,
                    threshold=threshold,
                    violation=max(0, threshold - value),
                    is_violated=value < 0
                ))
                
            except Exception as e:
                print(f"处理腿 {i} 的摩擦锥约束时出错: {str(e)}")
                print(f"contact_forces shape: {contact_forces.shape}")
                print(f"contact_state shape: {contact_state.shape}")
                print(f"force: {force if 'force' in locals() else 'N/A'}")
                raise

        return statuses

class ForceBalanceConstraint(GRFConstraint):
    """Constraint for total force and moment balance"""
    def __init__(self, config: Dict):
        
        self.name = "force_balance"
        self.num_legs = 4
        super().__init__(config)
    def _init_barrier_functions(self):
        self.force_barrier_config = BarrierConfig(
            threshold=self.config['thresholds']['force_imbalance_max'],
            warning_threshold=self.config['thresholds']['warning'],
            danger_threshold=self.config['thresholds']['danger'],
            emergency_threshold=self.config['thresholds']['emergency'],
            weight=self.config['weights'][0]
        )
        self.force_barrier = BarrierFunction(self.force_barrier_config)
        
        self.moment_barrier_config = BarrierConfig(
            threshold=self.config['thresholds']['moment_imbalance_max'],
            warning_threshold=self.config['thresholds']['warning'],
            danger_threshold=self.config['thresholds']['danger'],
            emergency_threshold=self.config['thresholds']['emergency'],
            weight=self.config['weights'][1]
        )
        self.moment_barrier = BarrierFunction(self.moment_barrier_config)

    def check(self, state_dict: Dict) -> ConstraintStatus:
        if not self.enabled:
            return ConstraintStatus(
                name=self.name,
                enabled=False,
                priority=ConstraintPriority.HIGH,
                value=0.0,
                threshold=float('inf'),
                violation=0.0,
                is_violated=False
            )

        # contact_forces = state_dict['contact_forces_base']  # Shape: (4, 3)
        # feet_positions = state_dict['feet_pos_base']  # Shape: (4, 3)
        # contact_state = state_dict['contact_state']  # Shape: (4,)
        
        # Resize the contact_forces and feet_positions to (num_legs, 3)
        contact_forces = np.array(state_dict['contact_forces_base']).reshape(self.num_legs, 3)
        feet_positions = np.array(state_dict['feet_pos_base']).reshape(self.num_legs, 3)
        contact_state = np.array(state_dict['contact_state'])


        # Calculate total force imbalance
        total_force = np.sum(contact_forces[contact_state], axis=0)
        force_imbalance = np.linalg.norm(total_force[:2])  # Only xy components
        
        # Calculate total moment imbalance
        total_moment = np.zeros(3)
        for i in range(4):
            if contact_state[i]:
                moment = np.cross(feet_positions[i], contact_forces[i])
                total_moment += moment
        moment_imbalance = np.linalg.norm(total_moment[:2])  # Only xy components
        
        # Evaluate both constraints
        force_value = self.force_barrier.evaluate(force_imbalance)
        moment_value = self.moment_barrier.evaluate(moment_imbalance)
        
        value = min(force_value, moment_value)
        threshold = min(self.force_barrier_config.threshold,
                       self.moment_barrier_config.threshold)

        return ConstraintStatus(
            name=self.name,
            enabled=True,
            priority=ConstraintPriority.HIGH,
            value=value,
            threshold=threshold,
            violation=max(0, threshold - value),
            is_violated=value < 0
        )

class GRFConstraintModule:
    """Module combining all GRF-related constraints"""
    def __init__(self, config: Dict):
        self.contact_force_constraint = ContactForceConstraint(config['contact_force'])
        self.friction_cone_constraint = FrictionConeConstraint(config['friction_cone'])
        self.force_balance_constraint = ForceBalanceConstraint(config['force_balance'])
        
    def check_all(self, state_dict: Dict) -> Dict[str, Union[ConstraintStatus, List[ConstraintStatus]]]:
        """Check all GRF constraints"""
        return {
            'contact_force': self.contact_force_constraint.check(state_dict),
            'friction_cone': self.friction_cone_constraint.check(state_dict),
            'force_balance': self.force_balance_constraint.check(state_dict)
        }

    def get_most_critical(self, state_dict: Dict) -> ConstraintStatus:
        """Get the most critical constraint status"""
        all_statuses = self.check_all(state_dict)
        
        # Flatten all constraints
        all_statuses_flat = (
            all_statuses['contact_force'] +
            all_statuses['friction_cone'] +
            [all_statuses['force_balance']]
        )
        
        # Find most critical
        return min(all_statuses_flat, key=lambda x: x.value)