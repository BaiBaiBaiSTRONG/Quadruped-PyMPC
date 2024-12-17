import numpy as np
from typing import Dict, List, Union
from ..core.barrier_functions import BarrierFunction, BarrierConfig
from ..core.safety_level import ConstraintStatus, ConstraintPriority

class TrackingConstraint:
    """Base class for tracking-related constraints"""
    def __init__(self, config: Dict):
        self.config = config
        self.enabled = config.get('enabled', True)
        self.name = "tracking_constraint"
        self._init_barrier_functions()

    def _init_barrier_functions(self):
        """Initialize barrier functions"""
        raise NotImplementedError

    def check(self, state_dict: Dict) -> ConstraintStatus:
        """Check constraint satisfaction"""
        raise NotImplementedError

class BasePositionTrackingConstraint(TrackingConstraint):
    """Constraint on base position tracking error"""
    def __init__(self, config: Dict):
        super().__init__(config)
        self.name = "base_position_tracking"

    def _init_barrier_functions(self):
        self.xy_barrier_config = BarrierConfig(
            threshold=self.config['thresholds']['max_error_xy'],
            warning_threshold=self.config['thresholds']['warning'],
            danger_threshold=self.config['thresholds']['danger'],
            emergency_threshold=self.config['thresholds']['emergency'],
            weight=np.mean(self.config['weights'][:2])
        )
        self.xy_barrier = BarrierFunction(self.xy_barrier_config)
        
        self.z_barrier_config = BarrierConfig(
            threshold=self.config['thresholds']['max_error_z'],
            warning_threshold=self.config['thresholds']['warning'],
            danger_threshold=self.config['thresholds']['danger'],
            emergency_threshold=self.config['thresholds']['emergency'],
            weight=self.config['weights'][2]
        )
        self.z_barrier = BarrierFunction(self.z_barrier_config)

    def check(self, state_dict: Dict) -> ConstraintStatus:
        if not self.enabled:
            return ConstraintStatus(
                name=self.name,
                enabled=False,
                priority=ConstraintPriority.MEDIUM,
                value=0.0,
                threshold=float('inf'),
                violation=0.0,
                is_violated=False
            )

        current_height = state_dict['base_pos'][2]
        ref_height = state_dict['ref_base_height']
        
        # Height tracking error
        z_error = abs(current_height - ref_height)
        z_value = self.z_barrier.evaluate(z_error)
        
        # XY tracking is optional (if reference is provided)
        if 'ref_base_position_xy' in state_dict:
            xy_error = np.linalg.norm(
                state_dict['base_pos'][:2] - state_dict['ref_base_position_xy']
            )
            xy_value = self.xy_barrier.evaluate(xy_error)
            value = min(xy_value, z_value)
            threshold = min(
                self.xy_barrier_config.threshold,
                self.z_barrier_config.threshold
            )
        else:
            value = z_value
            threshold = self.z_barrier_config.threshold

        return ConstraintStatus(
            name=self.name,
            enabled=True,
            priority=ConstraintPriority.MEDIUM,
            value=value,
            threshold=threshold,
            violation=max(0, threshold - value),
            is_violated=value < 0
        )

class BaseOrientationTrackingConstraint(TrackingConstraint):
    """Constraint on base orientation tracking error"""
    def __init__(self, config: Dict):
        super().__init__(config)
        self.name = "base_orientation_tracking"

    def _init_barrier_functions(self):
        self.barrier_config = BarrierConfig(
            threshold=self.config['thresholds']['max_error'],
            warning_threshold=self.config['thresholds']['warning'],
            danger_threshold=self.config['thresholds']['danger'],
            emergency_threshold=self.config['thresholds']['emergency'],
            weight=np.mean(self.config['weights'])
        )
        self.barrier = BarrierFunction(self.barrier_config)

    def check(self, state_dict: Dict) -> ConstraintStatus:
        if not self.enabled:
            return ConstraintStatus(
                name=self.name,
                enabled=False,
                priority=ConstraintPriority.MEDIUM,
                value=0.0,
                threshold=float('inf'),
                violation=0.0,
                is_violated=False
            )

        current_angles = state_dict['base_ori_euler_xyz']
        ref_angles = state_dict['ref_base_angles']
        
        # Calculate orientation error (considering angle wrapping)
        angle_error = np.array([
            min(abs(current_angles[i] - ref_angles[i]),
                2*np.pi - abs(current_angles[i] - ref_angles[i]))
            for i in range(3)
        ])
        error_norm = np.linalg.norm(angle_error)
        
        value = self.barrier.evaluate(error_norm)
        threshold = self.barrier_config.threshold

        return ConstraintStatus(
            name=self.name,
            enabled=True,
            priority=ConstraintPriority.MEDIUM,
            value=value,
            threshold=threshold,
            violation=max(0, threshold - value),
            is_violated=value < 0
        )

class FootPositionTrackingConstraint(TrackingConstraint):
    """Constraint on foot position tracking error"""
    def __init__(self, config: Dict):
        super().__init__(config)
        self.name = "foot_position_tracking"
        self.num_legs = 4

    def _init_barrier_functions(self):
        self.swing_barrier_config = BarrierConfig(
            threshold=self.config['thresholds']['swing_error_max'],
            warning_threshold=self.config['thresholds']['warning'],
            danger_threshold=self.config['thresholds']['danger'],
            emergency_threshold=self.config['thresholds']['emergency'],
            weight=self.config['weights'][0]
        )
        self.swing_barrier = BarrierFunction(self.swing_barrier_config)
        
        self.stance_barrier_config = BarrierConfig(
            threshold=self.config['thresholds']['stance_error_max'],
            warning_threshold=self.config['thresholds']['warning'],
            danger_threshold=self.config['thresholds']['danger'],
            emergency_threshold=self.config['thresholds']['emergency'],
            weight=self.config['weights'][1]
        )
        self.stance_barrier = BarrierFunction(self.stance_barrier_config)

    def check(self, state_dict: Dict) -> List[ConstraintStatus]:
        if not self.enabled:
            return [ConstraintStatus(
                name=f"{self.name}_leg_{i}",
                enabled=False,
                priority=ConstraintPriority.MEDIUM,
                value=0.0,
                threshold=float('inf'),
                violation=0.0,
                is_violated=False
            ) for i in range(self.num_legs)]

        current_positions = state_dict['feet_pos_base']
        ref_positions = state_dict['ref_feet_pos']
        contact_state = state_dict['contact_state']
        
        statuses = []
        leg_order = ['FL', 'FR', 'RL', 'RR']
        for i, leg in enumerate(leg_order):
            error = np.linalg.norm(current_positions[i] - ref_positions[leg])
            
            if contact_state[i]:  # Stance phase
                value = self.stance_barrier.evaluate(error)
                threshold = self.stance_barrier_config.threshold
            else:  # Swing phase
                value = self.swing_barrier.evaluate(error)
                threshold = self.swing_barrier_config.threshold

            statuses.append(ConstraintStatus(
                name=f"{self.name}_leg_{leg}",  # 使用腿的名称而不是索引
                enabled=True,
                priority=ConstraintPriority.MEDIUM,
                value=value,
                threshold=threshold,
                violation=max(0, threshold - value),
                is_violated=value < 0
            ))

        return statuses

class TrackingConstraintModule:
    """Module combining all tracking-related constraints"""
    def __init__(self, config: Dict):
        self.base_position_tracking = BasePositionTrackingConstraint(
            config['position_tracking'])
        self.base_orientation_tracking = BaseOrientationTrackingConstraint(
            config['orientation_tracking'])
        self.foot_position_tracking = FootPositionTrackingConstraint(
            config['foot_tracking'])
        
    def check_all(self, state_dict: Dict) -> Dict[str, Union[ConstraintStatus, List[ConstraintStatus]]]:
        """Check all tracking constraints"""
        return {
            'base_position': self.base_position_tracking.check(state_dict),
            'base_orientation': self.base_orientation_tracking.check(state_dict),
            'foot_position': self.foot_position_tracking.check(state_dict)
        }

    def get_most_critical(self, state_dict: Dict) -> ConstraintStatus:
        """Get the most critical constraint status"""
        all_statuses = self.check_all(state_dict)
        
        # Flatten all constraints
        all_statuses_flat = [
            all_statuses['base_position'],
            all_statuses['base_orientation']
        ] + all_statuses['foot_position']
        
        # Find most critical
        return min(all_statuses_flat, key=lambda x: x.value)