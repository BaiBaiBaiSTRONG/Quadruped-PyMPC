import numpy as np
from typing import Dict, Optional
from ..core.barrier_functions import BarrierFunction, BarrierConfig
from ..core.safety_level import ConstraintStatus, ConstraintPriority

class BaseConstraint:
    """Base class for all base-related constraints"""
    def __init__(self, config: Dict):
        self.config = config
        self.enabled = config.get('enabled', True)
        self.name = "base_constraint"
        self._init_barrier_functions()

    def _init_barrier_functions(self):
        """Initialize barrier functions"""
        raise NotImplementedError

    def check(self, state_dict: Dict) -> ConstraintStatus:
        """Check constraint satisfaction"""
        raise NotImplementedError

class BasePositionConstraint(BaseConstraint):
    """Constraint on base position"""
    def __init__(self, config: Dict):
        super().__init__(config)
        self.name = "base_position"

    def _init_barrier_functions(self):
        self.barrier_config = BarrierConfig(
            threshold=self.config['thresholds']['pos_max_xy'],
            warning_threshold=self.config['thresholds']['warning'],
            danger_threshold=self.config['thresholds']['danger'],
            emergency_threshold=self.config['thresholds']['emergency'],
            weight=self.config['weights'][0]
        )
        self.xy_barrier = BarrierFunction(self.barrier_config)
        
        # Separate barrier for z-axis
        z_barrier_config = BarrierConfig(
            threshold=self.config['thresholds']['pos_max_z'],
            warning_threshold=self.config['thresholds']['warning'],
            danger_threshold=self.config['thresholds']['danger'],
            emergency_threshold=self.config['thresholds']['emergency'],
            weight=self.config['weights'][2]
        )
        self.z_barrier = BarrierFunction(z_barrier_config)

    def check(self, state_dict: Dict) -> ConstraintStatus:
        if not self.enabled:
            return ConstraintStatus(
                name=self.name,
                enabled=False,
                priority=ConstraintPriority.CRITICAL,
                value=0.0,
                threshold=float('inf'),
                violation=0.0,
                is_violated=False
            )

        base_pos = state_dict['base_pos']
        xy_norm = np.linalg.norm(base_pos[:2])
        z_pos = abs(base_pos[2])

        xy_value = self.xy_barrier.evaluate(xy_norm)
        z_value = self.z_barrier.evaluate(z_pos)
        
        

        # Here we provide the xy_value and z_value to the threshold. But in real only add the z_value to the threshold.
        # value = min(xy_value, z_value)

        value = z_value
        threshold = self.barrier_config.threshold
        
        return ConstraintStatus(
            name=self.name,
            enabled=True,
            priority=ConstraintPriority.CRITICAL,
            value=value,
            threshold=threshold,
            violation=max(0, threshold - value),
            is_violated=value < 0
        )

class BaseVelocityConstraint(BaseConstraint):
    """Constraint on base velocity"""
    def __init__(self, config: Dict):
        super().__init__(config)
        self.name = "base_velocity"

    def _init_barrier_functions(self):
        self.barrier_config = BarrierConfig(
            threshold=self.config['thresholds']['vel_max_xy'],
            warning_threshold=self.config['thresholds']['warning'],
            danger_threshold=self.config['thresholds']['danger'],
            emergency_threshold=self.config['thresholds']['emergency'],
            weight=self.config['weights'][0]
        )
        self.xy_barrier = BarrierFunction(self.barrier_config)
        
        z_barrier_config = BarrierConfig(
            threshold=self.config['thresholds']['vel_max_z'],
            warning_threshold=self.config['thresholds']['warning'],
            danger_threshold=self.config['thresholds']['danger'],
            emergency_threshold=self.config['thresholds']['emergency'],
            weight=self.config['weights'][2]
        )
        self.z_barrier = BarrierFunction(z_barrier_config)

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

        base_vel = state_dict['base_lin_vel']
        xy_vel_norm = np.linalg.norm(base_vel[:2])
        z_vel = abs(base_vel[2])

        xy_value = self.xy_barrier.evaluate(xy_vel_norm)
        z_value = self.z_barrier.evaluate(z_vel)
        
        value = min(xy_value, z_value)
        threshold = self.barrier_config.threshold

        return ConstraintStatus(
            name=self.name,
            enabled=True,
            priority=ConstraintPriority.HIGH,
            value=value,
            threshold=threshold,
            violation=max(0, threshold - value),
            is_violated=value < 0
        )

class BaseOrientationConstraint(BaseConstraint):
    """Constraint on base orientation"""
    def __init__(self, config: Dict):
        super().__init__(config)
        self.name = "base_orientation"

    def _init_barrier_functions(self):
        self.barrier_config = BarrierConfig(
            threshold=self.config['thresholds']['angle_max'],
            warning_threshold=self.config['thresholds']['warning'],
            danger_threshold=self.config['thresholds']['danger'],
            emergency_threshold=self.config['thresholds']['emergency'],
            weight=self.config['weights'][0]
        )
        self.barrier = BarrierFunction(self.barrier_config)

    def check(self, state_dict: Dict) -> ConstraintStatus:
        if not self.enabled:
            return ConstraintStatus(
                name=self.name,
                enabled=False,
                priority=ConstraintPriority.CRITICAL,
                value=0.0,
                threshold=float('inf'),
                violation=0.0,
                is_violated=False
            )

        euler_angles = state_dict['base_ori_euler_xyz']
        roll, pitch = euler_angles[0], euler_angles[1]
        
        # Combine roll and pitch
        orientation_norm = np.sqrt(roll**2 + pitch**2)
        value = self.barrier.evaluate(orientation_norm)
        threshold = self.barrier_config.threshold

        return ConstraintStatus(
            name=self.name,
            enabled=True,
            priority=ConstraintPriority.CRITICAL,
            value=value,
            threshold=threshold,
            violation=max(0, threshold - value),
            is_violated=value < 0
        )

class BaseConstraintModule:
    """Module combining all base-related constraints"""
    def __init__(self, config: Dict):
        self.position_constraint = BasePositionConstraint(config['position'])
        self.velocity_constraint = BaseVelocityConstraint(config['velocity'])
        self.orientation_constraint = BaseOrientationConstraint(config['orientation'])
        
    def check_all(self, state_dict: Dict) -> Dict[str, ConstraintStatus]:
        """Check all base constraints"""
        return {
            'position': self.position_constraint.check(state_dict),
            'velocity': self.velocity_constraint.check(state_dict),
            'orientation': self.orientation_constraint.check(state_dict)
        }