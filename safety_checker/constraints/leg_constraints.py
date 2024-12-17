
import numpy as np
from typing import Dict, List, Optional, Union
from ..core.barrier_functions import BarrierFunction, BarrierConfig
from ..core.safety_level import ConstraintStatus, ConstraintPriority

class LegConstraint:
    """Base class for all leg-related constraints"""
    def __init__(self, config: Dict):
        self.config = config
        self.enabled = config.get('enabled', True)
        self.name = "leg_constraint"
        self._init_barrier_functions()

    def _init_barrier_functions(self):
        """Initialize barrier functions"""
        raise NotImplementedError

    def check(self, state_dict: Dict) -> ConstraintStatus:
        """Check constraint satisfaction"""
        raise NotImplementedError

class JointVelocityConstraint(LegConstraint):
    """Constraint on joint velocities"""
    def __init__(self, config: Dict):
        super().__init__(config)
        self.name = "joint_velocity"
        self.num_legs = 4
        self.joints_per_leg = 3  # hip, thigh, calf

    def _init_barrier_functions(self):
        self.barrier_config = BarrierConfig(
            threshold=self.config['thresholds']['vel_max'],
            warning_threshold=self.config['thresholds']['warning'],
            danger_threshold=self.config['thresholds']['danger'],
            emergency_threshold=self.config['thresholds']['emergency'],
            weight=np.mean(self.config['weights'])  # Average of joint weights
        )
        self.barrier = BarrierFunction(self.barrier_config)

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

        joint_velocities = state_dict['qvel_js']
        # Reshape to (num_legs, joints_per_leg)
        joint_velocities = joint_velocities.reshape(self.num_legs, self.joints_per_leg)
        
        # Check maximum velocity across all joints
        max_velocity = np.max(np.abs(joint_velocities))
        value = self.barrier.evaluate(max_velocity)
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

class WorkspaceConstraint(LegConstraint):
    """Constraint on foot positions relative to base"""
    def __init__(self, config: Dict):
        super().__init__(config)
        self.name = "workspace"
        self.num_legs = 4

    def _init_barrier_functions(self):
        self.barrier_config = BarrierConfig(
            threshold=self.config['thresholds']['workspace_max'],
            warning_threshold=self.config['thresholds']['warning'],
            danger_threshold=self.config['thresholds']['danger'],
            emergency_threshold=self.config['thresholds']['emergency'],
            weight=np.mean(self.config['weights'])
        )
        self.barrier = BarrierFunction(self.barrier_config)

    def check(self, state_dict: Dict) -> ConstraintStatus:
        """Check constraint satisfaction"""
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

        feet_positions = state_dict['feet_pos_base']  
        
        # 添加数据维度检查和处理
        feet_positions = np.array(feet_positions)
        if feet_positions.ndim == 1:
            # 如果是一维数组，假设它是展平的4*3数组，重塑它
            feet_positions = feet_positions.reshape(4, 3)
        elif feet_positions.ndim > 2:
            raise ValueError(f"Unexpected feet_positions dimension: {feet_positions.shape}")
        
        # 计算每个足端到原点的距离
        distances = np.linalg.norm(feet_positions, axis=1)
        max_distance = np.max(distances)
        
        value = self.barrier.evaluate(max_distance)
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

class LegPhaseConstraint(LegConstraint):
    """Constraint on leg motion during different phases"""
    def __init__(self, config: Dict):
        super().__init__(config)
        self.name = "leg_phase"
        self.num_legs = 4

    def _init_barrier_functions(self):
        self.swing_barrier_config = BarrierConfig(
            threshold=self.config['thresholds']['swing_height_max'],
            warning_threshold=self.config['thresholds']['warning'],
            danger_threshold=self.config['thresholds']['danger'],
            emergency_threshold=self.config['thresholds']['emergency'],
            weight=self.config['weights'][0]
        )
        self.swing_barrier = BarrierFunction(self.swing_barrier_config)
        
        self.stance_barrier_config = BarrierConfig(
            threshold=self.config['thresholds']['stance_vel_max'],
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
                priority=ConstraintPriority.HIGH,
                value=0.0,
                threshold=float('inf'),
                violation=0.0,
                is_violated=False
            ) for i in range(self.num_legs)]

        # 获取并重塑数据
        feet_positions = np.array(state_dict['feet_pos_base'])
        feet_velocities = np.array(state_dict['feet_vel_base'])
        contact_state = np.array(state_dict['contact_state'])

        # 重塑为正确的维度 (4, 3)
        feet_positions = feet_positions.reshape(self.num_legs, 3)
        feet_velocities = feet_velocities.reshape(self.num_legs, 3)
        
        statuses = []
        for i in range(self.num_legs):
            if contact_state[i]:  # 支撑相
                vel_norm = np.linalg.norm(feet_velocities[i])
                value = self.stance_barrier.evaluate(vel_norm)
                threshold = self.stance_barrier_config.threshold
            else:  # 摆动相
                height = feet_positions[i, 2]  # z-coordinate
                value = self.swing_barrier.evaluate(height)
                threshold = self.swing_barrier_config.threshold

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

class LegConstraintModule:
    """Module combining all leg-related constraints"""
    def __init__(self, config: Dict):
        self.joint_velocity_constraint = JointVelocityConstraint(config['joint_velocity'])
        self.workspace_constraint = WorkspaceConstraint(config['workspace'])
        self.phase_constraint = LegPhaseConstraint(config['phase'])
        
    def check_all(self, state_dict: Dict) -> Dict[str, Union[ConstraintStatus, List[ConstraintStatus]]]:
        """Check all leg constraints"""
        return {
            'joint_velocity': self.joint_velocity_constraint.check(state_dict),
            'workspace': self.workspace_constraint.check(state_dict),
            'phase': self.phase_constraint.check(state_dict)
        }

    def get_most_critical(self, state_dict: Dict) -> ConstraintStatus:
        """Get the most critical constraint status"""
        all_statuses = self.check_all(state_dict)
        
        # Flatten phase constraints
        all_statuses_flat = [
            all_statuses['joint_velocity'],
            all_statuses['workspace']
        ] + all_statuses['phase']
        
        # Find most critical (minimum value)
        return min(all_statuses_flat, key=lambda x: x.value)