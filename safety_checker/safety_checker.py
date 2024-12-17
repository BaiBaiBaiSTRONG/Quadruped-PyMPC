import numpy as np
from typing import Dict, List, Union, Optional, Tuple, NamedTuple
import yaml
import os
from pathlib import Path

from .core.safety_level import SafetyLevel, SafetyStatus, ConstraintStatus, ConstraintPriority
from .core.barrier_functions import BarrierFunction, BarrierConfig
from .constraints.base_constraints import BaseConstraintModule
from .constraints.leg_constraints import LegConstraintModule
from .constraints.grf_constraints import GRFConstraintModule
from .constraints.tracking_constraints import TrackingConstraintModule
from .constraints.prediction_constraints import PredictionConstraintModule
from .scheduler.constraint_scheduler import ConstraintScheduler
from .scheduler.activation_optimizer import ActivationOptimizer

class SafetyOutput(NamedTuple):
    """Safety checker output signals"""
    is_safe: bool                    # 整体是否安全
    safety_level: int                # 安全等级 (3:安全, 2:警告, 1:危险, 0:紧急)
    stop_requested: bool             # 是否需要停止
    slow_down_ratio: float          # 建议的减速比例 (0.0-1.0)
    violated_constraints: list       # 违反的约束列表
    critical_values: Dict[str, float] # 关键值字典

class SafetyChecker:
    """Main class for quadruped robot safety checking"""
    def __init__(self,
                 state_observable_names: List[str],
                 mpc_observable_names: List[str],
                 config_dict: Optional[Dict] = None,
                 config_path: Optional[str] = "safety_checker/config/safety_config.yaml"):
        """
        Initialize safety checker
        Args:
            state_observable_names: List of state observable names
            mpc_observable_names: List of MPC observable names
            config_dict: Configuration dictionary (optional)
            config_path: Path to configuration file (optional)
        """
        self.state_names = state_observable_names
        self.mpc_names = mpc_observable_names
        
        # Load configuration
        self.config = self._load_config(config_dict, config_path)
        
        # Initialize constraint modules
        self._init_constraint_modules()
        
        # Initialize scheduler
        self._init_scheduler()
        
        # Initialize history
        self.safety_history = []
        self.violation_history = {}

    def __call__(self, state_dict: Dict) -> SafetyOutput:
        """
        Simple safety check call interface
        Args:
            state_dict: Dictionary containing all necessary states
        Returns:
            SafetyOutput: Output containing key safety signals
        """
        # Get detailed safety status
        safety_status = self.check_safety(state_dict, return_details=True)
        
        # Calculate safety flags
        is_safe = safety_status.level >= SafetyLevel.WARNING
        safety_level = safety_status.level.value
        stop_requested = safety_status.level <= SafetyLevel.DANGER
        
        # Calculate slow down ratio
        slow_down_ratio = self._calculate_slow_down_ratio(safety_status)
        
        # Get violated constraints
        violated_constraints = list(safety_status.violations.keys())
        
        # Collect critical values
        critical_values = self._get_critical_values(state_dict, safety_status)
        
        return SafetyOutput(
            is_safe=is_safe,
            safety_level=safety_level,
            stop_requested=stop_requested,
            slow_down_ratio=slow_down_ratio,
            violated_constraints=violated_constraints,
            critical_values=critical_values
        )

    def _calculate_slow_down_ratio(self, safety_status: SafetyStatus) -> float:
        """Calculate recommended slow down ratio"""
        if safety_status.level == SafetyLevel.SAFE:
            return 1.0
        elif safety_status.level == SafetyLevel.WARNING:
            return 0.7
        elif safety_status.level == SafetyLevel.DANGER:
            return 0.3
        else:  # EMERGENCY
            return 0.0

    def _get_critical_values(self, 
                           state_dict: Dict,
                           safety_status: SafetyStatus) -> Dict[str, float]:
        """Collect critical safety values"""
        
        # 确保足端位置数据格式正确
        feet_pos = np.array(state_dict['feet_pos_base'])
        if len(feet_pos.shape) == 1:
            # 如果是一维数组，假设它是按[x1,y1,z1,x2,y2,z2,...]排列
            feet_pos = feet_pos.reshape(-1, 3)
        
        # 确保关节速度数据格式正确
        joint_vel = np.array(state_dict['qvel_js'])
        if len(joint_vel.shape) == 1:
            joint_vel = joint_vel.reshape(-1)
        
        # 确保接触力数据格式正确
        contact_forces = np.array(state_dict['contact_forces_base'])
        if len(contact_forces.shape) == 1:
            contact_forces = contact_forces.reshape(-1, 3)
        
        # 确保欧拉角数据格式正确
        base_ori = np.array(state_dict['base_ori_euler_xyz'])
        if len(base_ori.shape) == 1:
            base_ori = base_ori.reshape(-1)
        
        return {
            'max_base_tilt': np.linalg.norm(base_ori[:2]),
            'max_joint_vel': np.max(np.abs(joint_vel)),
            'min_foot_clearance': np.min(feet_pos[:, 2]),
            'max_violation': max(safety_status.violations.values()) if safety_status.violations else 0.0,
            'base_height': state_dict['base_pos'][2],
            'max_contact_force': np.max(np.linalg.norm(contact_forces, axis=1))
        }

    def _load_config(self, config_dict: Optional[Dict], config_path: Optional[str]) -> Dict:
        """Load configuration from dictionary or file"""
        if config_dict is not None:
            return config_dict
        elif config_path is not None:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Load default configuration
            default_path = Path(__file__).parent / 'config' / 'default_safety_config.yaml'
            with open(default_path, 'r') as f:
                return yaml.safe_load(f)

    def _init_constraint_modules(self):
        """Initialize all constraint modules"""
        self.base_module = BaseConstraintModule(self.config['base_constraints'])
        self.leg_module = LegConstraintModule(self.config['leg_constraints'])
        self.grf_module = GRFConstraintModule(self.config['grf_constraints'])
        self.tracking_module = TrackingConstraintModule(self.config['tracking_constraints'])
        self.prediction_module = PredictionConstraintModule(self.config['prediction_constraints'])

    def _init_scheduler(self):
        """Initialize constraint scheduler"""
        self.scheduler = ConstraintScheduler(self.config['scheduler_config'])
        self.scheduler.initialize_weights(self.config)
        
        if self.config['scheduler_config'].get('dynamic_activation', True):
            self.activation_optimizer = ActivationOptimizer(
                self.config['scheduler_config']
            )
        else:
            self.activation_optimizer = None

    def check_safety(self, 
                    state_dict: Dict,
                    return_details: bool = False
                    ) -> Union[SafetyLevel, SafetyStatus]:
        """
        Check safety of current state
        Args:
            state_dict: Dictionary containing current state
            return_details: Whether to return detailed status
        Returns:
            SafetyLevel or SafetyStatus
        """
        # Validate input state dictionary
        self._validate_state_dict(state_dict)
        
        # Check all constraints
        constraint_statuses = self._check_all_constraints(state_dict)
        
        # Update scheduler
        self._update_scheduler(state_dict, constraint_statuses)
        
        # Determine overall safety level
        safety_level, violations = self._determine_safety_level(constraint_statuses)
        
        # Update history
        self._update_history(safety_level, violations)
        
        if return_details:
            return SafetyStatus(
                level=safety_level,
                violations=violations,
                active_constraints=self._get_active_constraints(),
                recommended_action=self._get_recommended_action(safety_level, violations),
                details={'constraint_statuses': constraint_statuses}
            )
        else:
            return safety_level

    def _validate_state_dict(self, state_dict: Dict):
        """Validate input state dictionary"""
        required_keys = set(self.state_names)
        provided_keys = set(state_dict.keys())
        missing_keys = required_keys - provided_keys
        
        if missing_keys:
            raise ValueError(f"Provided state keys: {provided_keys} do not match required keys: {required_keys}, missing keys: {missing_keys}")

    def _check_all_constraints(self, state_dict: Dict) -> Dict[str, Dict]:
        """Check all constraint modules"""
        return {
            'base': self.base_module.check_all(state_dict),
            'leg': self.leg_module.check_all(state_dict),
            'grf': self.grf_module.check_all(state_dict),
            'tracking': self.tracking_module.check_all(state_dict),
            'prediction': self.prediction_module.check_all(state_dict)
        }

    def _update_scheduler(self, 
                         state_dict: Dict,
                         constraint_statuses: Dict[str, Dict]):
        """Update constraint scheduler"""
        # Update weights
        self.scheduler.update_weights(state_dict, constraint_statuses)
        
        # Update activations if enabled
        if self.activation_optimizer is not None:
            self.activation_optimizer.update_history(
                constraint_statuses,
                self.scheduler.current_activations
            )
            new_activations = self.activation_optimizer.optimize_activations(
                self.scheduler.current_activations,
                {name: self.scheduler.get_constraint_priority(name)
                 for name in self.scheduler.current_activations}
            )
            self.scheduler.current_activations = new_activations

    def _determine_safety_level(self,
                              constraint_statuses: Dict[str, Dict]
                              ) -> Tuple[SafetyLevel, Dict[str, float]]:
        """Determine overall safety level and collect violations"""
        violations = {}
        critical_violated = False
        high_violated = False
        medium_violated = False
        
        # Check all constraints
        for module_name, module_statuses in constraint_statuses.items():
            for constraint_name, status in module_statuses.items():
                if isinstance(status, list):
                    statuses = status
                else:
                    statuses = [status]
                
                for status in statuses:
                    if status.is_violated:
                        violations[f"{module_name}.{constraint_name}"] = status.violation
                        if status.priority == ConstraintPriority.CRITICAL:
                            critical_violated = True
                        elif status.priority == ConstraintPriority.HIGH:
                            high_violated = True
                        elif status.priority == ConstraintPriority.MEDIUM:
                            medium_violated = True
        
        # Determine safety level
        if critical_violated:
            return SafetyLevel.EMERGENCY, violations
        elif high_violated:
            return SafetyLevel.DANGER, violations
        elif medium_violated:
            return SafetyLevel.WARNING, violations
        else:
            return SafetyLevel.SAFE, violations

    def _get_active_constraints(self) -> List[str]:
        """Get list of currently active constraints"""
        return [name for name, active in self.scheduler.current_activations.items()
                if active]

    def _get_recommended_action(self, 
                              safety_level: SafetyLevel,
                              violations: Dict[str, float]) -> str:
        """Get recommended action based on safety level and violations"""
        if safety_level == SafetyLevel.EMERGENCY:
            return "Emergency stop required. Critical constraints violated."
        elif safety_level == SafetyLevel.DANGER:
            return "Reduce speed and prepare for safe stop. High priority constraints violated."
        elif safety_level == SafetyLevel.WARNING:
            return "Continue with caution. Monitor violated constraints."
        else:
            return "Continue normal operation."

    def _update_history(self, 
                       safety_level: SafetyLevel,
                       violations: Dict[str, float]):
        """Update safety history"""
        self.safety_history.append(safety_level)
        for constraint, violation in violations.items():
            if constraint not in self.violation_history:
                self.violation_history[constraint] = []
            self.violation_history[constraint].append(violation)
        
        # Keep history length manageable
        max_history = 1000
        if len(self.safety_history) > max_history:
            self.safety_history = self.safety_history[-max_history:]
            for constraint in self.violation_history:
                self.violation_history[constraint] = (
                    self.violation_history[constraint][-max_history:]
                )

    def get_statistics(self) -> Dict:
        """Get safety statistics"""
        if not self.safety_history:
            return {}
            
        return {
            'safety_levels': {
                level.name: self.safety_history.count(level) / len(self.safety_history)
                for level in SafetyLevel
            },
            'violation_frequencies': {
                constraint: np.mean(violations) if violations else 0.0
                for constraint, violations in self.violation_history.items()
            },
            'scheduler_stats': self.scheduler.current_weights,
            'activation_stats': (self.activation_optimizer.get_optimization_stats()
                               if self.activation_optimizer else None)
        }

    def reset(self):
        """Reset safety checker state"""
        self.safety_history.clear()
        self.violation_history.clear()
        self._init_scheduler()