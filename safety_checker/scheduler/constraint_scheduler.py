import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from ..core.safety_level import ConstraintStatus, ConstraintPriority

@dataclass
class SchedulerConfig:
    """Configuration for constraint scheduler"""
    phase_based_weights: bool = True
    dynamic_activation: bool = False
    update_frequency: int = 100  # Hz
    optimization_method: str = "gradient_based"
    weight_bounds: tuple = (0.1, 2.0)
    adaptation_rate: float = 0.1

class ConstraintScheduler:
    """Manages constraint weights and activation"""
    def __init__(self, config: Dict):
        self.config = SchedulerConfig(**config)
        self.current_weights = {}
        self.current_activations = {}
        self.history = []
        self.update_counter = 0
        
    def initialize_weights(self, constraint_configs: Dict[str, Dict]):
        """Initialize weights from config"""
        self.base_weights = {}
        for constraint_type, config in constraint_configs.items():
            if 'weights' in config:
                self.base_weights[constraint_type] = np.array(config['weights'])
                self.current_weights[constraint_type] = np.array(config['weights'])
            if 'enabled' in config:
                self.current_activations[constraint_type] = config['enabled']

    def update_weights(self, 
                      state_dict: Dict,
                      constraint_statuses: Dict[str, List[ConstraintStatus]]
                      ) -> Dict[str, np.ndarray]:
        """Update constraint weights based on current state and phase"""
        self.update_counter += 1
        
        # Only update at specified frequency
        if self.update_counter % (self.config.update_frequency // 10) != 0:
            return self.current_weights

        # Phase-based weight adjustment
        if self.config.phase_based_weights and 'phase_signal' in state_dict:
            self._adjust_weights_by_phase(state_dict['phase_signal'])

        # Performance-based weight adjustment
        self._adjust_weights_by_performance(constraint_statuses)

        # Clip weights to bounds
        for constraint_type in self.current_weights:
            self.current_weights[constraint_type] = np.clip(
                self.current_weights[constraint_type],
                self.config.weight_bounds[0],
                self.config.weight_bounds[1]
            )

        return self.current_weights

    def _adjust_weights_by_phase(self, phase_signal: np.ndarray):
        """Adjust weights based on gait phase"""
        for constraint_type in self.current_weights:
            if constraint_type.startswith('leg_') or constraint_type.startswith('grf_'):
                # Increase weights for stance legs
                stance_mask = (phase_signal < 0.5)
                swing_mask = ~stance_mask
                
                base_weight = self.base_weights[constraint_type]
                new_weights = base_weight.copy()
                
                # Modify weights based on phase
                new_weights[stance_mask] *= 1.5  # Increase stance weights
                new_weights[swing_mask] *= 0.8   # Decrease swing weights
                
                # Smooth weight transition
                alpha = self.config.adaptation_rate
                self.current_weights[constraint_type] = (
                    (1 - alpha) * self.current_weights[constraint_type] +
                    alpha * new_weights
                )

    def _adjust_weights_by_performance(self, constraint_statuses: Dict[str, List[ConstraintStatus]]):
        """Adjust weights based on constraint violations"""
        for constraint_type, statuses in constraint_statuses.items():
            if constraint_type not in self.current_weights:
                continue
                
            # Calculate violation severity
            violations = [status.violation for status in statuses]
            max_violation = max(violations) if violations else 0
            
            # Adjust weights based on violations
            if max_violation > 0:
                self.current_weights[constraint_type] *= (1 + 
                    self.config.adaptation_rate * max_violation)

    def update_activations(self, 
                          state_dict: Dict,
                          constraint_statuses: Dict[str, List[ConstraintStatus]]
                          ) -> Dict[str, bool]:
        """Update constraint activations"""
        if not self.config.dynamic_activation:
            return self.current_activations

        # This is a placeholder for future activation optimization
        # Currently just maintains existing activations
        return self.current_activations

    def get_constraint_priority(self, constraint_type: str) -> ConstraintPriority:
        """Get priority level for a constraint type"""
        priority_map = {
            'base_stability': ConstraintPriority.CRITICAL,
            'joint_limits': ConstraintPriority.CRITICAL,
            'grf_limits': ConstraintPriority.HIGH,
            'tracking': ConstraintPriority.MEDIUM
        }
        
        for key, priority in priority_map.items():
            if key in constraint_type.lower():
                return priority
                
        return ConstraintPriority.MEDIUM

    def log_state(self, 
                  state_dict: Dict,
                  constraint_statuses: Dict[str, List[ConstraintStatus]]):
        """Log current state for analysis"""
        self.history.append({
            'weights': self.current_weights.copy(),
            'activations': self.current_activations.copy(),
            'violations': {
                ctype: [status.violation for status in statuses]
                for ctype, statuses in constraint_statuses.items()
            }
        })
        
        # Keep history length manageable
        if len(self.history) > 1000:
            self.history = self.history[-1000:]