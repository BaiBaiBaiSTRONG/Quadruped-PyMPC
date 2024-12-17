import numpy as np
from typing import Dict, List, Optional
from ..core.safety_level import ConstraintStatus, ConstraintPriority

class ActivationOptimizer:
    """Optimizes constraint activation patterns"""
    def __init__(self, config: Dict):
        self.config = config
        self.method = config.get('optimization_method', 'gradient_based')
        self.learning_rate = config.get('learning_rate', 0.01)
        self.window_size = config.get('window_size', 100)
        
        self.constraint_history = {}
        self.activation_history = {}
        self.performance_metrics = {}

    def initialize_constraints(self, constraint_types: List[str]):
        """Initialize constraint tracking"""
        for ctype in constraint_types:
            self.constraint_history[ctype] = []
            self.activation_history[ctype] = []
            self.performance_metrics[ctype] = {
                'violation_frequency': 0.0,
                'average_violation': 0.0,
                'importance_score': 0.0
            }

    def update_history(self, 
                      constraint_statuses: Dict[str, List[ConstraintStatus]],
                      current_activations: Dict[str, bool]):
        """Update constraint violation history"""
        for ctype, statuses in constraint_statuses.items():
            # Record violations
            violations = [status.violation for status in statuses]
            self.constraint_history[ctype].append(max(violations))
            
            # Record activations
            self.activation_history[ctype].append(
                current_activations.get(ctype, True)
            )
            
            # Keep history length manageable
            if len(self.constraint_history[ctype]) > self.window_size:
                self.constraint_history[ctype] = (
                    self.constraint_history[ctype][-self.window_size:]
                )
                self.activation_history[ctype] = (
                    self.activation_history[ctype][-self.window_size:]
                )

    def update_metrics(self):
        """Update performance metrics for each constraint"""
        for ctype in self.constraint_history:
            violations = self.constraint_history[ctype]
            if not violations:
                continue
                
            # Calculate metrics
            self.performance_metrics[ctype].update({
                'violation_frequency': np.mean(np.array(violations) > 0),
                'average_violation': np.mean(violations),
                'importance_score': self._calculate_importance(ctype)
            })

    def _calculate_importance(self, constraint_type: str) -> float:
        """Calculate importance score for a constraint"""
        # This is a placeholder for more sophisticated importance calculation
        violations = self.constraint_history[constraint_type]
        if not violations:
            return 0.0
            
        recent_violations = violations[-10:]
        violation_trend = np.mean(recent_violations) / (np.mean(violations) + 1e-6)
        
        return violation_trend

    def optimize_activations(self, 
                           current_activations: Dict[str, bool],
                           constraint_priorities: Dict[str, ConstraintPriority]
                           ) -> Dict[str, bool]:
        """Optimize constraint activation pattern"""
        if not self.constraint_history:
            return current_activations

        self.update_metrics()
        new_activations = current_activations.copy()

        for ctype in current_activations:
            if constraint_priorities[ctype] == ConstraintPriority.CRITICAL:
                # Critical constraints are always active
                new_activations[ctype] = True
                continue

            metrics = self.performance_metrics[ctype]
            
            # Decision logic for activation
            if metrics['violation_frequency'] > 0.8:
                # High violation frequency -> keep/make active
                new_activations[ctype] = True
            elif metrics['average_violation'] < 0.1 and metrics['importance_score'] < 0.5:
                # Low violations and importance -> consider deactivating
                new_activations[ctype] = False

        return new_activations

    def get_optimization_stats(self) -> Dict:
        """Get optimization statistics"""
        return {
            'metrics': self.performance_metrics,
            'activation_changes': {
                ctype: np.sum(np.diff(self.activation_history[ctype]))
                for ctype in self.activation_history
            }
        }