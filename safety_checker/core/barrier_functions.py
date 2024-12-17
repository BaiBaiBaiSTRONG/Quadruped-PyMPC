import numpy as np
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass

@dataclass
class BarrierConfig:
    """Configuration for a barrier function"""
    threshold: float
    warning_threshold: float
    danger_threshold: float
    emergency_threshold: float
    weight: float = 1.0
    enabled: bool = True

class BarrierFunction:
    """Base class for barrier functions"""
    def __init__(self, config: BarrierConfig):
        self.config = config
        
    def evaluate(self, x: float) -> float:
        """Evaluate barrier function"""
        if not self.config.enabled:
            return float('inf')
        return self.config.weight * (self.config.threshold**2 - x**2)
        
    def get_safety_margin(self, x: float) -> float:
        """Get safety margin"""
        return self.evaluate(x)

class QuadraticBarrier(BarrierFunction):
    """Quadratic barrier function"""
    def evaluate(self, x: float) -> float:
        if not self.config.enabled:
            return float('inf')
        return self.config.weight * (self.config.threshold**2 - x**2)

class LogarithmicBarrier(BarrierFunction):
    """Logarithmic barrier function"""
    def evaluate(self, x: float) -> float:
        if not self.config.enabled:
            return float('inf')
        # 添加小的偏移量避免对数为0
        epsilon = 1e-6
        return self.config.weight * np.log((self.config.threshold - x) + epsilon)

class ExponentialBarrier(BarrierFunction):
    """Exponential barrier function"""
    def evaluate(self, x: float) -> float:
        if not self.config.enabled:
            return float('inf')
        return self.config.weight * np.exp(-(x / self.config.threshold))

class MultiDimensionalBarrier:
    """Barrier function for multi-dimensional constraints"""
    def __init__(self, 
                 config: BarrierConfig,
                 dimension: int):
        self.barriers = [QuadraticBarrier(config) for _ in range(dimension)]
        self.config = config
        
    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate barrier function for vector input"""
        if not self.config.enabled:
            return float('inf')
        return min(barrier.evaluate(xi) for barrier, xi in zip(self.barriers, x))

class CompositeBarrier:
    """Combination of multiple barrier functions"""
    def __init__(self, 
                 barriers: List[BarrierFunction],
                 smooth_param: float = 1.0):
        self.barriers = barriers
        self.smooth_param = smooth_param
        
    def evaluate(self, x: float) -> float:
        """Evaluate composite barrier function"""
        values = [barrier.evaluate(x) for barrier in self.barriers]
        return -1.0/self.smooth_param * np.log(
            np.sum(np.exp(-self.smooth_param * np.array(values))))

class BarrierFunctionFactory:
    """Factory class for creating barrier functions"""
    @staticmethod
    def create_barrier(barrier_type: str, 
                      config: BarrierConfig) -> BarrierFunction:
        """Create barrier function of specified type"""
        barriers = {
            'quadratic': QuadraticBarrier,
            'logarithmic': LogarithmicBarrier,
            'exponential': ExponentialBarrier
        }
        
        if barrier_type not in barriers:
            raise ValueError(f"Unknown barrier type: {barrier_type}")
            
        return barriers[barrier_type](config)