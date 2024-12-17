import numpy as np
from typing import List, Union, Callable

def smooth_min(values: Union[List[float], np.ndarray], 
               rho: float = 1.0) -> float:
    """
    Compute smooth minimum using log-sum-exp
    
    Args:
        values: List of values to compute minimum over
        rho: Smoothing parameter (larger means closer to true minimum)
    
    Returns:
        Smooth minimum value
    """
    values = np.array(values)
    return -1.0/rho * np.log(np.sum(np.exp(-rho * values)))

def smooth_max(values: Union[List[float], np.ndarray], 
               rho: float = 1.0) -> float:
    """
    Compute smooth maximum using log-sum-exp
    
    Args:
        values: List of values to compute maximum over
        rho: Smoothing parameter (larger means closer to true maximum)
    
    Returns:
        Smooth maximum value
    """
    values = np.array(values)
    return 1.0/rho * np.log(np.sum(np.exp(rho * values)))

def barrier_function(x: float, 
                    threshold: float, 
                    k: float = 1.0) -> float:
    """
    Compute barrier function value
    
    Args:
        x: Input value
        threshold: Barrier threshold
        k: Barrier steepness parameter
    
    Returns:
        Barrier function value
    """
    return np.log(x / threshold) / k

def sigmoid_activation(x: float, 
                      center: float = 0.0, 
                      steepness: float = 1.0) -> float:
    """
    Compute sigmoid activation
    
    Args:
        x: Input value
        center: Sigmoid center point
        steepness: Sigmoid steepness
    
    Returns:
        Sigmoid activation value
    """
    return 1.0 / (1.0 + np.exp(-steepness * (x - center)))

def weighted_sum(values: Union[List[float], np.ndarray],
                weights: Union[List[float], np.ndarray]) -> float:
    """
    Compute weighted sum of values
    
    Args:
        values: List of values
        weights: List of weights
    
    Returns:
        Weighted sum
    """
    return np.sum(np.array(values) * np.array(weights))

def create_barrier_function(threshold: float, 
                          k: float = 1.0) -> Callable[[float], float]:
    """
    Create a barrier function with fixed parameters
    
    Args:
        threshold: Barrier threshold
        k: Barrier steepness parameter
    
    Returns:
        Barrier function
    """
    return lambda x: barrier_function(x, threshold, k)