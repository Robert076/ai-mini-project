"""
Griewank function implementation for 2D input.
Domain: x, y in [-600, 600]
"""
import numpy as np

def griewank(x, y):
    """
    Compute the Griewank function for 2D input.
    Args:
        x (float or np.ndarray): x-coordinate(s)
        y (float or np.ndarray): y-coordinate(s)
    Returns:
        float or np.ndarray: Griewank function value(s)
    """
    return 1 + (x**2 + y**2) / 4000 - np.cos(x) * np.cos(y / np.sqrt(2))
