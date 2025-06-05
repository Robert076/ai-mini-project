"""
Ackley function implementation for 2D input.
Domain: x, y in [-32.768, 32.768]
"""
import numpy as np

def ackley(x, y, a=20, b=0.2, c=2 * np.pi):
    """
    Compute the Ackley function for 2D input.
    Args:
        x (float or np.ndarray): x-coordinate(s)
        y (float or np.ndarray): y-coordinate(s)
        a (float): parameter a (default 20)
        b (float): parameter b (default 0.2)
        c (float): parameter c (default 2*pi)
    Returns:
        float or np.ndarray: Ackley function value(s)
    """
    sum_sq = 0.5 * (x**2 + y**2)
    cos_comp = 0.5 * (np.cos(c * x) + np.cos(c * y))
    return -a * np.exp(-b * np.sqrt(sum_sq)) - np.exp(cos_comp) + a + np.exp(1)
