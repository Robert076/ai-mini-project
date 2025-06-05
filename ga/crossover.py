import numpy as np

def binary_1_point(parent1, parent2):
    """
    Perform 1-point crossover on two binary parents.
    Args:
        parent1, parent2 (np.ndarray): Binary arrays.
    Returns:
        tuple: Two offspring arrays.
    """
    point = np.random.randint(1, len(parent1))
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    return child1, child2

def binary_2_point(parent1, parent2):
    """
    Perform 2-point crossover on two binary parents.
    Args:
        parent1, parent2 (np.ndarray): Binary arrays.
    Returns:
        tuple: Two offspring arrays.
    """
    points = np.sort(np.random.choice(range(1, len(parent1)), 2, replace=False))
    p1, p2 = points
    child1 = np.concatenate([parent1[:p1], parent2[p1:p2], parent1[p2:]])
    child2 = np.concatenate([parent2[:p1], parent1[p1:p2], parent2[p2:]])
    return child1, child2

def arithmetic_crossover(parent1, parent2, alpha=None):
    """
    Perform arithmetic crossover for real-valued parents.
    Args:
        parent1, parent2 (np.ndarray): Real-valued arrays.
        alpha (float): Mixing parameter (default random in [0,1]).
    Returns:
        tuple: Two offspring arrays.
    """
    if alpha is None:
        alpha = np.random.rand()
    child1 = alpha * parent1 + (1 - alpha) * parent2
    child2 = alpha * parent2 + (1 - alpha) * parent1
    return child1, child2

def blx_alpha_crossover(parent1, parent2, alpha=0.5):
    """
    Perform BLX-alpha crossover for real-valued parents.
    Args:
        parent1, parent2 (np.ndarray): Real-valued arrays.
        alpha (float): BLX-alpha parameter.
    Returns:
        tuple: Two offspring arrays.
    """
    c_min = np.minimum(parent1, parent2)
    c_max = np.maximum(parent1, parent2)
    diff = c_max - c_min
    low = c_min - alpha * diff
    high = c_max + alpha * diff
    child1 = np.random.uniform(low, high)
    child2 = np.random.uniform(low, high)
    return child1, child2
