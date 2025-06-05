import numpy as np
from scipy import stats

def summarize_results(results):
    """
    Compute best, mean, and std of fitness values from a list/array.
    Args:
        results (list or np.ndarray): Fitness values from multiple runs.
    Returns:
        dict: {'best': float, 'mean': float, 'std': float}
    """
    results = np.array(results)
    return {
        'best': np.min(results),
        'mean': np.mean(results),
        'std': np.std(results)
    }

def t_test(group1, group2):
    """
    Perform independent t-test between two groups of results.
    Args:
        group1, group2 (list or np.ndarray): Fitness values.
    Returns:
        t-statistic, p-value
    """
    return stats.ttest_ind(group1, group2, equal_var=False)

def anova(*groups):
    """
    Perform one-way ANOVA across multiple groups.
    Args:
        *groups: Variable number of arrays/lists of fitness values.
    Returns:
        F-statistic, p-value
    """
    return stats.f_oneway(*groups)
