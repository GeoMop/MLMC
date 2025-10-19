import numpy as np
import matplotlib

matplotlib.rcParams.update({'font.size': 22})
import matplotlib.pyplot as plt


def log_var_per_level(l_vars, levels=None, moments=[0], err_l_vars=None):
    """
    Plot log₂ of variance per level and fit a slope to estimate the decay rate β.

    The function plots the base-2 logarithm of the variance for each level
    and fits a linear model to estimate the convergence rate β, based on
    the slope of log₂(variance) vs. level.

    :param l_vars: Array of shape (n_levels, n_moments) representing
                   the variance of each moment at each level.
    :param levels: Optional array of level indices (default: np.arange(n_levels)).
    :param moments: List of moment indices to include in the plot.
    :param err_l_vars: Optional array of errors corresponding to l_vars.
    :return: None
    """
    n_levels = l_vars.shape[0]
    if levels is None:
        levels = np.arange(n_levels)

    fig, ax = plt.subplots(figsize=(8, 5))

    for m in moments:
        y = np.log2(l_vars[:, m])
        ax.plot(levels, y, 'o-', label=f'm={m}')

        slope, intercept = np.polyfit(levels, y, 1)
        beta = -slope
        ax.plot(
            levels,
            slope * levels + intercept,
            '--',
            label=f'fit m={m}: slope={slope:.2f}, beta≈{beta:.2f}'
        )

    ax.set_ylabel(r'$\log_2 \, V_\ell$')
    ax.set_xlabel('level $\ell$')
    ax.legend()
    ax.grid(True, which="both")
    plt.tight_layout()
    plt.show()


def log_mean_per_level(l_means, err_means=0, err_l_means=0, moments=[1, 2, 3, 4]):
    """
    Plot log₂ of absolute mean per level for specified statistical moments.

    :param l_means: Array of mean values per level and moment.
    :param err_means: Optional array of mean estimation errors (unused).
    :param err_l_means: Optional array of level-mean estimation errors (unused).
    :param moments: List of moment indices to include in the plot.
    :return: None
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))
    print("l means ", l_means)
    for m in moments:
        line2, = ax1.plot(np.log2(np.abs(l_means[:, m])), label=f"m={m}", marker="s")

    ax1.set_ylabel('log' + r'$_2$' + 'mean')
    ax1.set_xlabel('level' + r'$l$')
    plt.legend()
    plt.tight_layout()
    plt.show()


def sample_cost_per_level(costs, levels=None):
    """
    Plot log₂ of sample cost per level and fit a slope to estimate γ.

    The slope of the linear regression line provides an estimate of the
    cost scaling parameter γ.

    :param costs: Array of computational costs per sample for each level.
    :param levels: Optional array of level indices (default: 0, 1, ...).
    :return: Estimated γ (float), the slope of the fitted line.
    """
    n_levels = len(costs)
    if levels is None:
        levels = np.arange(n_levels)

    y = np.log2(costs)
    slope, intercept = np.polyfit(levels, y, 1)
    gamma = slope

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(levels, y, 'o-', label='log2(cost)')
    ax.plot(
        levels,
        slope * levels + intercept,
        '--',
        label=f'fit: slope={slope:.2f}, gamma≈{gamma:.2f}'
    )

    ax.set_ylabel(r'$\log_2 \, C_\ell$')
    ax.set_xlabel('level $\ell$')
    ax.legend()
    ax.grid(True, which="both")
    plt.tight_layout()
    plt.show()

    return gamma


def variance_to_cost_ratio(l_vars, costs, moments=[1, 2, 3, 4]):
    """
    Plot the log₂ of variance-to-cost ratio per level for given statistical moments.

    The ratio Vₗ/Cₗ is computed for each level, and the slope of its
    log₂-linear fit indicates the decay behavior relative to computational cost.

    :param l_vars: Array of variances per level and moment (shape: n_levels × n_moments).
    :param costs: Array of costs per sample for each level.
    :param moments: List of moment indices to include in the plot.
    :return: None
    """
    print("l_vars ", l_vars)
    print(costs)
    n_levels = l_vars.shape[0]
    levels = np.arange(n_levels)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    print('costs ', costs)
    print("levels ", levels)
    for m in moments:
        line2, = ax1.plot(np.log2(l_vars[:, m] / costs), label=f"m={m}", marker="s")

        print("l vars ", l_vars[:, m])
        print("np.log2(l_vars[:, m]/costs) ", np.log2(l_vars[:, m] / costs))

        # Fit a straight line: log2(V/C) ≈ a + b * level
        coeffs = np.polyfit(levels, np.log2(l_vars[:, m] / costs), 1)
        slope, intercept = coeffs[0], coeffs[1]
        ax1.plot(levels, slope * levels + intercept, '--', label=f'fit: slope={slope:.2f}')

    ax1.set_ylabel('log' + r'$_2$' + 'variance to cost ratio')
    ax1.set_xlabel('level' + r'$l$')
    plt.legend()
    plt.tight_layout()
    plt.show()


def kurtosis_per_level(means, l_means, err_means=0, err_l_means=0, moments=[1, 2, 3, 4]):
    """
    Plot log₂ of mean values per level (often used for analyzing kurtosis trends).

    :param means: Array of global mean values per moment (unused in plotting).
    :param l_means: Array of level-wise mean values per moment.
    :param err_means: Optional array of mean estimation errors (unused).
    :param err_l_means: Optional array of level-mean estimation errors (unused).
    :param moments: List of moment indices to include in the plot.
    :return: None
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))
    for m in moments:
        line2, = ax1.plot(np.log2(np.abs(l_means[:, m])), label=f"m={m}", marker="s")

    ax1.set_ylabel('log ' + r'$_2$ ' + 'mean')
    ax1.set_xlabel('level ' + r'$l$')
    plt.legend()
    plt.tight_layout()
    plt.show()
