import numpy as np

def generate_synthetic_data(num_iterations, feature_dim):
    """
    Generate synthetic data for the experiment.

    :param num_iterations: Number of iterations
    :param feature_dim: Dimension of the features
    :return: Generated uk, nuk, x, and y
    """
    theta_k = np.random.uniform(0.2, 0.9)
    mu_k = np.random.uniform(-0.2, 0.2)
    sigma2_uk = np.random.uniform(0.2, 1.2)
    sigma2_nuk = np.random.uniform(0.005, 0.03)
    uk = np.random.normal(mu_k, np.sqrt(sigma2_uk), (num_iterations, feature_dim))
    nuk = np.random.normal(0, np.sqrt(sigma2_nuk), (num_iterations, 1))
    x = np.zeros((num_iterations, feature_dim))
    y = np.zeros((num_iterations, 1))
    x[0] = uk[0]

    for n in range(1, num_iterations):
        x[n] = theta_k * x[n-1] + np.sqrt(1 - theta_k**2) * uk[n]
        y[n] = np.sqrt(x[n, 0]**2 + np.sin(np.pi * x[n, 3])**2) + (0.8 - 0.5 * np.exp(-x[n, 1]**2)) * x[n, 2] + nuk[n]

    return uk, nuk, x, y
