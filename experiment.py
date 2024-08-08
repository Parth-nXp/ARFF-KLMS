import numpy as np
from random_feature_model import RandomFeatureModel
from data_generator import generate_synthetic_data

def run_experiment(independent_experiment, feature_dim, rff_dim, learning_rate, num_iterations, eta_omega, eta_b):
    """
    Run the experiment with the specified parameters.

    :param independent_experiment: Number of independent experiments
    :param feature_dim: Dimension of the features
    :param rff_dim: Dimension of the random Fourier features
    :param learning_rate: Learning rate for the model update
    :param num_iterations: Number of iterations
    :param eta_omega: Learning rate for the omega parameter update
    :param eta_b: Learning rate for the b parameter update
    :return: Mean Squared Error values for all trials
    """
    mse_values_all_trials = np.zeros(num_iterations)

    for _ in range(independent_experiment):
        model = RandomFeatureModel(feature_dim, rff_dim, learning_rate, eta_omega, eta_b)
        _, _, x, y = generate_synthetic_data(num_iterations, feature_dim)
        mse_values_per_iteration = np.zeros(num_iterations)

        for n in range(1, num_iterations):
            z = model.transform(x[n])
            epsilon = y[n] - np.dot(model.model, z)
            model.model += learning_rate * z * epsilon
            mse_values_per_iteration[n] = epsilon**2
            model.update_parameters(x[n], epsilon, model.model)

        mse_values_all_trials += mse_values_per_iteration

    mse_values_all_trials /= independent_experiment
    mse_values_all_trials /= max(mse_values_all_trials)

    return mse_values_all_trials
