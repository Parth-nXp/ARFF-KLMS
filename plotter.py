import numpy as np
import matplotlib.pyplot as plt

def plot_mse(mse_values):
    """
    Plot the Mean Squared Error values.

    :param mse_values: Mean Squared Error values
    """
    mse_value_all_trials = 10 * np.log10(mse_values)
    plt.plot(mse_value_all_trials)
    plt.xlabel('Iterations')
    plt.ylabel('MSE (dB)')
    plt.title('Mean Squared Error over Iterations')
    plt.show()
