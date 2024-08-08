from experiment import run_experiment
from plotter import plot_mse

if __name__ == "__main__":
    independent_experiment = 500
    feature_dim = 5
    rff_dim = 200
    learning_rate = 0.75
    num_iterations = 1000
    eta_omega = 0.05
    eta_b = 0.05

    mse_values_all_trials = run_experiment(independent_experiment, feature_dim, rff_dim, learning_rate, num_iterations, eta_omega, eta_b)
    plot_mse(mse_values_all_trials)
