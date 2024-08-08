# random_feature_model.py

import numpy as np

class RandomFeatureModel:
    """
    A class to represent the Random Feature Model.
    """

    def __init__(self, feature_dim, rff_dim, learning_rate, eta_omega, eta_b):
        """
        Initialize the RandomFeatureModel with the specified parameters.
        
        :param feature_dim: Dimension of the features
        :param rff_dim: Dimension of the random Fourier features
        :param learning_rate: Learning rate for the model update
        :param eta_omega: Learning rate for the omega parameter update
        :param eta_b: Learning rate for the b parameter update
        """
        self.feature_dim = feature_dim
        self.rff_dim = rff_dim
        self.learning_rate = learning_rate
        self.eta_omega = eta_omega
        self.eta_b = eta_b
        self.W = np.random.randn(feature_dim, rff_dim)
        self.b = np.random.uniform(0, 2 * np.pi, rff_dim)
        self.model = np.random.randn(rff_dim)

    def update_parameters(self, x, epsilon, model):
        """
        Update the parameters W and b using the given x, epsilon, and model.

        :param x: Input data
        :param epsilon: Error term
        :param model: Current model parameters
        """
        for m in range(self.rff_dim):
            self.W[:, m] -= self.eta_omega * epsilon * model[m] * np.sin(np.dot(self.W[:, m], x) + self.b[m]) * x
            self.b[m] -= self.eta_b * epsilon * model[m] * np.sin(np.dot(self.W[:, m], x) + self.b[m])

    def transform(self, x):
        """
        Transform the input x using the random Fourier features.

        :param x: Input data
        :return: Transformed data
        """
        return np.sqrt(2 / self.rff_dim) * np.cos(np.dot(x, self.W) + self.b)
