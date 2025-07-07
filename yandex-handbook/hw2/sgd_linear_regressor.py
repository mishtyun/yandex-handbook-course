import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class SGDLinearRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        lr=1e-3,
        regularization: float = 1,
        delta_converged: float = 1e-3,
        max_steps: int = 1000,
        batch_size: int = 64,
    ):
        self.lr = lr
        self.regularization = regularization
        self.delta_converged = delta_converged
        self.max_steps = max_steps
        self.batch_size = batch_size

        self.W = None
        self.b = None

    def fit(self, X, Y):
        n_samples, n_features = X.shape

        self.b = 0
        self.W = np.zeros(n_features)

        for _ in range(self.max_steps):

            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            X_shuffled = X[indices]
            Y_shuffled = Y[indices]

            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i : i + self.batch_size]
                Y_batch = Y_shuffled[i : i + self.batch_size]

                old_bias = self.b
                old_weights = self.W.copy()

                y_pred = X_batch.dot(old_weights) + old_bias
                error = Y_batch - y_pred

                bias_grad = -2 * np.sum(error) / self.batch_size
                weights_grad = (
                    -2 * X_batch.T.dot(error) / self.batch_size
                    + 2 * self.regularization * old_weights
                )

                self.b -= self.lr * bias_grad
                self.W -= self.lr * weights_grad

                if np.linalg.norm(old_weights - self.W) < self.delta_converged:
                    break

        return self

    def predict(self, X) -> np.ndarray:
        return X.dot(self.W) + self.b
