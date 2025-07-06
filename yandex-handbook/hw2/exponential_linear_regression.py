import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge


class ExponentialLinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, *, normalize_target: bool = True, alpha: int = 1):
        self.alpha = alpha
        self.normalize_target = normalize_target

    def fit(self, X, Y):
        self.model_ = Ridge(alpha=self.alpha)

        if self.normalize_target:
            Y = np.log(Y)

        self.model_.set_params()
        self.model_.fit(X, Y)

        return self

    def predict(self, X):
        prediction = self.model_.predict(X)

        if self.normalize_target:
            return np.exp(prediction)

        return prediction


class ExponentialLinearRegression2(Ridge):
    def fit(self, X, Y):
        super().fit(X, np.log(Y))
        return self

    def predict(self, X):
        return np.exp(super().predict(X))
