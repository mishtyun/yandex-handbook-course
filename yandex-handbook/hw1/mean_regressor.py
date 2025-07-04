import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class MeanRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, *args, **kwargs):
        self._mean = None

    def fit(self, X=None, y=None):
        self._mean = y.mean()
        return self

    def predict(self, X=None):
        return np.full(shape=X.shape[0], fill_value=self._mean)


class CityMeanRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, *args, **kwargs):
        self._mean_bill_by_city: pd.Series = None

    def fit(self, X=None, y=None):
        self._mean_bill_by_city = (
            pd.DataFrame({"city": X["city"], "y": y})
            .groupby("city")["y"]
            .mean()
            .to_dict()
        )
        return self

    def predict(self, X=None):
        return X["city"].map(self._mean_bill_by_city).values
