import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin


class MostFrequentClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mode = None

    def fit(self, X=None, y=None):
        self._mode = int(round(stats.mode(y, keepdims=True).mode[0]))
        return self

    def predict(self, X=None):
        return np.full(shape=X.shape[0], fill_value=self._mode)


class RubricCityMedianClassifier(ClassifierMixin):
    def fit(self, X=None, y=None):
        self._median_bill_by_key = (
            pd.DataFrame({"city": X["city"], "rubrics": X["modified_rubrics"], "y": y})
            .groupby(["city", "rubrics"])["y"]
            .median()
            .to_dict()
        )
        return self

    def predict(self, X=None):
        return (
            X[["city", "modified_rubrics"]]
            .apply(
                lambda x: self._median_bill_by_key[(x["city"], x["modified_rubrics"])],
                axis=1,
            )
            .values
        )


class ModifiedFeaturesClassifier(ClassifierMixin):

    def __init__(self):
        self._median_bill = None
        self._median_bill_by_key: dict = None

    def fit(self, X=None, y=None):
        self._median_bill = y.median()

        self._median_bill_by_key = (
            pd.DataFrame({"modified_features": X["modified_features"], "y": y})
            .groupby("modified_features")["y"]
            .median()
            .to_dict()
        )
        return self

    def predict(self, X=None):
        return (
            X["modified_features"]
            .apply(
                lambda x: self._median_bill_by_key.get(x, self._median_bill),
            )
            .values
        )
