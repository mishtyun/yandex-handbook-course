import numpy as np
import pandas as pd
from base_data_processor import BaseDataPreprocessor
from root_mean_squared_logarithmic_error import root_mean_squared_logarithmic_error
from sgd_linear_regressor import SGDLinearRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

one_hot_encode_columns = [
    "Overall_Qual",
    "Garage_Qual",
    "Sale_Condition",
    "MS_Zoning",
]
continuous_columns = [
    "Lot_Frontage",
    "Lot_Area",
    "Year_Built",
    "Year_Remod_Add",
    "Mas_Vnr_Area",
    "BsmtFin_SF_1",
    "BsmtFin_SF_2",
    "Bsmt_Unf_SF",
    "Total_Bsmt_SF",
    "First_Flr_SF",
    "Second_Flr_SF",
    "Gr_Liv_Area",
    "Bsmt_Full_Bath",
    "Bsmt_Half_Bath",
    "Full_Bath",
    "Half_Bath",
    "Bedroom_AbvGr",
    "Kitchen_AbvGr",
    "TotRms_AbvGrd",
    "Fireplaces",
    "Garage_Cars",
    "Garage_Area",
    "Wood_Deck_SF",
    "Open_Porch_SF",
    "Enclosed_Porch",
    "Screen_Porch",
    "Misc_Val",
    "Mo_Sold",
    "Year_Sold",
]


class DistanceFromCenterTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        lat_col: str = "Latitude",
        lon_col: str = "Longitude",
        center_coords: tuple = (42.0, -93.0),
    ):
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.center_coords = center_coords

    def fit(self, X, Y=None):
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        lat = X[self.lat_col].values
        lon = X[self.lon_col].values
        lat0, lon0 = self.center_coords

        dist = np.sqrt((lat - lat0) ** 2 + (lon - lon0) ** 2)
        return dist.reshape(-1, 1)


class OutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self, columns, factor=1.5):
        self.columns = columns
        self.factor = factor
        self.bounds = {}

    def fit(self, X, y=None):
        for col in self.columns:
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            self.bounds[col] = (q1 - self.factor * iqr, q3 + self.factor * iqr)
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in self.columns:
            low, high = self.bounds[col]
            X_copy[col] = X_copy[col].clip(lower=low, upper=high)
        return X_copy


def get_pipeline_preprocessors():

    numerical_pipeline = Pipeline(
        steps=[
            (
                "clipper",
                OutlierClipper(
                    columns=["Lot_Frontage", "Lot_Area", "Bsmt_Full_Bath", "Full_Bath"],
                    factor=1.5,
                ),
            ),
            ("scaler", BaseDataPreprocessor(needed_columns=continuous_columns)),
        ]
    )

    preprocessors = [
        ("numerical", numerical_pipeline, continuous_columns),
        (
            "distance",
            DistanceFromCenterTransformer(center_coords=(42.0, -93.0)),
            ["Latitude", "Longitude"],
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore"),
            one_hot_encode_columns,
        ),
    ]

    return ColumnTransformer(
        transformers=preprocessors,
    )


def make_ultimate_pipeline() -> GridSearchCV:
    preprocessors = get_pipeline_preprocessors()

    param_grid = {
        "regularization": np.logspace(-5, 3, num=9),
        "lr": [1e-5, 1e-4, 1e-3],
    }

    rmsle_scorer = make_scorer(
        root_mean_squared_logarithmic_error, greater_is_better=False
    )

    sgd_grid = GridSearchCV(
        estimator=SGDLinearRegressor(),
        param_grid=param_grid,
        scoring=rmsle_scorer,
        refit=True,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
    )

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessors),
            ("sgd_grid", sgd_grid),
        ],
    )

    return pipe
