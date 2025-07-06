import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.discriminant_analysis import StandardScaler


class SmartDataPreprocessor(TransformerMixin):
    def __init__(self, needed_columns: list[str] | None = None):
        """
        :param needed_columns: if not None select these columns from the dataframe
        """
        self.scaler = StandardScaler()

        self._needed_columns: list[str] | None = needed_columns
        self._selected_columns: list[str] | set[str] | None = None

    def get_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a dataframe with only needed columns
        :param data: pd.DataFrame with all data
        """
        if self._needed_columns is not None:
            return data[self._needed_columns]
        return data

    def fit(self, data, *args):
        """
        Prepares the class for future transformations
        :param data: pd.DataFrame with all available columns
        :return: self
        """
        df = data.copy()
        df = self.get_data(df)

        df = df.select_dtypes(include=["int", "float"])
        self._selected_columns = df.columns.tolist()
        self.scaler.fit(df)

        return self

    def transform(self, data: pd.DataFrame) -> np.array:
        """
        Transforms features so that they can be fed into the regressors
        :param data: pd.DataFrame with all available columns
        :return: np.array with preprocessed features
        """
        df = data.copy()

        if self._selected_columns:
            df = df[self._selected_columns]

        df[df.Lot_Frontage == 0] = df.Lot_Frontage.median()
        # iowa_center_coordinates = (41.5868, -93.6250)

        return self.scaler.transform(df)
