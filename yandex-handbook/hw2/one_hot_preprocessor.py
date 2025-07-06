import numpy as np
import pandas as pd
from base_data_processor import BaseDataPreprocessor
from sklearn.preprocessing import OneHotEncoder

interesting_columns = ["Overall_Qual", "Garage_Qual", "Sale_Condition", "MS_Zoning"]


class OneHotPreprocessor(BaseDataPreprocessor):
    def __init__(
        self, *, handle_unknown: str = "ignore", drop: str | None = None, **kwargs
    ):
        super().__init__(**kwargs)

        self._columns_to_one_hot = interesting_columns
        self._hot_encoded_columns: list[str] | None = None

        self.one_hot_encoder = OneHotEncoder(handle_unknown=handle_unknown, drop=drop)

    def fit(self, data, *args):
        super().fit(data, *args)

        self.one_hot_encoder.fit(data[self._columns_to_one_hot])
        self._hot_encoded_columns = self.one_hot_encoder.get_feature_names_out()

        return self

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        transformed_data: np.ndarray = super().transform(data)

        one_hot_transformed_data = self.one_hot_encoder.transform(
            data[self._columns_to_one_hot]
        ).toarray()

        res_transformed_data = np.concatenate(
            [transformed_data, one_hot_transformed_data], axis=1
        )

        return res_transformed_data
