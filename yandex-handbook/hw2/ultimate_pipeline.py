from sgd_linear_regressor import SGDLinearRegressor
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

one_hot_encode_columns = ["Overall_Qual", "Garage_Qual", "Sale_Condition", "MS_Zoning"]


def get_pipeline_preprocessors():
    one_hot_preprocessors = [
        (
            OneHotEncoder(handle_unknown="ignore", drop="first"),
            one_hot_encode_columns,
        ),
        (
            StandardScaler(),
            [],
        ),
    ]

    return ColumnTransformer(transformers=one_hot_preprocessors)


def make_ultimate_pipeline() -> Pipeline:

    preprocessors = get_pipeline_preprocessors()

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessors),
            ("classifier", SGDLinearRegressor()),
        ],
    )

    return pipe
