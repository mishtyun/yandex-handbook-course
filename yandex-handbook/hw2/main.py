import numpy as np
import pandas as pd
from base_data_processor import BaseDataPreprocessor
from exponential_linear_regression import ExponentialLinearRegression
from root_mean_squared_logarithmic_error import root_mean_squared_logarithmic_error
from sgd_linear_regressor import SGDLinearRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_val_score,
    train_test_split,
)
from ultimate_pipeline import make_ultimate_pipeline

base = "/Users/nikita/learn/stepic-ml-colabs/content/"


SEED = 24
CV_RANDOM_STATE = 42


def load_data():
    data = pd.read_csv(base + "housing_data.csv")
    return data


def split_feature_columns(
    data: pd.DataFrame, target_column: str | None = None
) -> tuple[list, list]:
    continuous_columns = [
        key for key in data.keys() if data[key].dtype in ("int64", "float64")
    ]
    categorical_columns = [key for key in data.keys() if data[key].dtype == "object"]

    if target_column:
        continuous_columns.remove(target_column)

    print(
        f"Continuous : {len(continuous_columns)}, Categorical : {len(categorical_columns)}"
    )
    return continuous_columns, categorical_columns


def linear_regression(X_train, X_test, Y_train):
    linear_regressor = LinearRegression()
    linear_regressor.fit(X_train, Y_train)
    Y_pred = linear_regressor.predict(X_test)
    return Y_pred


def linear_ridge(X_train, X_test, Y_train):
    linear_ridge = Ridge()
    linear_ridge.fit(X_train, Y_train)
    Y_pred = linear_ridge.predict(X_test)
    return Y_pred


def exp_linear_regression(X_train, X_test, Y_train, **kwargs):
    linear_ridge = ExponentialLinearRegression(**kwargs)
    linear_ridge.fit(X_train, Y_train)
    Y_pred = linear_ridge.predict(X_test)
    return Y_pred


def compare_base_and_exp_linear_with_cv(
    X_train, X_test, Y_train, Y_test, continuous_columns
):
    base_processor = BaseDataPreprocessor(needed_columns=continuous_columns)

    X_train = base_processor.fit_transform(X_train)
    X_test = base_processor.transform(X_test)

    X_full = np.concat([X_train, X_test])
    Y_full = np.concat([Y_train, Y_test])

    cv = KFold(n_splits=5, shuffle=True, random_state=CV_RANDOM_STATE)
    mae_scorer = make_scorer(mean_absolute_error)

    linear_model_scores = cross_val_score(
        LinearRegression(), X_full, Y_full, cv=cv, scoring=mae_scorer
    )
    exp_ridge_model_scores = cross_val_score(
        ExponentialLinearRegression(), X_full, Y_full, cv=cv, scoring=mae_scorer
    )
    return np.mean(linear_model_scores), np.mean(exp_ridge_model_scores)


def hyperparameters_selection(X_full, Y_full):
    param_grid = {
        "alpha": np.logspace(-3, 3, num=7, base=10.0),
    }

    rmsle_scorer = make_scorer(
        root_mean_squared_logarithmic_error, greater_is_better=False
    )

    grid_search = GridSearchCV(
        ExponentialLinearRegression(),
        param_grid=param_grid,
        scoring=rmsle_scorer,
        cv=5,
    )
    res = grid_search.fit(X_full, Y_full)

    print(f"\nBest estimator : {res.best_estimator_}")
    print(f"Best parameters : {res.best_params_}")
    print(f"Best score : {res.best_score_}")
    return


def main(
    data_train, data_test, Y_train, Y_test, continuous_columns, categorical_columns
):
    # Base Data Processing
    base_processor = BaseDataPreprocessor(needed_columns=continuous_columns)

    X_train = base_processor.fit_transform(data_train)
    X_test = base_processor.transform(data_test)

    linear_pred = linear_regression(X_train, X_test, Y_train)
    linear_mae = mean_absolute_error(Y_test, linear_pred)
    linear_mse = mean_squared_error(Y_test, linear_pred)
    print(f"Linear Regression MAE : {linear_mae}")
    print(f"Linear Regression MSE : {linear_mse}\n")

    ridge_pred = linear_ridge(X_train, X_test, Y_train)
    ridge__mae = mean_absolute_error(Y_test, ridge_pred)
    ridge__mse = mean_squared_error(Y_test, ridge_pred)

    print(f"Ridge MAE : {ridge__mae}")
    print(f"Ridge MSE : {ridge__mse}\n")

    # Smart Data Processing
    # smart_processor = SmartDataPreprocessor(needed_columns=continuous_columns)
    # X_train = smart_processor.fit_transform(data_train)
    # X_test = smart_processor.transform(data_test)
    from root_mean_squared_logarithmic_error import root_mean_squared_logarithmic_error

    ridge__custom_metric = root_mean_squared_logarithmic_error(Y_test, ridge_pred)
    print(f"Ridge Root Mean Squared Logarithmic Error : {ridge__custom_metric}")

    # exp_ridge_pred = exp_linear_regression(
    #     X_train, X_test, Y_train, normalize_target=False
    # )
    # exp_ridge__mae = mean_absolute_error(Y_test, exp_ridge_pred)
    # exp_ridge__mse = mean_squared_error(Y_test, exp_ridge_pred)

    # print(f"(non Exp) LinearRegression MAE : {exp_ridge__mae}")
    # print(f"(non Exp) LinearRegression MSE : {exp_ridge__mse}\n")

    # exp_ridge_pred = exp_linear_regression(X_train, X_test, Y_train)
    # exp_ridge__mae = mean_absolute_error(Y_test, exp_ridge_pred)
    # exp_ridge__mse = mean_squared_error(Y_test, exp_ridge_pred)

    # print(f"Exp LinearRegression MAE : {exp_ridge__mae}")
    # print(f"Exp LinearRegression MSE : {exp_ridge__mse}\n")

    return


def test_sgd_linear_regressor(X_train, Y_train, X_test, Y_test):
    model = SGDLinearRegressor()
    model.fit(X_train, Y_train)

    prediction = model.predict(X_test)
    print("SGD MAE : ", mean_absolute_error(Y_test, prediction))


def test_make_ultimate_pipeline(X_train, Y_train, X_test, Y_test):
    pipe = make_ultimate_pipeline()
    pipe.fit(X_train, Y_train)

    # print(pipe.best_params_)
    # print(pipe.best_score_)

    prediction = pipe.predict(X_test)
    print("Pipeline MAE : ", mean_absolute_error(Y_test, prediction))


if __name__ == "__main__":
    data = load_data()

    target_column = "Sale_Price"
    np.random.seed(SEED)

    continuous_columns, categorical_columns = split_feature_columns(data, target_column)

    test_size = 0.2
    data_train, data_test, Y_train, Y_test = train_test_split(
        data[data.columns.drop("Sale_Price")],
        np.array(data["Sale_Price"]),
        test_size=test_size,
        random_state=SEED,
    )
    print(f"Train : {data_train.shape} {Y_train.shape}")
    print(f"Test : {data_test.shape} {Y_test.shape}")

    base_processor = BaseDataPreprocessor(needed_columns=continuous_columns)

    X_train = base_processor.fit_transform(data_train)
    X_test = base_processor.transform(data_test)

    # one_hot_processor = OneHotPreprocessor(needed_columns=continuous_columns)
    # X_train2 = one_hot_processor.fit_transform(data_train)
    # X_test2 = one_hot_processor.transform(data_test)

    # X_full = base_processor.transform(data)
    # Y_full = np.array(data[target_column])

    if False:
        main(
            data_train,
            data_test,
            Y_train,
            Y_test,
            continuous_columns,
            categorical_columns,
        )
    if False:
        res = compare_base_and_exp_linear_with_cv(
            data_train,
            data_test,
            Y_train,
            Y_test,
            continuous_columns,
        )
    if False:
        res = hyperparameters_selection(X_full, Y_full)

    # test_sgd_linear_regressor(X_train, Y_train, X_test, Y_test)
    test_make_ultimate_pipeline(data_train, Y_train, data_test, Y_test)
