from collections import Counter

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from mean_regressor import CityMeanRegressor, MeanRegressor
from most_frequent_classifier import (
    ModifiedFeaturesClassifier,
    MostFrequentClassifier,
    RubricCityMedianClassifier,
)
from sklearn.metrics import balanced_accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

base = "/Users/nikita/learn/stepic-ml-colabs/content/public_places/"


def load_data():
    data = pd.read_csv(base + "organisations.csv")
    features = pd.read_csv(base + "features.csv")
    rubrics = pd.read_csv(base + "rubrics.csv")
    return data, features, rubrics


def get_cleaned_data(data):
    cleaned_data = data.copy()

    cleaned_data = cleaned_data[cleaned_data.average_bill.notnull()]
    cleaned_data = cleaned_data[cleaned_data.average_bill.between(0, 2500)]

    rating_media = cleaned_data.rating.median().astype(float).round(1)
    cleaned_data["rating"] = (
        cleaned_data["rating"].astype(float).round(1).fillna(rating_media)
    )

    return cleaned_data


def mean_reg(data_train, data_test):
    reg = MeanRegressor()
    reg.fit(y=data_train["average_bill"])

    reg_predict = reg.predict(X=data_test)
    reg_rmse = np.sqrt(
        mean_squared_error(y_true=data_test["average_bill"], y_pred=reg_predict)
    )

    print(f"\n-----------------------")
    print(f"1. MeanRegressor:\n\trmse={reg_rmse}")


def city_mean_reg(data_train, data_test):
    city_reg = CityMeanRegressor()

    city_reg.fit(X=data_train, y=data_train["average_bill"])

    city_reg_predict = city_reg.predict(X=data_test)

    rmse = np.sqrt(
        mean_squared_error(y_true=data_test["average_bill"], y_pred=city_reg_predict)
    )
    print(f"\n-----------------------")
    print(f"2. CityMeanRegressor: \n\trmse={rmse}")


def most_freq_class(data_train, data_test):
    clf = MostFrequentClassifier()
    clf.fit(y=data_train["average_bill"])

    clf_predict = clf.predict(X=data_test)
    y_test = data_test["average_bill"]

    clf_rmse = np.sqrt(mean_squared_error(y_true=y_test, y_pred=clf_predict))
    clf_accuracy_score = balanced_accuracy_score(
        y_true=data_test["average_bill"], y_pred=clf_predict
    )

    print(f"\n-----------------------")
    print(
        f"3. MostFrequentClassifier: \n\taccuracy score: {clf_accuracy_score}\n\trmse={clf_rmse}"
    )
    return y_test, clf_predict


def city_and_rubrics_most_freq_class(data_train, data_test):
    clf = RubricCityMedianClassifier()
    clf.fit(X=data_train, y=data_train["average_bill"])

    clf_predict = clf.predict(X=data_test)
    y_test = data_test["average_bill"]

    clf_rmse = np.sqrt(mean_squared_error(y_true=y_test, y_pred=clf_predict))
    clf_accuracy_score = balanced_accuracy_score(
        y_true=data_test["average_bill"], y_pred=clf_predict
    )

    print(f"\n-----------------------")
    print(
        f"4. RubricCityMedianClassifier: \n\taccuracy score: {clf_accuracy_score}\n\trmse={clf_rmse}"
    )
    return y_test, clf_predict


def modified_features_classifier(data_train, data_test):
    clf = ModifiedFeaturesClassifier()
    clf.fit(X=data_train, y=data_train["average_bill"])

    clf_predict = clf.predict(X=data_test)
    y_test = data_test["average_bill"]

    clf_train_predict = clf.predict(X=data_train)
    y_train_test = data_train["average_bill"]

    clf_train_rmse = np.sqrt(
        mean_squared_error(y_true=y_train_test, y_pred=clf_train_predict)
    )
    clf_train_accuracy_score = balanced_accuracy_score(
        y_true=y_train_test, y_pred=clf_train_predict
    )

    clf_rmse = np.sqrt(mean_squared_error(y_true=y_test, y_pred=clf_predict))
    clf_accuracy_score = balanced_accuracy_score(y_true=y_test, y_pred=clf_predict)

    print(f"\n-----------------------")
    print(
        f"5. ModifiedFeaturesClassifier:\n\taccuracy score: {clf_accuracy_score}\n\trmse={clf_rmse}"
    )
    print(
        f"\nTrain:\n\taccuracy score: {clf_train_accuracy_score}\n\trmse={clf_train_rmse}"
    )
    return y_test, clf_predict


def modify_rubrics(
    train_data: pd.DataFrame, test_data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    В исходных данных есть колонка rubrics_id, которая может в себе содержать от одного, до нескольких элементов
    Необходимо добавить новую фичу - modified_rubrics которая будет в себе содержать то же что и в rubrics_id, если таких
    комбинаций рубрик больше 100 из обучающей выборки, иначе там будет строка `other`
    """

    counts = Counter(train_data["rubrics_id"])

    def _modify_rubrics(rubrics):
        if counts[rubrics] >= 100:
            return rubrics
        else:
            return "other"

    modifed_train_data = train_data.copy()
    modifed_test_data = test_data.copy()

    modifed_train_data["modified_rubrics"] = modifed_train_data["rubrics_id"].apply(
        _modify_rubrics
    )
    modifed_test_data["modified_rubrics"] = modifed_test_data["rubrics_id"].apply(
        _modify_rubrics
    )

    return modifed_train_data, modifed_test_data


def modify_features(
    train_data: pd.DataFrame, test_data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:

    def concat_features(df: pd.DataFrame) -> pd.Series:
        return df["rubrics_id"] + " q " + df["features_id"]

    train_data = train_data.copy()
    test_data = test_data.copy()

    train_data["modified_features"] = concat_features(train_data)
    test_data["modified_features"] = concat_features(test_data)

    allowed_features = set(train_data["modified_features"])

    test_data["modified_features"] = test_data["modified_features"].apply(
        lambda x: x if x in allowed_features else "other"
    )

    return train_data, test_data


def preview_class_accuracy(y_test, y_pred):
    df = pd.DataFrame({"true": y_test, "pred": y_pred})

    per_class_stats = df.groupby("true").apply(
        lambda group: pd.Series(
            {
                "total": len(group),
                "correct": (group["pred"] == group.name).sum(),
                "accuracy": (group["pred"] == group.name).mean(),
            }
        ),
        include_groups=False,
    )

    per_class_stats["accuracy"] = per_class_stats["correct"] / per_class_stats["total"]

    print(per_class_stats)


def save_to_csv(y_test, y_pred):
    result_df = pd.DataFrame(
        {"shop_id": y_test.index, "predicted_average_bill": y_pred}
    )

    result_df.to_csv("predictions.csv", index=False)


def sparse_data(
    train_data: pd.DataFrame, test_data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from scipy.sparse import csr_matrix, hstack

    sparse_train_data = train_data.copy()
    sparse_test_data = test_data.copy()

    city_map = {"msk": 0, "spb": 1}
    sparse_train_data["city"] = sparse_train_data["city"].map(city_map)
    sparse_test_data["city"] = sparse_test_data["city"].map(city_map)

    city_csr_matrix_train = csr_matrix(sparse_train_data[["city"]].values)
    city_csr_matrix_test = csr_matrix(sparse_test_data[["city"]].values)

    sparse_train_data["rubrics_id"] = sparse_train_data["rubrics_id"].str.split()
    sparse_train_data["features_id"] = sparse_train_data["features_id"].str.split()

    sparse_test_data["rubrics_id"] = sparse_test_data["rubrics_id"].str.split()
    sparse_test_data["features_id"] = sparse_test_data["features_id"].str.split()

    mlb_rubrics = MultiLabelBinarizer(sparse_output=True)
    rubrics_train = mlb_rubrics.fit_transform(sparse_train_data["rubrics_id"])
    rubrics_test = mlb_rubrics.transform(sparse_test_data["rubrics_id"])

    mlb_features = MultiLabelBinarizer(sparse_output=True)
    features_train = mlb_features.fit_transform(sparse_train_data["features_id"])
    features_test = mlb_features.transform(sparse_test_data["features_id"])

    # Вариант 1 - разрядить рейтинг, предварительно немного нормализовав данные до float(1) и убрав пропуски
    # encoder = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
    # rating_train_sparse = encoder.fit_transform(sparse_train_data[["rating"]])
    # rating_test_sparse = encoder.transform(sparse_test_data[["rating"]])

    # Вариант 2 - не разряжать рейтинг
    rating_train_sparse = csr_matrix(sparse_train_data[["rating"]].values)
    rating_test_sparse = csr_matrix(sparse_test_data[["rating"]].values)

    known_rubrics = set(mlb_rubrics.classes_)
    known_features = set(mlb_features.classes_)

    def count_unknowns(row_ids, known_ids):
        return sum(1 for item in row_ids if item not in known_ids)

    rubrics_test_unknown = sparse_test_data["rubrics_id"].apply(
        lambda ids: count_unknowns(ids, known_rubrics)
    )
    features_test_unknown = sparse_test_data["features_id"].apply(
        lambda ids: count_unknowns(ids, known_features)
    )

    feature_other_test = rubrics_test_unknown + features_test_unknown
    feature_other_test_sparse = csr_matrix(feature_other_test.values.reshape(-1, 1))
    feature_other_train_sparse = csr_matrix(np.zeros((sparse_train_data.shape[0], 1)))

    X_train_final = hstack(
        [
            rating_train_sparse,
            rubrics_train,
            features_train,
            city_csr_matrix_train,
            feature_other_train_sparse,
        ]
    )

    X_test_final = hstack(
        [
            rating_test_sparse,
            rubrics_test,
            features_test,
            city_csr_matrix_test,
            feature_other_test_sparse,
        ]
    )

    return X_train_final, X_test_final


def main():
    data, features, rubrics = load_data()
    rubric_dict = rubrics.set_index("rubric_id").T.to_dict("records")[0]

    cleaned_data = get_cleaned_data(data)

    clean_data_train, clean_data_test = train_test_split(
        cleaned_data,
        stratify=cleaned_data["average_bill"],
        test_size=0.33,
        random_state=42,
    )
    modifed_clean_train_data, modifed_clean_test_data = modify_rubrics(
        clean_data_train, clean_data_test
    )

    mean_reg(clean_data_train, clean_data_test)
    city_mean_reg(clean_data_train, clean_data_test)

    y_test, y_pred = most_freq_class(clean_data_train, clean_data_test)
    preview_class_accuracy(y_test, y_pred)

    y_test, y_pred = city_and_rubrics_most_freq_class(
        modifed_clean_train_data, modifed_clean_test_data
    )
    preview_class_accuracy(y_test, y_pred)

    feature_modified_train_data, feature_modified_test_data = modify_features(
        clean_data_train, clean_data_test
    )
    y_test, y_pred = modified_features_classifier(
        feature_modified_train_data, feature_modified_test_data
    )
    # save_to_csv(y_test, y_pred)

    sparse_train_data, sparse_test_data = sparse_data(clean_data_train, clean_data_test)

    cat_boost = CatBoostClassifier()
    cat_boost.fit(
        sparse_train_data.toarray(),
        clean_data_train["average_bill"],
    )
    cat_boost_predict = cat_boost.predict(sparse_test_data.toarray())

    cat_boost_accuracy_score = balanced_accuracy_score(
        y_true=clean_data_test["average_bill"], y_pred=cat_boost_predict
    )

    print(f"CatBoost accuracy score: {cat_boost_accuracy_score:.3f}")


if __name__ == "__main__":
    main()
