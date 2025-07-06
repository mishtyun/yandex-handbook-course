import numpy as np


def root_mean_squared_logarithmic_error(
    y_true: np.ndarray, y_pred: np.ndarray, a_min=1.0
):
    if y_true[y_true < 0].sum():
        return -1

    y_pred[y_pred < a_min] = a_min

    error = np.power(np.log(y_true) - np.log(y_pred), 2)
    metric = np.sqrt(np.mean(error))

    return metric
