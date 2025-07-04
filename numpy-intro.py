import numpy as np


def cumulative_mean_v1(X: np.ndarray):
    result = np.cumsum(X)
    return [v / (i[0] + 1) for i, v in np.ndenumerate(result)]


def cumulative_mean_v2(X: np.ndarray):
    result = np.cumsum(X)
    n = np.arange(1, len(X) + 1)

    return result / n


def test_cumulative_mean():
    a = np.array([1, 2, 4, -9])

    assert np.allclose(cumulative_mean_v1(a), np.array([1, 1.5, 7.0 / 3, -0.5]))
    assert np.allclose(cumulative_mean_v2(a), np.array([1, 1.5, 7.0 / 3, -0.5]))


# ------------
def cumulative_mean_2d(A: np.ndarray):
    n = np.matrix([np.arange(1, A.shape[1] + 1) for _ in range(A.shape[0])])
    return np.cumsum(A, axis=1) / n


def cumulative_mean_2d_v2(A: np.ndarray):
    return np.cumsum(A, axis=1) / np.cumsum(np.ones(A.shape), axis=1, dtype=int)


def test_cumulative_mean_2d():
    a = np.array([1, 2, 4, -9, 1, 1, 1, 1]).reshape(2, 4)

    assert np.allclose(
        cumulative_mean_2d(a),
        np.array([1, 1.5, 7.0 / 3, -0.5, 1, 1, 1, 1]).reshape(2, 4),
    )
    assert np.allclose(
        cumulative_mean_2d_v2(a),
        np.array([1, 1.5, 7.0 / 3, -0.5, 1, 1, 1, 1]).reshape(2, 4),
    )


# ------------


def transform(X):
    result = X.copy()
    result[1::2] = 1
    result[::2] **= 3
    return result


def transform_v2(X):
    result = np.ones(X.shape)

    for i, v in np.ndenumerate(X):
        if i[0] % 2 == 0:
            result[i[0]] = v**3

    return result


def test_transform():
    X = np.array([5, -2, 44, 8, 0, -2, -3, 1, 5, 3])
    assert np.allclose(transform(X), np.array([125, 1, 85184, 1, 0, 1, -27, 1, 125, 1]))


# ------------
def diag_2k(a):
    diag_array = np.diag(a)
    return np.sum(diag_array[diag_array % 2 == 0])


def test_diag_2k():
    a = np.array([1, 3, 6, 2, 8, -2, -2, 0, -2], dtype=int).reshape(3, 3)
    assert np.allclose(diag_2k(a), 6)


if __name__ == "__main__":
    test_cumulative_mean()
    test_cumulative_mean_2d()
    test_transform()
    test_diag_2k()
