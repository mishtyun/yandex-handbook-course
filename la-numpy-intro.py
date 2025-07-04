import numpy as np
from numpy import linalg as LA


def no_numpy_scalar(v1, v2):
    """
    v1: list[n] --- первый вектор-аргумент длиной n
    v2: list[n] --- второй вектор-аргумент длиной n
    return c: float  --- результат скалярного произведения векторов v1 и v2

    Функция принимает на вход два вектора длиной n
    Возвращает число, равное их скалярному произведению v1 и v2

    Реализуйте скалярное умножение векторов, не используя функции из пакета numpy

    """

    result = 0
    for iv1, iv2 in zip(v1, v2):
        result += iv1 * iv2

    return result


def numpy_scalar(v1, v2):
    """
    v1: np.array[n] --- первый вектор-аргумент длиной n
    v2: np.array[n] --- второй вектор-аргумент длиной n
    return c: float  --- результат скалярного произведения векторов v1 и v2

    Функция принимает на вход два вектора длиной n
    Возвращает число, равное их скалярному произведению

    Реализуйте скалярное умножение векторов, используя функции из пакета numpy
    """
    return v1 @ v2


def test_scalars():
    a = np.random.sample((1, 3))
    b = np.random.sample((1, 3))

    a = list(a)[0]
    b = list(b)[0]

    product_1 = no_numpy_scalar(a, b)
    product_2 = numpy_scalar(a, b)

    assert np.allclose(product_1, product_2)


# ---------
def no_numpy_mult(a, b):
    """
    A: list of "size" lists, each contains "size" floats --- первая матрица-аргумент
    B: list of "size" lists, each contains "size" floats --- вторая матрица-аргумент
    return C: list of "size" lists, each contains "size" floats --- матрица, являющаяся результатом умножения матриц a и b

    Функция принимает на вход две матрицы: A и B размерностью size x size
    Возвращает матрицу их произведения A * B = C

    Реализуйте умножение матриц без использования функций из пакета numpy
    """

    rows_count = len(a)
    cols_count = len(b[0])
    inner_count = len(b)

    result = [[0 for _ in range(cols_count)] for _ in range(rows_count)]

    for i in range(rows_count):
        for j in range(cols_count):
            for k in range(inner_count):
                result[i][j] += a[i][k] * b[k][j]

    return result


def numpy_mult(a, b):
    """
    A: np.array[size, size]              --- первая матрица-аргумент
    B: np.array[size, size]              --- вторая матрица-аргумент
    return C: np.array[size, size]       --- матрица, являющаяся результатом умножения матриц A и B

    Функция принимает на вход две матрицы: A и B размерностью size x size
    Возвращает матрицу их произведения A * B = C

    Реализуйте умножение матриц, используя функции из пакета numpy
    """
    return np.dot(a, b)


def test_mult():
    a = np.random.sample((100, 100))
    b = np.random.sample((100, 100))

    M1 = no_numpy_mult(a, b)
    M2 = numpy_mult(a, b)

    assert np.allclose(np.array(M1), M2)


# ---------
def calculate_loss_function(X, w, y):
    return LA.norm(X @ w - y) ** 2


def solve_linear_regression(X, y):
    return LA.inv(X.T @ X) @ X.T @ y


if __name__ == "__main__":
    test_scalars()
    test_mult()
    test_calculate_loss_function()
