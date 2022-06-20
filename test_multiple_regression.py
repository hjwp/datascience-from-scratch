from multiple_regression import predict, error, squared_error, least_squares_fit
from statistics import daily_minutes_good


def test_predict():
    # linear: y = bx + c
    # multiple:
    # y = b1.x1 + b2.x2 + b3.x3 + c
    # or, x = vector
    #     b = vector
    # first compenent of x is all 1s
    # first b is the constant factor

    x = [1, 2, 3]
    beta = [4, 4, 4]
    # so prediction = 4 + 8 + 12 = 24
    assert predict(x, beta) == 24


def test_error():
    x = [1, 2, 3]
    beta = [4, 4, 4]
    y = 30  # known value
    assert error(x, y, beta) == -6
    assert squared_error(x, y, beta) == 36


inputs = [
    [1.0, 49, 4, 0],
    [1, 41, 9, 0],
    [1, 40, 8, 0],
    [1, 25, 6, 0],
    [1, 21, 1, 0],
    [1, 21, 0, 0],
    [1, 19, 3, 0],
    [1, 19, 0, 0],
    [1, 18, 9, 0],
    [1, 18, 8, 0],
    [1, 16, 4, 0],
    [1, 15, 3, 0],
    [1, 15, 0, 0],
    [1, 15, 2, 0],
    [1, 15, 7, 0],
    [1, 14, 0, 0],
    [1, 14, 1, 0],
    [1, 13, 1, 0],
    [1, 13, 7, 0],
    [1, 13, 4, 0],
    [1, 13, 2, 0],
    [1, 12, 5, 0],
    [1, 12, 0, 0],
    [1, 11, 9, 0],
    [1, 10, 9, 0],
    [1, 10, 1, 0],
    [1, 10, 1, 0],
    [1, 10, 7, 0],
    [1, 10, 9, 0],
    [1, 10, 1, 0],
    [1, 10, 6, 0],
    [1, 10, 6, 0],
    [1, 10, 8, 0],
    [1, 10, 10, 0],
    [1, 10, 6, 0],
    [1, 10, 0, 0],
    [1, 10, 5, 0],
    [1, 10, 3, 0],
    [1, 10, 4, 0],
    [1, 9, 9, 0],
    [1, 9, 9, 0],
    [1, 9, 0, 0],
    [1, 9, 0, 0],
    [1, 9, 6, 0],
    [1, 9, 10, 0],
    [1, 9, 8, 0],
    [1, 9, 5, 0],
    [1, 9, 2, 0],
    [1, 9, 9, 0],
    [1, 9, 10, 0],
    [1, 9, 7, 0],
    [1, 9, 2, 0],
    [1, 9, 0, 0],
    [1, 9, 4, 0],
    [1, 9, 6, 0],
    [1, 9, 4, 0],
    [1, 9, 7, 0],
    [1, 8, 3, 0],
    [1, 8, 2, 0],
    [1, 8, 4, 0],
    [1, 8, 9, 0],
    [1, 8, 2, 0],
    [1, 8, 3, 0],
    [1, 8, 5, 0],
    [1, 8, 8, 0],
    [1, 8, 0, 0],
    [1, 8, 9, 0],
    [1, 8, 10, 0],
    [1, 8, 5, 0],
    [1, 8, 5, 0],
    [1, 7, 5, 0],
    [1, 7, 5, 0],
    [1, 7, 0, 0],
    [1, 7, 2, 0],
    [1, 7, 8, 0],
    [1, 7, 10, 0],
    [1, 7, 5, 0],
    [1, 7, 3, 0],
    [1, 7, 3, 0],
    [1, 7, 6, 0],
    [1, 7, 7, 0],
    [1, 7, 7, 0],
    [1, 7, 9, 0],
    [1, 7, 3, 0],
    [1, 7, 8, 0],
    [1, 6, 4, 0],
    [1, 6, 6, 0],
    [1, 6, 4, 0],
    [1, 6, 9, 0],
    [1, 6, 0, 0],
    [1, 6, 1, 0],
    [1, 6, 4, 0],
    [1, 6, 1, 0],
    [1, 6, 0, 0],
    [1, 6, 7, 0],
    [1, 6, 0, 0],
    [1, 6, 8, 0],
    [1, 6, 4, 0],
    [1, 6, 2, 1],
    [1, 6, 1, 1],
    [1, 6, 3, 1],
    [1, 6, 6, 1],
    [1, 6, 4, 1],
    [1, 6, 4, 1],
    [1, 6, 1, 1],
    [1, 6, 3, 1],
    [1, 6, 4, 1],
    [1, 5, 1, 1],
    [1, 5, 9, 1],
    [1, 5, 4, 1],
    [1, 5, 6, 1],
    [1, 5, 4, 1],
    [1, 5, 4, 1],
    [1, 5, 10, 1],
    [1, 5, 5, 1],
    [1, 5, 2, 1],
    [1, 5, 4, 1],
    [1, 5, 4, 1],
    [1, 5, 9, 1],
    [1, 5, 3, 1],
    [1, 5, 10, 1],
    [1, 5, 2, 1],
    [1, 5, 2, 1],
    [1, 5, 9, 1],
    [1, 4, 8, 1],
    [1, 4, 6, 1],
    [1, 4, 0, 1],
    [1, 4, 10, 1],
    [1, 4, 5, 1],
    [1, 4, 10, 1],
    [1, 4, 9, 1],
    [1, 4, 1, 1],
    [1, 4, 4, 1],
    [1, 4, 4, 1],
    [1, 4, 0, 1],
    [1, 4, 3, 1],
    [1, 4, 1, 1],
    [1, 4, 3, 1],
    [1, 4, 2, 1],
    [1, 4, 4, 1],
    [1, 4, 4, 1],
    [1, 4, 8, 1],
    [1, 4, 2, 1],
    [1, 4, 4, 1],
    [1, 3, 2, 1],
    [1, 3, 6, 1],
    [1, 3, 4, 1],
    [1, 3, 7, 1],
    [1, 3, 4, 1],
    [1, 3, 1, 1],
    [1, 3, 10, 1],
    [1, 3, 3, 1],
    [1, 3, 4, 1],
    [1, 3, 7, 1],
    [1, 3, 5, 1],
    [1, 3, 6, 1],
    [1, 3, 1, 1],
    [1, 3, 6, 1],
    [1, 3, 10, 1],
    [1, 3, 2, 1],
    [1, 3, 4, 1],
    [1, 3, 2, 1],
    [1, 3, 1, 1],
    [1, 3, 5, 1],
    [1, 2, 4, 1],
    [1, 2, 2, 1],
    [1, 2, 8, 1],
    [1, 2, 3, 1],
    [1, 2, 1, 1],
    [1, 2, 9, 1],
    [1, 2, 10, 1],
    [1, 2, 9, 1],
    [1, 2, 4, 1],
    [1, 2, 5, 1],
    [1, 2, 0, 1],
    [1, 2, 9, 1],
    [1, 2, 9, 1],
    [1, 2, 0, 1],
    [1, 2, 1, 1],
    [1, 2, 1, 1],
    [1, 2, 4, 1],
    [1, 1, 0, 1],
    [1, 1, 2, 1],
    [1, 1, 2, 1],
    [1, 1, 5, 1],
    [1, 1, 3, 1],
    [1, 1, 10, 1],
    [1, 1, 6, 1],
    [1, 1, 0, 1],
    [1, 1, 8, 1],
    [1, 1, 6, 1],
    [1, 1, 4, 1],
    [1, 1, 9, 1],
    [1, 1, 9, 1],
    [1, 1, 4, 1],
    [1, 1, 2, 1],
    [1, 1, 9, 1],
    [1, 1, 0, 1],
    [1, 1, 8, 1],
    [1, 1, 6, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 5, 1],
]


def test_least_squares_fit():
    learning_rate = 0.001
    beta = least_squares_fit(inputs, daily_minutes_good, learning_rate, 5000, 25)
    print("beta", beta)
    assert 30.50 < beta[0] < 30.70  # constant
    assert 0.96 < beta[1] < 1.00  # num friends
    assert -1.89 < beta[2] < -1.85  # work hours per day
    assert 0.91 < beta[3] < 0.94  # has PhD
