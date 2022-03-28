from typing import Callable
import pytest
from vectors import Vector


def sum_of_squares(v: Vector) -> float:
    return sum([i**2 for i in v])


def difference_quotient(
    f: Callable[[float], float],
    x: float,
    epsilon: float,
) -> float:
    return (f(x + epsilon) - f(x)) / epsilon


def partial_difference_quotient(
    f: Callable[[Vector], float],
    parameter_no: int,
    input_vector: Vector,
    epsilon: float,
) -> float:
    """
    returns the "gradient" at the point along f with value=input_vector
    with respect to the nth parameter where n=parameter_no
    """
    incremented_vector = input_vector.copy()
    incremented_vector[parameter_no] = incremented_vector[parameter_no] + epsilon
    return (f(incremented_vector) - f(input_vector)) / epsilon


def test_partial_difference_quotient():
    # eg gradient of f(x) = 2x should be 2
    def two_x(v: Vector) -> float:
        x = v[0]
        return 2 * x

    assert partial_difference_quotient(
        two_x, parameter_no=0, input_vector=[1], epsilon=0.001
    ) == pytest.approx(2)

    # partial gradient of f(x^2 + y^2)  should be 2x wrt x and 2y wrt y
    def sum_of_two_squares(v: Vector) -> float:
        x, y = v
        return x**2 + y**2

    # eg if x = 3, gradient of f(x^2 + y^2) at x should be 2x = 6
    x = 3
    y = 2
    expected = 6
    assert partial_difference_quotient(
        sum_of_two_squares,
        parameter_no=0,
        input_vector=[x, y],
        epsilon=0.001,
    ) == pytest.approx(expected, abs=0.001)


def test_sum_of_squares():
    assert sum_of_squares([2, 2]) == 8
    assert sum_of_squares([3, 3, 3]) == 27


def test_difference_quotient():
    # gradient of f(x) = 2x should be 2
    def f(x: float) -> float:
        return 2 * x

    assert difference_quotient(f, x=1, epsilon=0.001) == pytest.approx(2)


def test_sum_of_squares():
    assert sum_of_squares([2, 2]) == 8
    assert sum_of_squares([3, 3, 3]) == 27