from typing import Callable
import pytest
from vectors import Vector, add, multiply

MultivariateFunction = Callable[[Vector], float]

def sum_of_squares(v: Vector) -> float:
    return sum([i**2 for i in v])


def difference_quotient(
    f: Callable[[float], float],
    x: float,
    epsilon: float,
) -> float:
    return (f(x + epsilon) - f(x)) / epsilon


def partial_difference_quotient(
    f: MultivariateFunction,
    parameter_no: int,
    input_vector: Vector,
    epsilon: float,
) -> float:
    """
    returns the "partial gradient" at the point along f with value=input_vector
    with respect to the nth parameter where n=parameter_no
    """
    incremented_vector = input_vector.copy()
    incremented_vector[parameter_no] = incremented_vector[parameter_no] + epsilon
    return (f(incremented_vector) - f(input_vector)) / epsilon


def estimate_gradient(
    f: MultivariateFunction,
    input_vector: Vector,
    epsilon: float,
) -> Vector:
    """
    estimate the gradeint at the given point
    """
    all_parameter_numbers = range(len(input_vector))
    return [
        partial_difference_quotient(
            f, parameter_no, input_vector=input_vector, epsilon=epsilon
        )
        for parameter_no in all_parameter_numbers
    ]


def gradient_descent(
    f: MultivariateFunction,
    starting_vector: Vector,
) -> Vector:
    new_position = starting_vector
    standard_step_scaling_factor = -0.1
    for epoch in range(1000):
        print(f"Epoch {epoch}, position {new_position}")
        gradient = estimate_gradient(f, new_position, epsilon=0.001)
        step = multiply(gradient, standard_step_scaling_factor)
        new_position = add(new_position, step)
    return new_position


def test_gradient_descent():
    # gradient descent on f(x^2 + y^2) should equal [0, 0]
    def sum_of_two_squares(v: Vector) -> float:
        x, y = v
        return x**2 + y**2

    expected = [0, 0]
    assert gradient_descent(f=sum_of_two_squares, starting_vector=[3, 2]) == pytest.approx(expected, abs=0.001)


def test_estimate_gradient():
    # gradient of a multidemnsional function, ie one that takes a vector,
    # should itself be a vector

    # partial gradient of f(x^2 + y^2)  should be 2x wrt x and 2y wrt y
    def sum_of_two_squares(v: Vector) -> float:
        x, y = v
        return x**2 + y**2

    # eg gradient of f(x^2 + y^2) at x should be 2x = 6, and at y should be 4
    x = 3
    y = 2
    expected = [6, 4]

    assert estimate_gradient(
        f=sum_of_two_squares, input_vector=[x, y], epsilon=0.001
    ) == pytest.approx(expected, abs=0.001)


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
