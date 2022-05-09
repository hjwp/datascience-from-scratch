import tqdm
from matplotlib import pyplot as plt
from pca_data import pca_data
from typing import List, Tuple
from vectors import Vector, subtract, dot
from gradient_descent import gradient_step
from stats import mean
import math as maths


def main():
    xs, ys = get_xs_and_ys(de_mean(pca_data))
    plt.scatter(xs, ys)
    plt.show()


def columns_of(data: List[Vector]) -> List[Vector]:
    return [[row[col_no] for row in data] for col_no in range(len(data[0]))]


def vector_mean(data: List[Vector]) -> Vector:
    return [mean(column) for column in columns_of(data)]


def de_mean(data: List[Vector]) -> List[Vector]:
    """
    return data centered around the mean for each column
    """
    means_for_each_column = vector_mean(data)
    return [subtract(row, means_for_each_column) for row in data]


def test_de_mean():
    data = [
        [1, -10],
        [2, -8],
        [3, -3],
    ]
    assert de_mean(data) == [
        [-1, -3],
        [0, -1],
        [1, 4],
    ]
    data = [
        [1, -10, 5],
        [2, -8, 4],
        [3, -3, 3],
    ]
    assert de_mean(data) == [
        [-1, -3, 1],
        [0, -1, 0],
        [1, 4, -1],
    ]


def get_xs_and_ys(data: List[List[float]]) -> Tuple[List[float], List[float]]:
    xs = [row[0] for row in data]
    ys = [row[1] for row in data]
    return xs, ys


def test_getting_xs_and_ys():
    xs, ys = get_xs_and_ys(pca_data)
    assert xs[0] == 20.9666776351559
    assert ys[0] == -13.1138080189357


def first_principal_component(
    data: List[Vector], n: int = 100, step_size: float = 0.1
) -> Vector:
    # Start with a random guess
    guess = [1.0 for _ in data[0]]

    for _ in range(1000):
        dv = directional_variance(data, guess)
        gradient = directional_variance_gradient(data, guess)
        input(f"Current guess: {guess}, variance: {dv:.2f}, gradient: {gradient}")
        guess = gradient_step(guess, gradient, step_size)
        # t.set_description(f"dv: {dv:.3f}")

    return direction(guess)


def directional_variance(data: List[Vector], direction_vector: Vector) -> float:
    # sum of the squares of the dot products of each element with the direction vector
    dir = direction(direction_vector)
    return sum(dot(v, dir) ** 2 for v in data)


def test_directional_variance():
    data = [
        [0, 0],
        [1, 0],
        [2, 0],
    ]
    y_axis_vector = [0, 1]
    x_axis_vector = [1, 0]
    assert directional_variance(data, x_axis_vector) == 5
    assert directional_variance(data, y_axis_vector) == 0


def direction(v: Vector) -> Vector:
    """
    return a unit vector in the direction of v
    """
    magnitude = maths.sqrt(sum(e**2 for e in v))
    return [e / magnitude for e in v]


def test_direction():
    assert direction([3, 4]) == [3 / 5, 4 / 5]


def directional_variance_gradient(data: List[Vector], w: Vector) -> Vector:
    """
    The gradient of directional variance with respect to w
    """
    w_dir = direction(w)
    return [sum(2 * dot(v, w_dir) * v[i] for v in data) for i in range(len(w))]


if __name__ == "__main__":
    main()
