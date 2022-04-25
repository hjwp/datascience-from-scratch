# from matplotlib import pyplot as plt
from pca_data import pca_data
from typing import List, Tuple
from vectors import Vector, subtract
from stats import mean


def main():
    xs, ys = get_xs_and_ys(pca_data)
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


if __name__ == "__main__":
    main()
