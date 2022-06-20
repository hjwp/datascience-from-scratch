from typing import List, Tuple, Iterable
from vectors import dot, Vector, vector_mean
from gradient_descent import gradient_step


def predict(x: Vector, beta: Vector) -> float:
    return dot(x, beta)


def error(x: Vector, y: float, beta: Vector) -> float:
    return predict(x, beta) - y


def squared_error(x: Vector, y: float, beta: Vector) -> float:
    return error(x, y, beta) ** 2


def sqerror_gradient(x: Vector, y: float, beta: Vector) -> Vector:
    err = error(x, y, beta)
    return [2 * err * x_i for x_i in x]


def mean_squerror_gradient(xs: List[Vector], ys: List[float], beta: Vector) -> Vector:
    # return average squared error gradient
    return vector_mean([sqerror_gradient(x, y, beta) for x, y in zip(xs, ys)])


import random
def break_into_batches(
    xs: List[Vector], ys: List[Vector], batch_size: int
) -> Iterable[Tuple[List[Vector], List[Vector]]]:
    zipped_lists = list(zip(xs, ys))
    random.shuffle(zipped_lists)
    for start in range(0, len(xs), batch_size):
        batch = zipped_lists[start: start+batch_size]
        yield [b[0] for b in batch], [b[1] for b in batch]


def least_squares_fit(
    xs: List[Vector],
    ys: List[float],
    learning_rate: float = 0.001,
    num_steps: int = 1000,
    batch_size: int = 1,
) -> Vector:
    guess = [0.9 for _ in xs[0]]
    for step in range(num_steps):
        for x_batch, y_batch in break_into_batches(xs, ys, batch_size):
            gradient = mean_squerror_gradient(x_batch, y_batch, beta=guess)
            guess = gradient_step(
                guess,
                gradient,
                step_size=-learning_rate,
            )
    # this also works but converges slower
    # gradient = mean_squerror_gradient(xs, ys, beta=guess)
    # guess = gradient_step(
    #     guess, gradient, step_size=-learning_rate,
    # )
    return guess
