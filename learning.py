"""
model = SomeKindOfModel()
x_train, x_test, y_train, y_test = train_test_split(xs, ys, 0.33)
model.train(x_train, y_train)
performance = model.test(x_test, y_test)
"""
import pytest
import random
from typing import List, Tuple


def _split_one(
    data: List[float], split_ratio: float
) -> Tuple[List[float], List[float]]:
    num_train = round(len(data) * split_ratio)
    all_indices = list(range(len(data)))
    training_indices = random.sample(all_indices, k=num_train)
    train = [data[i] for i in training_indices]
    test = [x for x in data if x not in train]
    return train, test


def train_test_split(
    xs: List[float],
    ys: List[float],
    split_ratio: float,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    assert len(xs) == len(ys)
    x_train, x_test = _split_one(xs, split_ratio)
    y_train, y_test = _split_one(ys, split_ratio)
    return x_train, x_test, y_train, y_test


def test_train_split():
    xs = list(range(200))
    ys = list(range(50, 250))
    x_train, x_test, y_train, y_test = train_test_split(xs, ys, 0.33)

    # checks we didnt lose any datapoints
    assert sorted(x_train + x_test) == xs
    assert sorted(y_train + y_test) == ys

    # check it split according to ratios
    assert len(x_train) / len(xs) == pytest.approx(0.33, abs=1e-2)
    assert len(x_test) / len(xs) == pytest.approx(0.67, abs=1e-2)
    assert len(y_train) / len(ys) == pytest.approx(0.33, abs=1e-2)
    assert len(y_test) / len(ys) == pytest.approx(0.67, abs=1e-2)

    # (imperfect) sanity-check that numbers are randomised
    assert x_train != xs[: len(x_train)]
