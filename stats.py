import math
import numpy as np
from typing import List


def mean(series: List[float]) -> float:
    return sum(series) / len(series)


def variance(series: List[float]) -> float:
    m = mean(series)
    n = len(series)
    return sum( 
        (item - m) ** 2 for item in series 
    ) / n


def standard_deviation(series: List[float]) -> float:
    return math.sqrt(variance(series))


def covariance(series1: List[float], series2: List[float]) -> float:
    assert len(series1) == len(series2)
    mean1 = mean(series1)
    mean2 = mean(series2)
    n = len(series1)
    co_deviations = [
        (value1 - mean1) * (value2 - mean2)
        for value1, value2 
        in zip(series1, series2)
    ]
    return sum(co_deviations) / n


def correlation(series1: List[float], series2: List[float]) -> float:
    assert len(series1) == len(series2)
    stdev1 = standard_deviation(series1)
    stdev2 = standard_deviation(series2)
    return covariance(series1, series2) / (stdev1 * stdev2)


# --- tests -----


def test_mean():
    assert mean([1, 2, 3, 4, 5]) == 3
    assert mean([1, 2, 3]) == 2


def test_standard_deviation():
    assert standard_deviation([1, 2, 3, 4, 5]) == np.std([1, 2, 3, 4, 5])


def test_correlation():
    assert correlation([1, 2, 3], [1, 2, 3]) == 1
    assert correlation([3, 2, 1], [1, 2, 3]) == -1
    assert correlation([1, 2, 3], [2, 4, 6]) == 1
