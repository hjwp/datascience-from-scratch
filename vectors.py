from typing import List, Union


x = 3  # type: int
y = "hello"  # type: str
things = [1, 2, 3]  # type: list


Vector = List[Union[float, int]]

assert 1 + 1 == 2
assert True


def add(v1: Vector, v2: Vector) -> Vector:
    assert len(v1) == len(v2), "vectors not same length"
    return [v1[i] + v2[i] for i in range(len(v1))]


def subtract(v1: Vector, v2: Vector) -> Vector:
    assert len(v1) == len(v2), "vectors not same length"
    return [v1[i] - v2[i] for i in range(len(v1))]


def sum_of_ith_elements(vectors: List[Vector], i) -> float:
    return sum([v[i] for v in vectors])


def vector_sum(vectors: List[Vector]) -> Vector:
    assert len(vectors) > 0, "need at least one vector"
    vector_length = len(vectors[0])

    return [
        sum_of_ith_elements(vectors, i) 
        for i in range(vector_length)
    ]

def multiply(v: Vector, scalar: float) -> Vector:
    return [i * scalar for i in v]

def test_add():
    v1 = [1, 2]
    v2 = [2, 3]
    assert add(v1, v2) == [3, 5]
    assert add([1, 2, 3], [4, 5, 7]) == [5, 7, 10]


def test_subtract():
    assert subtract([4, 5, 7], [1, 2, 3]) == [3, 3, 4]



def test_sum():
    v1 = [4, 5, 7]
    v2 = [1, 2, 3]
    v3 = [10, 11, 12]
    vectors = [v1, v2, v3]
    assert vector_sum(vectors) == [15, 18, 22]
    vectors = [v1, v2]
    assert vector_sum(vectors) == [5, 7, 10]

