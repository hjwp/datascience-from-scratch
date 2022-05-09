import numpy as np

def test_matrix_addition():
    matrix1 = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])
    matrix2 = np.array([
        [10, 11, 12],
        [13, 14, 15],
    ])

    result = matrix1 + matrix2

    print("result", result)

    expected = np.array([
        [11, 13, 15],
        [17, 19, 21],
    ])
    print("expected", expected)

    assert np.equal(result, expected).all()
