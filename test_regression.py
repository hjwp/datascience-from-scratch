import pytest
from regression import least_squares_fit


def test_basic_linear_regression():
    x = [i for i in range(-100, 110, 10)]
    print(x)
    y = [3 * i - 5 for i in x]

    # Should find that y = 3x - 5
    assert least_squares_fit(x, y) == pytest.approx(
        (-5, 3),
        abs=1e-6,
    )

    # Should find that y = 17x + 8
    y = [17 * i + 8 for i in x]
    assert least_squares_fit(x, y) == pytest.approx(
        (8, 17),
        abs=1e-6,
    )
