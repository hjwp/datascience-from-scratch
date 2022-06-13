from stats import correlation, standard_deviation, mean

def least_squares_fit(x, y):
    """
    return coefficients a, b such that y = bx + a
    """
    beta = correlation(x, y) * standard_deviation(y) / standard_deviation(x)
    alpha = mean(y) - beta * mean(x)
    return alpha, beta
