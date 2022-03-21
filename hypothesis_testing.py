import random
from typing import List


def run_experiment() -> List[str]:
    return [random.choice(["heads", "tails"]) for _ in range(1000)]


def try_until_we_find_a_biased_coin(num_attempts):
    for attempt_no in range(num_attempts):
        tosses = run_experiment()
        num_heads = len([flip for flip in tosses if flip == "heads"])
        lower_bound = 469   # TODO: calculate this
        upper_bound = 531   # TODO: calculate this
        if num_heads < lower_bound or num_heads > upper_bound:
            print(f"Found a biased coin after {attempt_no} attempts!  it had {num_heads} heads, woa!")
            return num_heads, attempt_no
    return None   # all the experiments looked like fair coins


def run_lots_of_experiments(n):
    """
    run coin toss experiment n times
    and return a list containing the string "fair" or "unfair" for each one
    """
    fairs_or_unfairs = []

    for attempt_no in range(n):
        tosses = run_experiment()
        num_heads = len([flip for flip in tosses if flip == "heads"])
        lower_bound = 469   # TODO: calculate this
        upper_bound = 531   # TODO: calculate this
        if num_heads < lower_bound or num_heads > upper_bound:
            # it was an unfair coin
            fairs_or_unfairs.append("unfair")
        else:
            fairs_or_unfairs.append("fair")
    return fairs_or_unfairs


def test_p_hacking2():
    random.seed(0)  # fix the random number generator
    fairs_and_unfairs = run_lots_of_experiments(100)
    print("fairs_and_unfairs", fairs_and_unfairs)
    unfair_results = [result for result in fairs_and_unfairs if result == "unfair"]
    num_unfair_coins = len(unfair_results)
    assert num_unfair_coins == 3


def DONTtest_p_hacking1():
    random.seed(0)  # fix the random number generator
    result = try_until_we_find_a_biased_coin(1)
    assert result is None
    result, attempt_no = try_until_we_find_a_biased_coin(50)
    assert result > 531 or result < 469
    assert attempt_no == 32

