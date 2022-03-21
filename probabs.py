import random

BOY = "boy"
GIRL = "girl"

def generate_family():
    first_child = random.choice([BOY, GIRL])
    second_child = random.choice([BOY, GIRL])
    return first_child, second_child


def generate_lots_of_families_and_analyse_them():
    n = 9999
    count_two_girls = 0
    count_at_least_one_girl = 0
    for _ in range(n):
        first, second = generate_family()
        if first == GIRL and second ==  GIRL:
            count_two_girls += 1

        if first == GIRL or second ==  GIRL:
            count_at_least_one_girl += 1

    print("proportion of two girls", count_two_girls / n)
    print("proportion of at least one girl", count_at_least_one_girl / n)
    print("proportion of two girls given at least one girl", (count_two_girls / n) / (count_at_least_one_girl / n))



def test_generate_family():
    first, second = generate_family()
    assert first in {BOY, GIRL}
    assert second in {BOY, GIRL}


if __name__ == "__main__":
    generate_lots_of_families_and_analyse_them()
