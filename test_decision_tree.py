from decision_tree import (
    class_probabilities,
    classify,
    entropy,
    data_entropy,
    partition_by,
)


def test_class_probabilities():
    # q = more than 5 legs?
    labels = [True, True, False, True, False]
    assert sorted(class_probabilities(labels)) == [2 / 5, 3 / 5]

    labels = ["small", "medium", "large", "large", "small", "large"]
    assert sorted(class_probabilities(labels)) == [1 / 6, 2 / 6, 3 / 6]


def test_entropy():
    assert entropy([1.0]) == 0
    assert entropy([0.5, 0.5]) == 1
    assert 0.81 < entropy([0.25, 0.75]) < 0.82


def test_data_entropy():
    assert data_entropy(["a"]) == 0
    assert data_entropy([True, False]) == 1
    assert data_entropy([3, 4, 4, 4]) == entropy([0.25, 0.75])


from dataclasses import dataclass
from typing import Optional


@dataclass
class Candidate:
    level: str
    lang: str
    tweets: bool
    phd: bool
    did_well: Optional[bool] = None  # allow unlabeled data


# fmt: off
                  
candidates = [
    #          level     lang     tweets phd    did_well
    Candidate('Senior', 'Java',   False, False, False),
    Candidate('Senior', 'Java',   False, True,  False),
    Candidate('Mid',    'Python', False, False, True),
    Candidate('Junior', 'Python', False, False, True),
    Candidate('Junior', 'R',      True,  False, True),
    Candidate('Junior', 'R',      True,  True,  False),
    Candidate('Mid',    'R',      True,  True,  True),
    Candidate('Senior', 'Python', False, False, False),
    Candidate('Senior', 'R',      True,  False, True),
    Candidate('Junior', 'Python', True,  False, True),
    Candidate('Senior', 'Python', True,  True,  True),
    Candidate('Mid',    'Python', False, True,  True),
    Candidate('Mid',    'Java',   True,  False, True),
    Candidate('Junior', 'Python', False, True,  False),
]


def test_partition_by():
    assert partition_by(candidates, "level") == {
        "Junior": [
            Candidate('Junior', 'Python', False, False, True),
            Candidate('Junior', 'R',      True,  False, True),
            Candidate('Junior', 'R',      True,  True,  False),
            Candidate('Junior', 'Python', True,  False, True),
            Candidate('Junior', 'Python', False, True,  False),
        ],
        "Mid": [
            Candidate('Mid',    'Python', False, False, True),
            Candidate('Mid',    'R',      True,  True,  True),
            Candidate('Mid',    'Python', False, True,  True),
            Candidate('Mid',    'Java',   True,  False, True),
        ],
        "Senior": [
            Candidate('Senior', 'Java',   False, False, False),
            Candidate('Senior', 'Java',   False, True,  False),
            Candidate('Senior', 'Python', False, False, False),
            Candidate('Senior', 'R',      True,  False, True),
            Candidate('Senior', 'Python', True,  True,  True),
        ],
    }

# fmt: on

tree = {
    "level": {
        "Mid": True,
        "Senior": {
            "tweets": {
                True: True,
                False: False,
            },
        },
        "Junior": {
            "phd": {
                False: True,
                True: False,
            }
        },
    },
}


def test_classify_against_existing_data():
    for candidate in candidates:
        assert classify(tree, candidate) == candidate.did_well


def test_classify_against_new_inputs():
    new_candidate = Candidate("Senior", "R", tweets=False, phd=True, did_well=None)
    assert classify(tree, new_candidate) == False
    new_candidate = Candidate("Junior", "R", tweets=False, phd=False, did_well=None)
    assert classify(tree, new_candidate) == True
