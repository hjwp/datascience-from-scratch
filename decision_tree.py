from collections import Counter
from typing import Any, Dict, List
import math


def class_probabilities(labels: List[Any]) -> List[float]:
    """
    if we've applied a set of labels to some data,
    count up all the labels of each type and return
    their ratio of the total (aka probability).
    for our purposes, we don't care about which label has which probability.
    """
    return [count / len(labels) for count in Counter(labels).values()]


def entropy(class_probabilities: List[float]) -> float:
    """
    Entropy = -p . log2 (p) for each class probability
    """
    return sum([-p * math.log(p, 2) for p in class_probabilities])


def data_entropy(labels: List[Any]) -> float:
    """
    returns the entropy of the class probabilities of each label
    """
    return entropy(class_probabilities(labels))


def partition_entropy(subsets: List[List[Any]]) -> float:
    """
    partitioning into lots of small subsets = good, ie low entropy
    """


from collections import defaultdict

from typing import TypeVar
T = TypeVar("T")

def partition_by(things: List[T], attribute_name: str) -> Dict[Any, List[T]]:
    """
    returns a dictionary of attribute values to lists of the inputs which match that value
    """
    partitioned_things = defaultdict(list)
    for thing in things:
        value = getattr(thing, attribute_name)
        partitioned_things[value].append(thing)

    return partitioned_things

