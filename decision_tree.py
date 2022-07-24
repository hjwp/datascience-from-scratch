from collections import Counter
from typing import Any, Dict, List, Union
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


def is_leaf_node(node: Union[Dict, Any]) -> bool:
    if type(node) == dict:
        return False
    else:
        return True



def classify(tree: Dict[str, Any], input: Any) -> Any:
    # look at the next node in the tree
    # assume it's a decision node for now
    questions = tree.keys()
    [question] = questions
    answer_nodes = tree[question]

    # ask the question of our input,
    # ie get the input's attribute value for that question
    print('question', question)
    answer = getattr(input, question)
    print('answer', answer)
    print('answer nodes', answer_nodes)

    # go to the next bit in the tree by looking at the answer nodes
    next_node = answer_nodes[answer]

    # if the next bit is another question, then we go down the tree
    # recursively,
    # but if it's a leaf node, just return the value
    print('next node', next_node)
    if is_leaf_node(next_node):
        return next_node

    else:
        return classify(tree=next_node, input=input)
