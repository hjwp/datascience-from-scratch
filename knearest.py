from dataclasses import dataclass
from typing import List, Iterator
from collections import defaultdict, Counter
import csv
from vectors import Vector, distance

EXAMPLE_DATA = """
x, y,label

1,   100,   topleftie
2,   90,    topleftie
0.3, 94,    topleftie

9,   10,    bottomrightie
9.3, 5,     bottomrightie
8.9, 15,    bottomrightie

5,   50, middley
4.9, 54, middley
5.2, 60, middley
"""


@dataclass
class LabelledData:
    vector: Vector
    label: str


def load_data(raw: str) -> List[LabelledData]:
    csv_lines = [line.strip().replace(", ", ",") for line in raw.strip().splitlines()]
    return [
        LabelledData(
            vector=[float(row["x"]), float(row["y"])],
            label=row["label"].strip(),
        )
        for row in csv.DictReader(csv_lines)
    ]


def test_load_data():
    assert load_data(EXAMPLE_DATA) == [
        LabelledData([1, 100], "topleftie"),
        LabelledData([2, 90], "topleftie"),
        LabelledData([0.3, 94], "topleftie"),
        LabelledData([9, 10], "bottomrightie"),
        LabelledData([9.3, 5], "bottomrightie"),
        LabelledData([8.9, 15], "bottomrightie"),
        LabelledData([5, 50], "middley"),
        LabelledData([4.9, 54], "middley"),
        LabelledData([5.2, 60], "middley"),
    ]


def k_nearest(
    datapoint: Vector, k: int, population: List[LabelledData]
) -> List[LabelledData]:
    def distance_from_datapoint(d: LabelledData) -> float:
        return distance(d.vector, datapoint)

    # return sorted(population, key=lambda d: distance(d.vector, datapoint))[:k]
    return sorted(population, key=distance_from_datapoint)[:k]


def test_k_nearest():
    example_data = load_data(EXAMPLE_DATA)
    new_datapoint = [6, 20]
    assert k_nearest(new_datapoint, k=3, population=example_data) == [
        LabelledData([8.9, 15], "bottomrightie"),
        LabelledData([9, 10], "bottomrightie"),
        LabelledData([9.3, 5], "bottomrightie"),
    ]




def vote(results: List[LabelledData]) -> str:
    # go throuh results, find all the differetn labels, and count how many times we see each
    counts = Counter(d.label for d in results)
    return counts.most_common(1)[0][0]


def test_voting():
    assert (
        vote(
            [
                LabelledData([9.3, 5], "bottomrightie"),
                LabelledData([8.9, 15], "bottomrightie"),
                LabelledData([5, 50], "middley"),
            ]
        )
        == "bottomrightie"
    )
