import csv
from dataclasses import dataclass
from typing import Any, List, Dict, NamedTuple

NUMBER_COLUMNS = ["Open", "Close", "High", "Low", "Adj Close", "Volume"]


@dataclass(frozen=True)
class Stock:
    name: str
    opening_price: float
    closing_price: float


def fix_row(raw_row: Dict[str, str]) -> Dict[str, Any]:
    fixed = {}  # type: Dict[str, Any]
    for column_name, string_value in raw_row.items():
        if column_name in NUMBER_COLUMNS:
            fixed[column_name] = float(string_value)
        else:
            fixed[column_name] = string_value
    return fixed


def load_csv_dicts() -> List[Dict[str, Any]]:
    with open("stocks.csv") as f:
        reader = csv.DictReader(f.readlines())
    return [fix_row(raw_row) for raw_row in list(reader)]


def load_csv_objects() -> List:
    row_dicts = load_csv_dicts()

    results = []
    for raw_row in row_dicts:
        stock = Stock(
            name=raw_row["Symbol"],
            opening_price=raw_row["Open"],
            closing_price=raw_row["Close"],
        )
        results.append(stock)
    return results


def main():
    all_stocks = load_csv()
    print(all_stocks[0])
    print(all_stocks[1])


def test_load_csv_dicts_parses_numbers():
    all_stocks = load_csv_dicts()
    assert all_stocks[0]["Open"] == 0.513393
    assert all_stocks[0]["Close"] == 0.513393
    assert all_stocks[0]["High"] == 0.515625
    assert all_stocks[0]["Low"] == 0.513393
    assert all_stocks[0]["Adj Close"] == 0.023106
    assert all_stocks[0]["Volume"] == 117258400

    for column_name in NUMBER_COLUMNS:
        assert all_stocks[0][column_name] == float(all_stocks[0][column_name])


def test_load_csv_objects_parses_numbers():
    all_stocks = load_csv_objects()
    example_stock = all_stocks[0]
    assert example_stock.name == "AAPL"
    assert example_stock.opening_price == 0.513393


def test_stock():
    mystock = Stock("stocky", opening_price=100, closing_price=200)
    assert mystock.name == "stocky"
    assert mystock.opening_price == 100
    assert mystock.closing_price == 200


if __name__ == "__main__":
    print("-" * 80)
    main()
