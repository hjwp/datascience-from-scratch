from typing import List
from bs4 import BeautifulSoup
import requests


def main() -> None:
    # find all the congresspeoples personal web pages
    # in there, find press releases page
    # in there, find all the title s of press releases
    # find the ones that mention "Data"
    url = "https://www.house.gov/representatives"
    text = requests.get(url).text
    soup = BeautifulSoup(text, "html5lib")
    urls = extract_congressperson_urls(soup)
    print("\n".join(urls))


def extract_congressperson_urls(soup: BeautifulSoup) -> List[str]:
    all_links = soup('a')
    proper_links = [link for link in all_links if link.has_attr('href')]  # <-- a list comprehension

    # find all the actual urls (hrefs) of links that look like they
    # are congressperson links
    congresspeople_urls = [
        # expression (probably based on "item")
        link
        # for item some list
        for link in proper_links
        # if (optionally) some condition
        if is_congressperson_url(link)  # TODO: something is going wrong here.
    ]
    return congresspeople_urls


def is_congressperson_url(url: str) -> bool:
    print('url', url)
    return url.startswith('https') and url.endswith('.house.gov')


def test_finding_the_right_urls():
    assert is_congressperson_url('https://sires.house.gov') is True
    assert is_congressperson_url('/watch-houselive') is False
    assert is_congressperson_url('https://www.google.com') is False

def test_finding_the_right_urls():
    assert is_congressperson_url('https://sires.house.gov') is True
    assert is_congressperson_url('/watch-houselive') is False
    assert is_congressperson_url('https://www.google.com') is False

if __name__ == "__main__":
    main()
