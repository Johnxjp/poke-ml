import pytest
from bs4 import BeautifulSoup

from pokegan.webparser import is_poketable, extract_pokemon_tables


def soupify(html: str):
    return BeautifulSoup(html, "html.parser")


def html_table(header_cells: list):
    html = "<table>"
    for hcell in header_cells:
        html += f"<th> {hcell} </th>"

    html += "</table>"
    return html


@pytest.mark.parametrize("html_contents, expected_value", [
    (soupify("<tag>"), False),
    (soupify(html_table([])), False),
    (soupify(html_table(["Ndex", "MS"])), False),
    (soupify(html_table(["Ndex", "MS", "Type"])), True),
    (soupify(html_table(["Ndex", "MS", "Type", "Extra"])), True),
])
def test_is_poketable(html_contents, expected_value):
    assert is_poketable(html_contents) == expected_value


@pytest.mark.parametrize("html_contents, expected_value", [
    (
        soupify(
            "".join([
                html_table(["Ndex", "MS", "Type"]),
                html_table(["Ndex", "MS", "Type"])])), 2
    ),
    (
        soupify(
            "".join([
                html_table(["Ndex", "MS"]),
                html_table(["Ndex", "MS", "Type"])])), 1
    ),
    (
        soupify(
            "".join([
                html_table(["Ndex", "MS"]),
                html_table([])])), 0
    ),
])
def test_extract_poketables(html_contents, expected_value):
    assert len(extract_pokemon_tables(html_contents)) == expected_value
