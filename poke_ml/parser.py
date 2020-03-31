import os
import subprocess

from bs4 import BeautifulSoup
from dataclasses import dataclass
from typing import Sequence, Optional, Iterator


@dataclass
class TableRow:
    national_index: str
    image_file: str
    pokemon_name: str
    type_1: str
    type_2: Optional[str]

    def __iter__(self) -> Iterator:
        return iter(
            [
                self.national_index,
                self.image_file,
                self.pokemon_name,
                self.type_1,
                self.type_2,
            ]
        )


@dataclass
class PokemonEntry:
    national_index: str
    image_file: str
    pokemon_name: str
    type_1: str
    type_2: Optional[str]
    shape: str

    def __repr__(self) -> str:
        unpacked = [
            self.national_index,
            self.image_file,
            self.pokemon_name,
            self.type_1,
            "" if self.type_2 is None else self.type_2,
            self.shape,
        ]
        return ",".join(unpacked)


TitleNode = BeautifulSoup
TableNode = BeautifulSoup
RowNode = BeautifulSoup


def parse_page(page_html: BeautifulSoup) -> Sequence[PokemonEntry]:
    """Returns all relevant information on the page"""
    body_html = page_html.find("body")
    title_nodes = extract_table_titles(body_html)
    titles = [parse_title(title) for title in title_nodes]

    table_nodes = extract_tables(body_html)
    tables = [
        parse_table(table) for table in table_nodes if is_poketable(table)
    ]

    entries = []
    for table, title in zip(tables, titles):
        for row in table:
            entry = PokemonEntry(*row, shape=title)
            entries.append(entry)

    return entries


def extract_table_titles(body_html: BeautifulSoup) -> Sequence[TitleNode]:
    """Extracts the table titles from the page"""
    return body_html.select("h3 > span.mw-headline")


def extract_tables(body_html: BeautifulSoup) -> Sequence[TableNode]:
    """Extracts the tables from the page"""
    return body_html.select("table.roundy")


def parse_title(title_html: TitleNode) -> str:
    """Extracts the title from the title html"""
    text = _extract_text(title_html)
    text = text.replace("é", "e")
    text = text.replace(",", "")
    return text.lower()


def is_poketable(table_html: TableNode) -> bool:
    header_row = table_html.find("tr")
    header_columns = header_row.find_all(["td", "th"])
    return _extract_text(header_columns[0]) == "ndex"


def parse_table(table_html: TableNode) -> Sequence[TableRow]:
    """Extracts the contents from the table html"""
    rows = table_html.find_all("tr")
    # header = rows[0]
    contents = rows[1:]
    return [parse_row(row) for row in contents]


def parse_row(row_html: RowNode) -> TableRow:
    items = row_html.find_all(["td", "th"])
    national_index = _extract_text(items[0])[1:]
    image_file = items[1].find("img")["src"].split("/")[-1]
    name = _extract_text(items[2])
    n_type_columns = int(items[3]["colspan"])
    type_1 = _extract_text(items[3])
    if n_type_columns == 2:
        type_2 = None
    else:
        type_2 = _extract_text(items[4])

    return TableRow(national_index, image_file, name, type_1, type_2)


def _extract_text(node: BeautifulSoup) -> str:
    return node.text.strip().lower()


def move_images(folder: str, file_names: Sequence[str]) -> None:
    """
    Provide full path of folder
    """
    if not os.path.exists(folder):
        os.mkdir(folder)

    raw_folder = "./data/raw/pokemon_by_shape_files"
    for file in file_names:
        subprocess.call(f"cp {raw_folder}/{file} {folder}", shell=True)


def main():
    """
    Loads the raw html, parses it and creates a csv file from the output.

    Run from root
    """
    data_file = "./data/raw/pokemon_by_shape.htm"
    save_file = "./data/tables/pokemon_by_shape.csv"

    with open(data_file) as f:
        raw_html = f.read()

    soup = BeautifulSoup(raw_html, "html.parser")
    entries = parse_page(soup)
    print(f"Extracted {len(entries)} entries. Saving at {save_file!r}")
    with open(save_file, "w") as f:
        columns = list(PokemonEntry.__dataclass_fields__.keys())
        f.write(",".join(columns) + "\n")
        for entry in entries:
            f.write(str(entry) + "\n")

    image_folder = "./data/images"
    print(f"Moving images to {image_folder!r}")
    files = [e.image_file for e in entries]
    move_images(image_folder, files)


if __name__ == "__main__":
    main()
