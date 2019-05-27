# Web parser
# Source https://bulbapedia.bulbagarden.net/wiki/List_of_Pok%C3%A9mon_by_National_Pok%C3%A9dex_number  # noqa

# Desired output: Table of pokemon name, type 1, type 2, image_name

from bs4 import BeautifulSoup


def is_poketable(table_html: BeautifulSoup):
    """
    The pokemon tables all contain the unique column names  "Ndex", "MS",
    and "Type". Use these to extract the right tables from the page
    """
    header_cells = table_html.find_all('th')
    # Get tag string
    header_cells = [
        cell.string.strip() for cell in header_cells if cell.string is not None
    ]

    required_header_cells = ["Ndex", "MS", "Type"]
    if len(header_cells) < len(required_header_cells):
        return False

    if all([req_cell in header_cells for req_cell in required_header_cells]):
        return True

    return False


def extract_pokemon_tables(content: BeautifulSoup):
    """
    Returns the html for all the pokemon tables
    """
    all_tables = content.find_all('table')
    poketables = [table for table in all_tables if is_poketable(table)]
    return poketables


def parse_table():
    pass


if __name__ == "__main__":
    handler = open("data/pokelist.html")
    content = BeautifulSoup(handler, "html.parser")

    all_tables = extract_pokemon_tables(content)
    print(f"Num Tables: {len(all_tables)}")
