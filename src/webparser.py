# Web parser for list of pokemon on bulbapedia
# Source https://bulbapedia.bulbagarden.net/wiki/List_of_Pok%C3%A9mon_by_National_Pok%C3%A9dex_number  # noqa

from bs4 import BeautifulSoup


def is_poketable(table_html: BeautifulSoup) -> bool:
    """
    The pokemon tables all contain the unique column names  "Ndex", "MS",
    and "Type". Use these to extract the right tables from the page
    """
    header_cells = table_html.find_all('th')
    # Get tag string. cell.string returns None if tag has more than one child
    header_cells = [
        cell.string.strip() for cell in header_cells if cell.string is not None
    ]

    required_header_cells = ["Ndex", "MS", "Type"]
    if len(header_cells) < len(required_header_cells):
        return False

    if all([req_cell in header_cells for req_cell in required_header_cells]):
        return True

    return False


def extract_pokemon_tables(content: BeautifulSoup) -> list:
    """
    Returns the html for all the pokemon tables
    """
    all_tables = content.find_all('table')
    poketables = [table for table in all_tables if is_poketable(table)]
    return poketables


def parse_poketable(poketable_html: BeautifulSoup):
    table_rows = poketable_html.tbody.find_all('tr')
    table_rows = table_rows[1:]  # Ignore header cell
    pokemon = [parse_poketable_row(row) for row in table_rows]
    return pokemon


def parse_poketable_row(row_html: BeautifulSoup) -> tuple:
    """
    Returns the Ndex, Name, MS code, Type 1, Type 2 in a named tuple
    """
    cells = row_html.find_all('td')

    # Extract national index
    ndex = cells[1].string.strip()[1:]  # Remove the ' #'

    # Extract name
    name = cells[3].a.string.strip().lower()

    # Extract image index
    # Example <img alt="Geodude" src="..../<id>.png" ...>
    ms = cells[2].img['src']
    ms = ms[ms.rfind("/")+1:]

    # Extract types
    type_1 = cells[4].span.string.strip().lower()
    if len(cells) > 5:
        type_2 = cells[5].span.string.strip().lower()
    else:
        type_2 = None

    return (ndex, name, ms, type_1, type_2)


if __name__ == "__main__":
    handler = open("data/pokelist.html")
    content = BeautifulSoup(handler, "html.parser")

    all_tables = extract_pokemon_tables(content)
    print(f"Num Tables: {len(all_tables)}")
    poke_info_list = [parse_poketable(table) for table in all_tables]

    all_gen_handler = open("data/pokelists/all_generations.tsv", 'w')
    for gen, pokelist in enumerate(poke_info_list, 1):

        gen_handler = open(f"data/pokelists/gen_{gen}.tsv", 'w')
        for data in pokelist:
            if data.type_2 is None:
                data_string = "\t".join(data[:-1])+"\n"
            else:
                data_string = "\t".join(data)+"\n"

            all_gen_handler.write(data_string)
            gen_handler.write(data_string)

        gen_handler.close()

    all_gen_handler.close()
