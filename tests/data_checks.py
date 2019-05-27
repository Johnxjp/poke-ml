import os


def all_images_exist():
    data = open("data/pokelists/all_generations.tsv").readlines()
    image_files = os.listdir("data/images")

    missing_files = []
    for row in data:
        row = row.strip().split("\t")
        image_name = row[2]

        if image_name not in image_files:
            missing_files.append(image_name)

    print(f"Number of missing files: {len(missing_files)}")
    return missing_files


if __name__ == "__main__":
    missing_files = all_images_exist()
    print(missing_files)
