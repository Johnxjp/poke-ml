import os
from typing import Tuple, List

import numpy as np
from PIL import Image


def load_raw_data(root_dir: str) -> Tuple[List[Image.Image], List[str]]:
    """
    Loads all images as color image objects.
    """
    files = os.listdir(root_dir)
    print(f"Total number of files: {len(files)}")
    images, filenames = [], []
    for ind, file in enumerate(files):
        img = Image.open(f"{root_dir}/{file}").convert("RGBA")

        # Convert to white background
        background = Image.new("RGBA", img.size, "white")
        img = Image.alpha_composite(background, img)
        # Most images are 40 x 40 (apart from 1 which is 40 x 30)
        img = img.resize((40, 40))
        img = img.convert("RGB")

        # Crop to remove white surrounding and to 32 x 32
        img = img.crop((4, 4, 36, 36))

        images.append(img)
        filenames.append(file)

    return images, filenames


def generate_random(size, mean=0, stdev=1):
    return np.random.normal(mean, stdev, size)


def conv_output_dim(x, kernel_size, padding, stride):
    return (x - kernel_size + 2 * padding) / stride + 1


def conv_output(w, h, kernel_size, padding, stride):
    return (
        conv_output_dim(w, kernel_size, padding, stride),
        conv_output_dim(h, kernel_size, padding, stride),
    )
