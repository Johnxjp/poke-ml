import os

import numpy as np
from PIL import Image


def load_images(root_dir):
    """Loads all images in a numpy array. All images are given as 40x40"""
    files = os.listdir(root_dir)
    print(f"Total number of files: {len(files)}")

    max_size = 40
    images = np.zeros((len(files), max_size, max_size, 3), dtype=np.float32)
    filenames = []
    for ind, file in enumerate(files):
        original_image = Image.open(f"{root_dir}/{file}").convert('RGBA')

        # Convert to white background
        new_image = Image.new('RGBA', original_image.size, "white")
        new_image.paste(original_image, mask=original_image)
        im = new_image.convert('RGB')
        # Crop image if to large. All images will be 40x40
        if im.size[0] > max_size:
            top_left = (max_size - im.size[0]) // 2
            im = im.crop(
                (top_left, top_left, top_left + max_size, top_left + max_size)
            )

        images[ind] = np.asarray(im) / 255
        filenames.append(file)

    return images, filenames


def generate_random(size, mean=0, stdev=1):
    return np.random.normal(mean, stdev, size)
