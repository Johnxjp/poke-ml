from typing import Tuple, Optional

import numpy as np
from PIL import Image

from poke_ml.datatypes import ImageObject, NumpyArray, GenericArray


def image_preprocessing(
    img: ImageObject,
    size: Optional[Tuple[int, int]] = (40, 40),
    crop_pixels: int = 4,
) -> ImageObject:
    """
    Some images are 40x40 and some are 68x68. Resize all to 40x40.
    """

    # Convert to white background
    background = Image.new("RGBA", img.size, "white")
    img = Image.alpha_composite(background, img)
    if size:
        img = img.resize(size)
    img = img.convert("RGB")

    # Crop to remove white surrounding and to 32 x 32
    if crop_pixels > 0:
        height, width = img.size
        new_height, new_width = height - crop_pixels, width - crop_pixels
        img = img.crop((crop_pixels, crop_pixels, new_height, new_width))

    return img


def scale(
    x: GenericArray, min_val: float = -1.0, max_val: float = 1.0
) -> GenericArray:
    return x * (max_val - min_val) + min_val


def generate_random(
    size: Tuple[int, int], mean: float = 0, stdev: float = 1
) -> NumpyArray:
    return np.random.normal(mean, stdev, size)


def conv_output_dim(
    x: GenericArray, kernel_size: int, padding: int, stride: int
) -> GenericArray:
    return (x - kernel_size + 2 * padding) / stride + 1


def conv_output(
    w: int, h: int, kernel_size: int, padding: int, stride: int
) -> Tuple[GenericArray, GenericArray]:
    return (
        conv_output_dim(w, kernel_size, padding, stride),
        conv_output_dim(h, kernel_size, padding, stride),
    )
