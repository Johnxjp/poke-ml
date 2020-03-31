"""
Module for loading the data
"""
import os
from typing import Sequence, Optional, Tuple, Mapping

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision.transforms import ToTensor


from poke_ml.datatypes import (
    ImageObject,
    ImageTransform,
    ComposedTorchTransforms,
    Tensor,
)


def load_index_table() -> pd.DataFrame:
    file = "../data/tables/pokemon_by_shape.csv"
    return pd.read_csv(file, sep=",", header=0)


def load_image(
    file_name: str, image_transform: Optional[ImageTransform] = None,
) -> ImageObject:
    """File name must be full path"""
    img = Image.open(file_name).convert("RGBA")
    if image_transform:
        return image_transform(img)
    return img


def load_raw_data(
    root_dir: str, image_transform: Optional[ImageTransform] = None
) -> Tuple[Sequence[ImageObject], Sequence[str]]:
    """
    Loads all images as color image objects.
    """
    files = os.listdir(root_dir)
    print(f"Total number of files: {len(files)}")
    images, filenames = [], []
    for ind, file in enumerate(files):
        img = load_image(f"{root_dir}/{file}", image_transform)
        images.append(img)
        filenames.append(file)

    return images, filenames


class IndexDataset(Dataset):
    def __init__(
        self,
        index_table: pd.DataFrame,
        img_dir: str,
        file_col: int = 1,
        label_col: int = 5,
        label_mapping: Optional[Mapping[str, int]] = None,
        image_transform: Optional[ImageTransform] = None,
        torch_transforms: Optional[ComposedTorchTransforms] = None,
    ) -> None:
        """
        file and label col are 0-based.

        label_mapping is to convert strings labels into integers.
        If none this is done implicitly.
        """
        self.index_table = index_table
        self.file_column = index_table.columns[file_col]
        self.label_column = index_table.columns[label_col]
        self.img_dir = img_dir
        self.image_transform = image_transform

        self.torch_transforms = torch_transforms
        if not torch_transforms:
            self.torch_transforms = ToTensor

        self.label_mapping = label_mapping
        if not label_mapping:
            labels = self.index_table[self.label_column].unique()
            self.label_mapping = {k: v for v, k in enumerate(labels)}

    def __len__(self) -> int:
        return len(self.index_table)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        data_item = self.index_table.iloc[idx]
        file_name = self.img_dir + "/" + data_item[self.file_column]
        img = load_image(file_name, self.image_transform)
        label = self.label_mapping[data_item[self.label_column]]

        if self.torch_transforms:
            img = self.torch_transforms(img)

        label = torch.tensor(label, dtype=torch.long)
        return img, label


class ImageDataset(Dataset):
    def __init__(
        self,
        images: ImageObject,
        transforms: Optional[ComposedTorchTransforms] = None,
    ) -> None:
        """Takes in PIL Images"""
        self.data = images
        self.transforms = transforms

    def __getitem__(self, idx: int) -> Tensor:
        item = self.data[idx]
        return self.transforms(item)

    def __len__(self) -> int:
        return len(self.data)
