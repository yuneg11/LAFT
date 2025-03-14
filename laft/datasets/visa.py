import os
from collections.abc import Callable
from csv import DictReader

import torch
import numpy as np

from PIL import Image

from .base import IndustrialAnomalyDataset


CATEGORIES = (
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
)


class VisA(IndustrialAnomalyDataset):
    csv_filename = "1cls.csv"

    def __init__(
        self,
        category: str,
        root: str,
        split: str,
        transform: Callable | None = None,
        mask_transform: Callable | None = None,
    ):
        assert category in CATEGORIES
        assert split in ("train", "test")

        super().__init__(root=root, split=split, transform=transform, mask_transform=mask_transform)

        self.data_root = os.path.join(root, "VisA_20220922")
        self.category = category
        self.image_filenames = []
        self.mask_filenames = []

        label_list = []

        with open(os.path.join(self.data_root, "split_csv", self.csv_filename)) as f:
            for row in DictReader(f):
                if row["object"] == category and row["split"] == split:
                    self.image_filenames.append(row["image"])
                    self.mask_filenames.append(row["mask"] or None)
                    label_list.append(row["label"] != "normal")

        self.labels = torch.as_tensor(label_list, dtype=torch.bool)

    def load_image(self, index: int):
        with open(os.path.join(self.data_root, self.image_filenames[index]), "rb") as f:
            image = Image.open(f)
            image.load()
        return image

    def load_mask(self, index: int):
        with open(os.path.join(self.data_root, self.mask_filenames[index]), "rb") as f:
            mask = Image.open(f)
            mask.load()
        return torch.from_numpy(np.asarray(mask).copy()) > 0  # Avoid PyTorch warning (copy)


def visa(
    category: str,
    root: str = "./data",
    split: str = "train",
    transform: Callable | None = None,
    mask_transform: Callable | None = None,
):
    return VisA(category, root, split, transform, mask_transform)
