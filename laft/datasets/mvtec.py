import os
from collections.abc import Callable

import torch
import numpy as np

from PIL import Image

from .base import IndustrialAnomalyDataset


CATEGORIES = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)


class MVTecAD(IndustrialAnomalyDataset):
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

        self.data_root = os.path.join(root, "mvtec_anomaly_detection", category)
        self.category = category
        self.filenames = []

        label_list = []

        for anomaly in [v for v in os.scandir(os.path.join(self.data_root, split)) if v.is_dir()]:
            images = [
                os.path.join(anomaly.name, v.name[:-4]) for v in os.scandir(anomaly.path)
                if v.name.endswith(".png")
            ]
            self.filenames.extend(images)
            label_list.extend([anomaly.name != "good"] * len(images))

        self.labels = torch.as_tensor(label_list, dtype=torch.bool)

    def load_image(self, index: int):
        with open(os.path.join(self.data_root, self.split, f"{self.filenames[index]}.png"), "rb") as f:
            image = Image.open(f)
            image.load()
        return image

    def load_mask(self, index: int):
        with open(os.path.join(self.data_root, "ground_truth", f"{self.filenames[index]}_mask.png"), "rb") as f:
            mask = Image.open(f)
            mask.load()
        return torch.from_numpy(np.asarray(mask).copy()) > 0  # Avoid PyTorch warning (copy)


def mvtec(
    category: str,
    root: str = "./data",
    split: str = "train",
    transform: Callable | None = None,
    mask_transform: Callable | None = None,
):
    return MVTecAD(category, root, split, transform, mask_transform)
