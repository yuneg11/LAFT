from collections.abc import Callable
from typing import Any
from itertools import product

import torch
from torch.utils.data import Dataset, Subset


from PIL import Image

from tabulate import tabulate


class SemanticAnomalyDataset(Dataset):
    attr_names: list[str]
    attrs: torch.Tensor  # [num_samples, num_attrs], bool (False: normal, True: anomaly)

    def __init__(self, root: str, split: str, transform: Callable | None = None) -> None:
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform

    def load_image(self, index: int) -> Image.Image:
        raise NotImplementedError

    def get_normal_subset(self):
        return Subset(self, (~self.attrs.any(dim=1)).nonzero().squeeze().tolist())

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        image = self.load_image(index)
        target = self.attrs[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def __len__(self) -> int:
        return self.attrs.size(0)

    def __str__(self):
        stats = []
        stats.append([*self.attr_names, "per. %", "num. #"])

        for values in product([False, True], repeat=self.attrs.size(1)):
            condition = torch.tensor(values, dtype=torch.bool)
            subset = (self.attrs == condition).all(dim=1)
            stats.append([*values, subset.float().mean().item() * 100, subset.sum().item()])

        return tabulate(stats, headers="firstrow", tablefmt="fancy_grid")


class IndustrialAnomalyDataset(Dataset):
    labels: torch.Tensor  # [num_samples], bool (False: normal, True: anomaly)

    def __init__(
        self,
        root: str,
        split: str,
        transform: Callable | None = None,
        mask_transform: Callable | None = None,
    ) -> None:
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.mask_transform = mask_transform

    def load_image(self, index: int) -> Image.Image:
        raise NotImplementedError

    def load_mask(self, index: int) -> torch.Tensor:
        raise NotImplementedError

    def __getitem__(self, index: int):
        image = self.load_image(index)
        label = self.labels[index]

        if label.item():  # anomaly
            mask = self.load_mask(index)
        else:
            mask = torch.zeros((image.size[1], image.size[0]), dtype=torch.bool)

        if self.transform is not None:
            image = self.transform(image)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return image, mask, label

    def __len__(self):
        return self.labels.size(0)
