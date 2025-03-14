from collections.abc import Callable

import torch
from torchvision.datasets import MNIST

from PIL import Image

from .base import SemanticAnomalyDataset


DEFAULT_CONFIG = {
    "number": {
        0: False,  # Normal
        1: False,  # Normal
        2: False,  # Normal
        3: False,  # Normal
        4: False,  # Normal
        5: True,   # Anomaly
        6: True,   # Anomaly
        7: True,   # Anomaly
        8: True,   # Anomaly
        9: True,   # Anomaly
    },
    "color": {
        "red": False,   # Normal
        "green": True,  # Anomaly
        "blue": True,   # Anomaly
    },
}


def _train_valid_split(dataset, num_train, num_valid, seed=42):
    rng = torch.Generator().manual_seed(seed)
    train_idxs, valid_idxs = [], []

    for i in range(10):
        class_idxs = (dataset.targets == i).nonzero(as_tuple=False).flatten().tolist()
        perm_class_idxs = torch.randperm(len(class_idxs), generator=rng).tolist()

        train_idxs.extend([class_idxs[idx] for idx in perm_class_idxs[:num_train]])
        valid_idxs.extend([class_idxs[idx] for idx in perm_class_idxs[num_train:num_train+num_valid]])

    return train_idxs, valid_idxs


def _coloring(image, color: str) -> Image.Image:
    image = torch.constant_pad_nd(image, (28, 28, 28, 28), 0)
    zero_image = torch.zeros_like(image)

    if color == "red":
        image = torch.stack([image, zero_image, zero_image], dim=-1)
    elif color == "green":
        image = torch.stack([zero_image, image, zero_image], dim=-1)
    elif color == "blue":
        image = torch.stack([zero_image, zero_image, image], dim=-1)
    else:
        raise ValueError(f"Color {color} is not supported.")

    return Image.fromarray(image.numpy())


class ColorMNIST(SemanticAnomalyDataset):
    attr_names = ["number", "color"]

    def __init__(
        self,
        root: str,
        split: str,
        transform: Callable | None = None,
        config: dict = DEFAULT_CONFIG,
        seed: int = 42,
    ):
        assert split in ["train", "valid", "test"]
        assert isinstance(config, dict) and set(config.keys()) == {"number", "color"}

        super().__init__(root=root, split=split, transform=transform)

        self.config = config
        self.images = []

        number_config = config["number"]
        color_config = config["color"]

        # Load MNIST dataset
        dataset = MNIST(root=root, train=False if split == "test" else True)

        # Number of data for each class
        if split == "train":
            num_of_data = [4500] * 10
        elif split == "valid":
            num_of_data = [900] * 10
        else:
            num_of_data = [870] * 10

        for i in range(10):
            if i not in config["number"] or config["number"][i] is None:
                num_of_data[i] = 0

        # Split train dataset into train and valid
        if split in ["train", "valid"]:
            train_idxs, valid_idxs = _train_valid_split(dataset, 4500, 900, seed=seed)
            idxs = train_idxs if split == "train" else valid_idxs
        else:
            idxs = list(range(len(dataset)))

        # Make color MNIST
        attrs = []
        cnt = [0] * 10
        colors = list(color_config.keys())

        for idx in idxs:
            number = int(dataset.targets[idx])

            if cnt[number] >= num_of_data[number]:
                continue

            image: torch.Tensor = dataset.data[idx]
            color = colors[cnt[number] % len(colors)]

            self.images.append(_coloring(image, color))
            attrs.append([number_config[number], color_config[color]])
            cnt[number] += 1

        self.attrs = torch.tensor(attrs, dtype=torch.bool)

    def load_image(self, index: int) -> Image.Image:
        image = self.images[index]
        return image


def color_mnist(
    root: str = "./data",
    split: str = "train",
    transform: Callable | None = None,
    config: dict = DEFAULT_CONFIG,
    seed: int = 42,
):
    return ColorMNIST(root, split, transform, config, seed)
