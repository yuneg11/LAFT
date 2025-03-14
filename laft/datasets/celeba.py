import os
from collections.abc import Callable

import torch
from torchvision.datasets import CelebA as _CelebA

from PIL import Image

from .base import SemanticAnomalyDataset


ATTRS = [
    "5_o_Clock_Shadow",
    "Arched_Eyebrows",
    "Attractive",
    "Bags_Under_Eyes",
    "Bald",
    "Bangs",
    "Big_Lips",
    "Big_Nose",
    "Black_Hair",
    "Blond_Hair",
    "Blurry",
    "Brown_Hair",
    "Bushy_Eyebrows",
    "Chubby",
    "Double_Chin",
    "Eyeglasses",
    "Goatee",
    "Gray_Hair",
    "Heavy_Makeup",
    "High_Cheekbones",
    "Male",
    "Mouth_Slightly_Open",
    "Mustache",
    "Narrow_Eyes",
    "No_Beard",
    "Oval_Face",
    "Pale_Skin",
    "Pointy_Nose",
    "Receding_Hairline",
    "Rosy_Cheeks",
    "Sideburns",
    "Smiling",
    "Straight_Hair",
    "Wavy_Hair",
    "Wearing_Earrings",
    "Wearing_Hat",
    "Wearing_Lipstick",
    "Wearing_Necklace",
    "Wearing_Necktie",
    "Young",
]


DEFAULT_CONFIG = {
    "Blond_Hair": False,  # Blonde hair is normal (Blond: normal, No blond: anomaly)
    "Eyeglasses": True,   # Eyeglasses is anomaly (No eyeglasses: normal, Eyeglasses: anomaly)
}


class CelebA(SemanticAnomalyDataset):
    def __init__(
        self,
        root: str,
        split: str,
        transform: Callable | None = None,
        config: dict = DEFAULT_CONFIG,
    ):
        assert split in ["train", "valid", "test"]
        assert isinstance(config, dict) and len(set(config.keys()) & set(ATTRS)) == len(config)

        super().__init__(root=root, split=split, transform=transform)

        self.config = config

        # Load CelebA dataset
        dataset = _CelebA(root=root, split=split, transform=transform, target_type="attr")

        # Make CelebA with specific attributes
        self.filenames: list[str] = dataset.filename
        self.attr_names = list(config.keys())

        attrs = []

        for attr_name, attr_value in config.items():
            attr_idx = ATTRS.index(attr_name)
            attr = dataset.attr[:, attr_idx].bool()
            attrs.append(attr if attr_value else ~attr)

        self.attrs = torch.stack(attrs, dim=1)

    def load_image(self, index: int) -> Image.Image:
        with open(os.path.join(self.root, "celeba", "img_align_celeba", self.filenames[index]), "rb") as f:
            image = Image.open(f)
            image.load()
        return image


def celeba(
    root: str = "./data",
    split: str = "train",
    transform: Callable | None = None,
    config: dict = DEFAULT_CONFIG,
):
    return CelebA(root, split, transform, config)
