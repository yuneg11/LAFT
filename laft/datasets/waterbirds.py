import os
import csv
from collections.abc import Callable

import torch

from PIL import Image

from .base import SemanticAnomalyDataset


DEFAULT_CONFIG = {
    "bird": {
        "land": True,    # Anomaly
        "water": False,  # Normal
    },
    "background": {
        "land": True,    # Anomaly
        "water": False,  # Normal
    }
}


class Waterbirds(SemanticAnomalyDataset):
    attr_names = ["bird", "background"]

    def __init__(
        self,
        root: str,
        split: str,
        transform: Callable | None = None,
        config: dict = DEFAULT_CONFIG,
    ):
        assert split in ["train", "valid", "test"]
        assert isinstance(config, dict) and set(config.keys()) == {"bird", "background"}
        assert set(config["bird"].keys()) == set(config["background"].keys()) == {"land", "water"}
        assert set(config["bird"].values()) == set(config["background"].values()) == {False, True}

        super().__init__(root=root, split=split, transform=transform)

        self.config = config
        self.filenames: list[str] = []

        # Load Waterbirds dataset
        split_id = {"train": "0", "valid": "1", "test": "2"}[split]
        bird_anomaly = "0" if config["bird"]["land"] else "1"
        background_anomaly = "0" if config["background"]["land"] else "1"

        attrs = []

        with open(os.path.join(self.root, "waterbirds_v1.0", "metadata.csv"), "r") as f:
            for row in csv.DictReader(f):
                if row["split"] != split_id:
                    continue

                self.filenames.append(row["img_filename"])
                attrs.append([row["y"] == bird_anomaly, row["place"] == background_anomaly])

        self.attrs = torch.tensor(attrs, dtype=torch.bool)

    def load_image(self, index: int) -> Image.Image:
        with open(os.path.join(self.root, "waterbirds_v1.0", self.filenames[index]), "rb") as f:
            image = Image.open(f)
            image.load()
        return image


def waterbirds(
    root: str = "./data",
    split: str = "train",
    transform: Callable | None = None,
    config: dict = DEFAULT_CONFIG,
):
    return Waterbirds(root, split, transform, config)
