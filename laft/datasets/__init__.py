from collections.abc import Callable
from typing import Literal

from .base import SemanticAnomalyDataset, IndustrialAnomalyDataset


def build_semantic_dataset(
    name: Literal["color_mnist", "waterbirds", "celeba"],
    split: Literal["train", "valid", "test"],
    root: str = "./data",
    transform: Callable | None = None,
    config: dict | None = None,  # None for default config
    **kwargs,
) -> SemanticAnomalyDataset:
    assert name in ("color_mnist", "waterbirds", "celeba")

    _config = {} if config is None else {"config": config}

    if name == "color_mnist":
        from .color_mnist import color_mnist
        return color_mnist(root, split, transform, **_config, seed=kwargs.get("seed", 42))
    elif name == "waterbirds":
        from .waterbirds import waterbirds
        return waterbirds(root, split, transform, **_config)
    elif name == "celeba":
        from .celeba import celeba
        return celeba(root, split, transform, **_config)
    else:
        raise ValueError(f"Unknown semantic dataset '{name}'")


def build_industrial_dataset(
    name: Literal["mvtec", "visa"],
    category: str,
    split: Literal["train", "test"],
    root: str = "./data",
    transform: Callable | None = None,
    mask_transform: Callable | None = None,
) -> IndustrialAnomalyDataset:
    if name == "mvtec":
        from .mvtec import mvtec
        return mvtec(category, root, split, transform, mask_transform)
    elif name == "visa":
        from .visa import visa
        return visa(category, root, split, transform, mask_transform)
    else:
        raise ValueError(f"Unknown industrial dataset '{name}'")
