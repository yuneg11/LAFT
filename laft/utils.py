import os
import json
import hashlib
from itertools import chain
from typing import Literal
from collections.abc import Callable, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

import numpy as np
from tabulate import tabulate

from .clip import load_clip
from .datasets import build_semantic_dataset


def get_dataset(
    dataset_name: Literal["color_mnist", "waterbirds", "celeba"],
    dataset_config: dict | None = None,  # None for default config
    dataset_kwargs: dict | None = None,
    transform: Callable | None = None,
    dataset_root: str = "./data",
    *,
    splits: Sequence[Literal["train", "valid", "test"]] = ("train", "test"),
    verbose: bool = True,
    print_fn: Callable = print,
):
    assert len(set(splits) & {"train", "valid", "test"}) == len(splits)
    assert dataset_name in ("color_mnist", "waterbirds", "celeba")

    data = {}

    for split in splits:
        dataset = build_semantic_dataset(
            dataset_name, split, dataset_root, transform, dataset_config, **(dataset_kwargs or {}),
        )

        if split == "train":
            subset = dataset.get_normal_subset()
            attrs = torch.zeros((len(subset), dataset.attrs.size(1)), dtype=torch.bool)
        else:
            subset = dataset
            attrs = dataset.attrs  # NOTE: hack to avoid image loading

        if verbose:
            print_fn(f"{split} set size: {len(subset)} ({len(subset) / len(dataset) * 100:.2f}%)")

        data[split] = (subset, attrs)

    return data


def get_clip_cached_features(
    model_name: str,
    dataset_name: Literal["color_mnist", "waterbirds", "celeba"],
    splits: Sequence[Literal["train", "train-all", "valid", "test"]] = ("train", "test"),
    # Model
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    model_root: str | None = "./checkpoints/open_clip",
    # Dataset
    dataset_root: str = "./data",
    dataset_config: dict | None = None,  # None for default config
    dataset_kwargs: dict | None = None,
    # Cache
    verbose: bool = True,
    print_fn: Callable = print,
    cache_root: str = "./.cache",
    flush: bool = False,
):
    assert len(set(splits) & {"train", "train-all", "valid", "test"}) == len(splits)
    assert dataset_name in ("color_mnist", "waterbirds", "celeba")

    model, transform = load_clip(model_name, device=device, download_root=model_root)

    keyset = {
        "model_name": model_name,
        "dataset_name": dataset_name,
    }

    if dataset_name == "color_mnist":
        # color_mnist images are determined with dataset config. We need to recompute image features.
        keyset.update({"dataset_config": dataset_config, **(dataset_kwargs or {})})  # type: ignore

    hashkey = hashlib.md5(json.dumps(keyset, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()
    data = {}

    for split in splits:
        cache_path = os.path.join(cache_root, hashkey, f"{split}.pt")

        # Cache hit
        if os.path.exists(cache_path) and not flush:
            if verbose:
                print_fn(f"Loading {split} from cache '{hashkey}'...")

            features, attrs = torch.load(cache_path, weights_only=True, map_location=device)
            data[split] = (features, attrs)
            continue

        # Cache miss (or flush)
        _split = "train" if split == "train-all" else split
        dataset = build_semantic_dataset(
            dataset_name, _split, dataset_root, transform, dataset_config, **(dataset_kwargs or {}),
        )
        subset = dataset.get_normal_subset() if split == "train" else dataset
        dataloader = DataLoader(subset, batch_size=32, num_workers=2, shuffle=False)

        if verbose:
            print_fn(dataset)
            print_fn(f"{split} set size: {len(subset)} ({len(subset) / len(dataset) * 100:.2f}%)")

            from tqdm.auto import tqdm
            load_iter = tqdm(dataloader, desc=f"Computing {split}", ncols=80, leave=False)
        else:
            load_iter = dataloader

        # Compute features
        with torch.inference_mode():
            _features = []
            _attrs = []

            for images, attrs in load_iter:
                _features.append(model.encode_image(images))
                _attrs.append(attrs)

            features = torch.cat(_features, dim=0)
            attrs = torch.cat(_attrs, dim=0).to(device)

        data[split] = (features, attrs)

        # Save to cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save((features.cpu(), attrs.cpu()), cache_path)

        if verbose:
            print_fn(f"Saved {split} data to cache '{hashkey}'")

    return model, data


def build_table(
    metrics: Mapping[str, Mapping[str, Mapping[str, float] | Sequence[Mapping[str, float]]]],
    group_headers: Sequence[str] | None = None,
    label_headers: Sequence[str] | None = None,
    types: Sequence[str] = ("auroc", "auprc", "fpr95"),
    meanfmt: str = "5.1f",
    stdfmt: str = "3.1f",
):
    formatted = {k: {kk: {} for kk in v} for k, v in metrics.items()}
    max_num_group_cols = 1
    group_names = list(list(metrics.values())[0].keys())

    if label_headers is None:
        label_headers = list(metrics.keys())
    elif len(label_headers) != len(metrics):
        raise ValueError(f"Expected {len(metrics)} label headers, got {len(label_headers)}")

    for k, v in metrics.items():
        for kk, vv in v.items():
            if isinstance(vv, dict):
                for t in types:
                    if t not in vv:
                        raise ValueError(f"Missing metric '{t}' for '{k}' / '{kk}'")
                    formatted[k][kk][t] = f"{vv[t]*100:{meanfmt}}"
            else:
                for t in types:
                    vs = []
                    for vvv in vv:
                        if t not in vvv:
                            raise ValueError(f"Missing metric '{t}' for '{k}' / '{kk}'")
                        vs.append(vvv[t])
                    formatted[k][kk][t] = f"{np.mean(vs)*100:{meanfmt}} ± {np.std(vs)*100:{stdfmt}}"

            num_group_cols = len(kk.split("/"))
            if max_num_group_cols < num_group_cols:
                max_num_group_cols = num_group_cols

    if group_headers is None:
        group_headers = [""] * max_num_group_cols
    elif len(group_headers) != max_num_group_cols:
        raise ValueError(f"Expected {max_num_group_cols} group headers, got {len(group_headers)}")

    types_headers = {
        "auroc": "AUROC",
        "auprc": "AUPRC",
        "accuracy": "Acc.",
        "f1": "F1",
        "fpr95": "FPR95",
    }

    table_label_headers = (
        [""] * max_num_group_cols +
        list(chain(*[[l.capitalize()] + [""] * (len(types)-1) for l in label_headers]))
    )
    table_metric_headers = list(group_headers) + [types_headers[t] for t in types] * len(label_headers)
    table_content = [
        [f"{v0}\n{v1}" for v0, v1 in zip(table_label_headers, table_metric_headers)]
    ]

    for group_name in group_names:
        cur_row = [v for v in group_name.split("/")]
        cur_row += [""] * (max_num_group_cols - len(cur_row))

        for k, v in formatted.items():
            if group_name in v:
                cur_row.extend(v[group_name].values())
            else:
                cur_row.extend([""] * len(types))

        table_content.append(cur_row)

    table = tabulate(
        table_content,
        headers="firstrow",
        colalign=("left",) * len(table_content[0]),
        disable_numparse=True,
    )
    return table


def save_table(table: str, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(table)
