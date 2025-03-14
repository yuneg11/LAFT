import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from open_clip.transform import _convert_to_rgb

from tqdm.auto import tqdm, trange

import laft
import baselines

torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("high")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-name", default="ViT-B-16-plus-240:laion400m_e31")
    parser.add_argument("-d", "--dataset-name", required=True, choices=("mvtec", "visa"))
    parser.add_argument("-c", "--category", type=str, required=True)
    parser.add_argument("-s", "--n-seeds", type=int, default=5)
    parser.add_argument("-o", "--output", type=str, default="results/winclip.txt")
    args = parser.parse_args()

    types = ["auroc"]

    # Load model and transform
    model, transform = baselines.load_winclip(args.model_name)

    mask_transform = transforms.Compose([
        transforms.Lambda(lambda x: x.unsqueeze(0).float()),
        *transform.transforms[:transform.transforms.index(_convert_to_rgb)],
        transforms.Lambda(lambda x: x.squeeze(0) > 0.5),
    ])

    model.setup_prompts(class_name=args.category)

    # Load dataset
    train_dataset = laft.build_industrial_dataset(
        args.dataset_name, args.category, split="train", transform=transform,
    )
    test_dataset = laft.build_industrial_dataset(
        args.dataset_name, args.category, split="test", transform=transform, mask_transform=mask_transform,
    )
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)

    # Evaluate
    metrics = {"image": {}, "pixel": {}}

    for i, n_sample in enumerate([0, 1, 2, 4, 8]):
        if n_sample > 0:
            image_metrics_list, pixel_metrics_list = [], []

            for seed in trange(args.n_seeds, ncols=80, leave=False, desc="Seeds"):
                rng = torch.Generator().manual_seed(seed)
                idxs = torch.randperm(len(train_dataset), generator=rng)[:n_sample].tolist()
                reference_images = torch.stack([train_dataset[i][0] for i in idxs]).cuda()
                model.setup_images(reference_images)

                scores_list, heatmaps_list, masks_list, labels_list = [], [], [], []

                for images, masks, labels in tqdm(test_loader, ncols=80, leave=False, desc=f"{n_sample}-shot"):
                    scores, heatmaps = model(images.cuda())
                    scores_list.append(scores.cpu())
                    heatmaps_list.append(heatmaps.cpu())
                    masks_list.append(masks)
                    labels_list.append(labels)

                image_metrics_list.append(laft.binary_metrics(torch.cat(scores_list), torch.cat(labels_list), types=types))
                pixel_metrics_list.append(laft.binary_metrics(torch.cat(heatmaps_list), torch.cat(masks_list), types=types))

            metrics["image"][f"WinCLIP+/{n_sample}"] = image_metrics_list
            metrics["pixel"][f"WinCLIP+/{n_sample}"] = pixel_metrics_list

        else:
            scores_list, heatmaps_list, masks_list, labels_list = [], [], [], []

            for images, masks, labels in tqdm(test_loader, ncols=80, leave=False, desc=f"{n_sample}-shot"):
                scores, heatmaps = model(images.cuda())
                scores_list.append(scores.cpu())
                heatmaps_list.append(heatmaps.cpu())
                masks_list.append(masks)
                labels_list.append(labels)

            metrics["image"][f"WinCLIP/{n_sample}"] = laft.binary_metrics(torch.cat(scores_list), torch.cat(labels_list), types=types)
            metrics["pixel"][f"WinCLIP/{n_sample}"] = laft.binary_metrics(torch.cat(heatmaps_list), torch.cat(masks_list), types=types)

        table = laft.utils.build_table(metrics, group_headers=("Method", "#Shot"), types=types)
        print(table if i == 0 else table.split("\n")[-1])
        laft.utils.save_table(table, args.output)

    # Print
    table = laft.utils.build_table(metrics, group_headers=("Method", "#Shot"), types=types)
    laft.utils.save_table(table, args.output)
