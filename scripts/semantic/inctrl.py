import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
from torch.utils.data import DataLoader

from tqdm.auto import tqdm, trange

import laft
import baselines

torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("high")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset-name", required=True, choices=("color_mnist", "waterbirds", "celeba"))
    parser.add_argument("-m", "--checkpoint", type=str, default="checkpoints/inctrl/mvtec2.pt")
    parser.add_argument("-g", "--guidance", type=str, required=True)
    parser.add_argument("-s", "--n-seeds", type=int, default=5)
    parser.add_argument("-o", "--output", type=str, default="results/inctrl.txt")
    args = parser.parse_args()

    assert args.guidance.startswith("guide_"), "inctrl only supports guide"

    # Load model and data
    model, transform = baselines.load_inctrl(args.checkpoint)
    data = laft.get_dataset(args.dataset_name, transform=transform, splits=["train", "test"])
    train_dataset, _ = data["train"]
    test_dataset, test_attrs = data["test"]
    test_loader = DataLoader(test_dataset, batch_size=64, num_workers=8, persistent_workers=True)

    # Setup attributes and prompts
    attend_name, ignore_name, attend_labels, ignore_labels = \
        laft.prompts.get_labels(args.dataset_name, test_attrs, args.guidance)
    prompts = laft.prompts.get_prompts(args.dataset_name, args.guidance)
    model.setup_prompts(prompts["normal"], prompts["anomaly"])
    print()

    # Evaluate
    metrics = {attend_name: {}, ignore_name: {}}

    for i, n_sample in enumerate([1, 2, 4, 8, 16, 32, 64]):
        attend_metrics_list = []
        ignore_metrics_list = []

        for seed in trange(args.n_seeds, ncols=80, leave=False, desc="Seeds"):
            rng = torch.Generator().manual_seed(seed)
            idxs = torch.randperm(len(train_dataset), generator=rng)[:n_sample].tolist()
            reference_images = torch.stack([train_dataset[i][0] for i in idxs]).cuda()
            model.setup_images(reference_images)

            scores = torch.cat([
                model(images.cuda())[0].cpu()
                for images, _ in tqdm(test_loader, ncols=80, leave=False, desc=f"{n_sample}-shot")
            ])
            attend_metrics_list.append(laft.binary_metrics(scores, attend_labels))
            ignore_metrics_list.append(laft.binary_metrics(scores, ignore_labels))

        metrics[attend_name][f"Image + Language/InCTRL/{n_sample}"] = attend_metrics_list
        metrics[ignore_name][f"Image + Language/InCTRL/{n_sample}"] = ignore_metrics_list

        table = laft.utils.build_table(metrics, group_headers=("Guidance", "Method", "#Shot",))
        print(table if i == 0 else table.split("\n")[-1])
        laft.utils.save_table(table, args.output)

    # Print
    table = laft.utils.build_table(metrics, group_headers=("Guidance", "Method", "#Shot",))
    print(table)
    laft.utils.save_table(table, args.output)
