import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

import laft
import baselines

torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("high")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset-name", required=True, choices=("color_mnist", "waterbirds", "celeba"))
    parser.add_argument("-m", "--checkpoint", type=str, default="checkpoints/clipn/repeat1.pt")
    parser.add_argument("-g", "--guidance", type=str, required=True)
    parser.add_argument("-w", "--words", action="store_true")
    parser.add_argument("-o", "--output", type=str, default="results/clipn.txt")
    args = parser.parse_args()

    assert args.guidance.startswith("guide_"), "clipn only supports guide"

    # Load model and data
    model, transform = baselines.load_clipn(args.checkpoint)
    data = laft.get_dataset(args.dataset_name, transform=transform, splits=["test"])
    test_dataset, test_attrs = data["test"]
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)

    # Setup attributes and prompts
    attend_name, ignore_name, attend_labels, ignore_labels = \
        laft.prompts.get_labels(args.dataset_name, test_attrs, args.guidance)

    if args.words:
        words = laft.prompts.get_words(args.dataset_name, args.guidance)
        model.setup_prompts(words=words["normal"])  # type: ignore
    else:
        prompts = laft.prompts.get_prompts(args.dataset_name, args.guidance)
        model.setup_prompts(prompts=prompts["normal"])  # type: ignore

    print()

    # Evaluate
    metrics = {attend_name: {}, ignore_name: {}}
    ctw_score_list, atd_score_list = [], []

    for images, _ in tqdm(test_loader, ncols=80, leave=False, desc="Computing"):
        _ctw_scores, _atd_scores = model(images.cuda(), score="all")
        ctw_score_list.append(_ctw_scores.cpu())
        atd_score_list.append(_atd_scores.cpu())

    ctw_scores = torch.cat(ctw_score_list)
    atd_scores = torch.cat(atd_score_list)

    metrics[attend_name]["Language/CLIPN-C"] = laft.binary_metrics(ctw_scores, attend_labels)
    metrics[ignore_name]["Language/CLIPN-C"] = laft.binary_metrics(ctw_scores, ignore_labels)

    metrics[attend_name]["Language/CLIPN-A"] = laft.binary_metrics(atd_scores, attend_labels)
    metrics[ignore_name]["Language/CLIPN-A"] = laft.binary_metrics(atd_scores, ignore_labels)

    # Print
    table = laft.utils.build_table(metrics, group_headers=("Guidance", "Method"))
    print(table)
    laft.utils.save_table(table, args.output)
