import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch

import laft
import baselines

torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("high")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-name", default="ViT-B-16-quickgelu:dfn2b")
    parser.add_argument("-d", "--dataset-name", required=True, choices=("color_mnist", "waterbirds", "celeba"))
    parser.add_argument("-g", "--guidance", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, default="results/clip_score.txt")
    args = parser.parse_args()

    assert args.guidance.startswith("guide_"), "clip_score only supports guide"

    # Load model and data
    model, data = laft.get_clip_cached_features(args.model_name, args.dataset_name, splits=["test"])
    test_features, test_attrs = data["test"]

    # Setup attributes and prompts
    attend_name, ignore_name, attend_labels, ignore_labels = \
        laft.prompts.get_labels(args.dataset_name, test_attrs, args.guidance)
    prompts = laft.prompts.get_prompts(args.dataset_name, args.guidance)
    print()

    # Evaluate
    metrics = {attend_name: {}, ignore_name: {}}

    ## MCM
    scores = baselines.clip_score(model, test_features, prompts["normal"])
    metrics[attend_name]["Language/MCM"] = laft.binary_metrics(scores, attend_labels)
    metrics[ignore_name]["Language/MCM"] = laft.binary_metrics(scores, ignore_labels)

    ## ZOE
    scores = baselines.clip_score(model, test_features, prompts["normal"], prompts["anomaly"])
    metrics[attend_name]["Language/ZOE"] = laft.binary_metrics(scores, attend_labels)
    metrics[ignore_name]["Language/ZOE"] = laft.binary_metrics(scores, ignore_labels)

    # Print
    table = laft.utils.build_table(metrics, group_headers=("Guidance", "Method"))
    print(table)
    laft.utils.save_table(table, args.output)
