import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import laft

torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("high")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-name", default="ViT-B-16-quickgelu:dfn2b")
    parser.add_argument("-d", "--dataset-name", required=True, choices=("color_mnist", "waterbirds", "celeba"))
    parser.add_argument("-k", "--n-neighbors", type=int, default=30)
    parser.add_argument("-g", "--guidance", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, default="results/laft.txt")
    parser.add_argument("-p", "--prompt", default="all", choices=("normal", "half", "exact", "all"))
    parser.add_argument("--mnist-seed", type=int)
    args = parser.parse_args()

    # Load model and data
    model, data = laft.get_clip_cached_features(
        args.model_name, args.dataset_name, splits=["train", "test"],
        dataset_kwargs=(None if args.mnist_seed is None else {"seed": args.mnist_seed}),
    )
    train_features, _ = data["train"]
    test_features, test_attrs = data["test"]

    # Setup attributes and prompts
    attend_name, ignore_name, attend_labels, ignore_labels = \
        laft.prompts.get_labels(args.dataset_name, test_attrs, args.guidance)
    prompts = laft.prompts.get_prompts(args.dataset_name, args.guidance)
    print()

    # Evaluate
    metrics = {attend_name: {}, ignore_name: {}}

    projection = laft.inner if args.guidance.startswith("guide") else laft.orthogonal
    features = model.encode_text(prompts[args.prompt])
    pairs = laft.prompt_pair(features)
    concept_basis = laft.pca(pairs)

    guide, attr = args.guidance.split("_")
    metric_name = f"{guide.capitalize()}/{attr.capitalize()}"

    for i, n_components in enumerate(range(2, 385)):
        train_laft_features = projection(train_features, concept_basis[:n_components])
        test_laft_features = projection(test_features, concept_basis[:n_components])

        scores = laft.knn(train_laft_features, test_laft_features, n_neighbors=args.n_neighbors)
        metrics[attend_name][f"{metric_name}/{n_components}"] = laft.binary_metrics(scores, attend_labels)
        metrics[ignore_name][f"{metric_name}/{n_components}"] = laft.binary_metrics(scores, ignore_labels)

        table = laft.utils.build_table(metrics, group_headers=("Guide", "Attr.", "Comp."))
        print(table if i == 0 else table.split("\n")[-1])

    # Print
    table = laft.utils.build_table(metrics, group_headers=("Guide", "Attr.", "Comp."))
    print(table)
    laft.utils.save_table(table, args.output)
