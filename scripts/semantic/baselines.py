import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import laft
from sklearn.linear_model import LogisticRegression

torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("high")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-name", default="ViT-B-16-quickgelu:dfn2b")
    parser.add_argument("-d", "--dataset-name", required=True, choices=("color_mnist", "waterbirds", "celeba"))
    parser.add_argument("-g", "--guidance", type=str, required=True)
    parser.add_argument("-k", "--n-neighbors", type=int, default=30)
    parser.add_argument("-s", "--n-seeds", type=int, default=5)
    parser.add_argument("-o", "--output", type=str, default="results/baselines.txt")
    parser.add_argument("--mnist-seed", type=int)
    args = parser.parse_args()

    assert args.guidance.startswith("guide_"), "baselines only supports guide"

    # Load model and data
    model, data = laft.get_clip_cached_features(
        args.model_name, args.dataset_name, splits=["train", "train-all", "test"],
        dataset_kwargs=(None if args.mnist_seed is None else {"seed": args.mnist_seed}),
    )
    train_features, _ = data["train"]
    train_all_features, train_all_attrs = data["train-all"]
    test_features, test_attrs = data["test"]

    # Setup attributes and prompts
    _, _, train_all_attend_labels, _ = \
        laft.prompts.get_labels(args.dataset_name, train_all_attrs, args.guidance)
    attend_name, ignore_name, attend_labels, ignore_labels = \
        laft.prompts.get_labels(args.dataset_name, test_attrs, args.guidance)
    prompts = laft.prompts.get_prompts(args.dataset_name, args.guidance)

    train_all_normal_features = train_all_features[~train_all_attend_labels]
    print()

    # Evaluate
    metrics = {attend_name: {}, ignore_name: {}}

    ## kNN (Only partial normal samples)
    scores = laft.knn(train_features, test_features, n_neighbors=args.n_neighbors)
    metrics[attend_name]["Normals/kNN"] = laft.binary_metrics(scores, attend_labels)
    metrics[ignore_name]["Normals/kNN"] = laft.binary_metrics(scores, ignore_labels)

    ## kNN (All normal samples)
    scores = laft.knn(train_all_normal_features, test_features, n_neighbors=args.n_neighbors)
    metrics[attend_name]["+ Unseen normals/kNN"] = laft.binary_metrics(scores, attend_labels)
    metrics[ignore_name]["+ Unseen normals/kNN"] = laft.binary_metrics(scores, ignore_labels)

    ## Linear probe (All normal and anomalous samples)
    metrics_attend = metrics[attend_name]["+ Anomalies/LinearProbe"] = []
    metrics_ignore = metrics[ignore_name]["+ Anomalies/LinearProbe"] = []

    for i in range(args.n_seeds):
        regressor = LogisticRegression(random_state=0, max_iter=1000)
        regressor.fit(train_all_features.cpu(), train_all_attend_labels.cpu())
        scores = torch.from_numpy(regressor.predict_proba(test_features.cpu())[:, 1])

        metrics_attend.append(laft.binary_metrics(scores, attend_labels))
        metrics_ignore.append(laft.binary_metrics(scores, ignore_labels))

    # Print
    table = laft.utils.build_table(metrics, group_headers=("Guidance", "Method"))
    print(table)
    laft.utils.save_table(table, args.output)
