import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import laft
from baselines import clip_score

torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("high")


DATASET_CONFIG = {
    "Blond_Hair": True,  # 9
    "Eyeglasses": False,  # 15
    "Young": True,  # 39
}
DATASET_ATTR_INDICES = [9, 15, 20]

GENDER_TEMPLATES = [
    "a photo of {}",
    "a good photo of {}",
    "a bad photo of {}",
    "a small photo of {}",
    "a big photo of {}",
    "a picture of {}",
    "a photograph of {}",
    "a portrait of {}",
]
GENDER_WORDS = [
    ["man",  "male", "boy", "masculine"],
    ["woman", "female", "girl", "feminine"],
]

PROMPT_GENDER_IND = [[f.format(v) for f in GENDER_TEMPLATES] for w in GENDER_WORDS[:1] for v in w]
PROMPT_GENDER_OOD = [[f.format(v) for f in GENDER_TEMPLATES] for w in GENDER_WORDS[1:] for v in w]
PROMPT_GENDER_NOT_IND = PROMPT_GENDER_OOD
PROMPT_GENDER = PROMPT_GENDER_IND + PROMPT_GENDER_OOD


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--results-dir", default="results")
    parser.add_argument("-m", "--model-name", default="ViT-B-16-quickgelu:dfn2b")
    args = parser.parse_args()

    dataset_name = "celeba"

    ds = laft.prompts.get_ds(dataset_name)
    model, data = laft.get_cached_features(args.model_name, dataset_name, DATASET_CONFIG)

    train_features, train_attrs = data["train"]
    valid_features, valid_attrs = data["valid"]
    test_features, test_attrs = data["test"]

    gender_features = model.encode_text([v for w in PROMPT_GENDER for v in w]).float()
    prompt_pairs = laft.prompt_pair(gender_features)
    concept_basis = laft.pca(prompt_pairs)

    with torch.inference_mode():
        blond_ind_features = model.encode_text(ds.PROMPT_BLOND_IND)
        blond_ood_features = model.encode_text(ds.PROMPT_BLOND_OOD)
        glass_ind_features = model.encode_text(ds.PROMPT_GLASS_IND)
        glass_ood_features = model.encode_text(ds.PROMPT_GLASS_OOD)
        gender_ind_features = model.encode_text(PROMPT_GENDER_IND)
        gender_ood_features = model.encode_text(PROMPT_GENDER_OOD)

    mcm_metrics = {
        "Blond": {},
        "Glass": {},
        "Gender": {},
    }
    zoc_metrics = {
        "Blond": {},
        "Glass": {},
        "Gender": {},
    }

    # MCM
    mcm_scores = clip_score(model, test_features, blond_ind_features)
    mcm_metrics["Blond"]["None"] = laft.binary_metrics(mcm_scores, 1 - test_attrs[:, 9])

    mcm_scores = clip_score(model, test_features, gender_ind_features)
    mcm_metrics["Gender"]["None"] = laft.binary_metrics(mcm_scores, 1 - test_attrs[:, 20])

    zoc_scores = clip_score(model, test_features, blond_ind_features, blond_ood_features)
    zoc_metrics["Blond"]["None"] = laft.binary_metrics(zoc_scores, 1 - test_attrs[:, 9])

    zoc_scores = clip_score(model, test_features, gender_ind_features, gender_ood_features)
    zoc_metrics["Gender"]["None"] = laft.binary_metrics(zoc_scores, 1 - test_attrs[:, 20])

    for i in range(2, concept_basis.size(0) + 1):
        blond_ind_laft_features = laft.orthogonal(blond_ind_features, concept_basis[:i])
        blond_ood_laft_features = laft.orthogonal(blond_ood_features, concept_basis[:i])
        glass_ind_laft_features = laft.orthogonal(glass_ind_features, concept_basis[:i])
        glass_ood_laft_features = laft.orthogonal(glass_ood_features, concept_basis[:i])
        gender_ind_laft_features = laft.orthogonal(gender_ind_features, concept_basis[:i])
        gender_ood_laft_features = laft.orthogonal(gender_ood_features, concept_basis[:i])
        test_laft_features = laft.orthogonal(test_features, concept_basis[:i])

        mcm_scores = clip_score(model, test_laft_features, blond_ind_laft_features)
        mcm_metrics["Blond"][f"{i}"] = laft.binary_metrics(mcm_scores, 1 - test_attrs[:, 9])

        mcm_scores = clip_score(model, test_laft_features, gender_ind_laft_features)
        mcm_metrics["Gender"][f"{i}"] = laft.binary_metrics(mcm_scores, 1 - test_attrs[:, 20])

        zoc_scores = clip_score(model, test_laft_features, blond_ind_laft_features, blond_ood_laft_features)
        zoc_metrics["Blond"][f"{i}"] = laft.binary_metrics(zoc_scores, 1 - test_attrs[:, 9])

        zoc_scores = clip_score(model, test_laft_features, gender_ind_laft_features, gender_ood_laft_features)
        zoc_metrics["Gender"][f"{i}"] = laft.binary_metrics(zoc_scores, 1 - test_attrs[:, 20])

    mcm_table = laft.utils.build_table(mcm_metrics)
    print(mcm_table)
    mcm_table_path = os.path.join(args.results_dir, "discussion", "celeba", "mcm_laft.txt")
    laft.utils.save_table(mcm_table, mcm_table_path)

    zoc_table = laft.utils.build_table(zoc_metrics)
    print(zoc_table)
    zoc_table_path = os.path.join(args.results_dir, "discussion", "celeba", "zoc_laft.txt")
    laft.utils.save_table(zoc_table, zoc_table_path)
