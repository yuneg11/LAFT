import os
import sys
import argparse

from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from open_clip.transform import _convert_to_rgb

from tqdm.auto import tqdm, trange

import laft
from laft.clip import encode_text_ensemble
from laft.winclip_abl import load_winclip_laft

torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("high")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-name", default="ViT-B-16-plus-240:laion400m_e31")
    parser.add_argument("-d", "--dataset-name", required=True, choices=("mvtec", "visa"))
    parser.add_argument("-c", "--category", type=str, required=True)
    parser.add_argument("-n", "--n-samples", type=int, default=0)
    parser.add_argument("-s", "--n-seeds", type=int, default=1)
    parser.add_argument("-p", "--prompt", type=str, default="")
    parser.add_argument("-mx", "--max-components", type=int, default=300)
    parser.add_argument("-o", "--output", type=str, default="results/winclip.txt")
    parser.add_argument("-t", "--targets", choices=["it", "wt", "ww"], nargs="+")
    args = parser.parse_args()

    types = ["auroc"]

    # Load model and transform
    model, transform = load_winclip_laft(args.model_name)

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

    # Caching
    masks_list, labels_list = [], []
    image_embeds_list, window_embeds_list, patch_embeds_list = [], [], []

    for images, masks, labels in tqdm(test_loader, ncols=80, leave=False, desc="Caching"):
        _image_embeds, _window_embeds, _patch_embeds = model.encode_image(images.cuda())
        masks_list.append(masks)
        labels_list.append(labels)
        image_embeds_list.append(_image_embeds)
        window_embeds_list.append(_window_embeds)
        patch_embeds_list.append(_patch_embeds)

    masks = torch.cat(masks_list).cuda()
    labels = torch.cat(labels_list).cuda()
    image_embeds = torch.cat(image_embeds_list)
    window_embeds = [torch.cat(window_embeds) for window_embeds in zip(*window_embeds_list)]
    patch_embeds = torch.cat(patch_embeds_list)


    metrics = {
        "image": defaultdict(list),
        "pixel": defaultdict(list),
    }

    exec(f"from laft.prompts.industrial{args.prompt} import get_prompts as _get_prompts")
    normal_prompts, anomaly_prompts = _get_prompts(args.category)

    features = encode_text_ensemble(
        normal_prompts + anomaly_prompts,
        encode_fn=model.clip.encode_text,
        tokenize_fn=model.tokenizer,
        device="cuda",
    )
    pairs = laft.prompt_pair(features)
    basis = laft.pca(pairs)

    image_embeds_t = image_embeds @ basis.T
    window_embeds_t = [v @ basis.T for v in window_embeds]


    n_sample = args.n_samples
    ranges = lambda: (*range(2, 32), *range(32, args.max_components + 1, 2))

    with torch.inference_mode():
        if n_sample > 0:
            image_metrics_list, pixel_metrics_list = [], []

            for seed in trange(args.n_seeds, ncols=80, leave=False, desc="Seeds"):
                rng = torch.Generator().manual_seed(seed)
                idxs = torch.randperm(len(train_dataset), generator=rng)[:n_sample].tolist()
                reference_images = torch.stack([train_dataset[i][0] for i in idxs]).cuda()
                model.setup_images(reference_images)

                ref_text_embeds, ref_window_embeds, ref_patch_embeds = model.get_ref_embeds()
                ref_text_embeds_t = ref_text_embeds @ basis.T
                ref_window_embeds_t = [v @ basis.T for v in ref_window_embeds]

                for j, n_component in enumerate(tqdm(ranges(), ncols=80, leave=False, desc="Components")):
                    if "it" in args.targets:
                        it_image_embeds = image_embeds_t[..., :n_component]
                        it_ref_text_embeds = ref_text_embeds_t[..., :n_component]
                    else:
                        it_image_embeds = image_embeds
                        it_ref_text_embeds = ref_text_embeds

                    if "wt" in args.targets:
                        wt_window_embeds = [v[..., :n_component] for v in window_embeds_t]
                        wt_ref_text_embeds = ref_text_embeds_t[..., :n_component]
                    else:
                        wt_window_embeds = window_embeds
                        wt_ref_text_embeds = ref_text_embeds

                    if "ww" in args.targets:
                        ww_window_embeds = [v[..., :n_component] for v in window_embeds_t]
                        ww_ref_window_embeds = [v[..., :n_component] for v in ref_window_embeds_t]
                    else:
                        ww_window_embeds = window_embeds
                        ww_ref_window_embeds = ref_window_embeds

                    scores, heatmaps = model.scoring(
                        images,
                        it_image_embeds=it_image_embeds,
                        it_ref_text_embeds=it_ref_text_embeds,
                        wt_window_embeds=wt_window_embeds,
                        wt_ref_text_embeds=wt_ref_text_embeds,
                        ww_window_embeds=ww_window_embeds,
                        ww_ref_window_embeds=ww_ref_window_embeds,
                        pp_patch_embeds=patch_embeds,
                        pp_ref_patch_embeds=ref_patch_embeds,
                    )
                    metrics["image"][f"{n_sample}/{n_component}"].append(laft.binary_metrics(scores, labels, types=types))
                    metrics["pixel"][f"{n_sample}/{n_component}"].append(laft.binary_metrics(heatmaps, masks, types=types))

                    table = laft.utils.build_table(metrics, group_headers=("#Shot", "#Comp."), types=types)
                    tqdm.write(table if j == 0 else table.split("\n")[3 + j])

                laft.utils.save_table(table, args.output)

        else:
            ref_text_embeds, _, _ = model.get_ref_embeds()
            ref_text_embeds_t = ref_text_embeds @ basis.T

            for j, n_component in enumerate(tqdm(ranges(), ncols=80, leave=False, desc="Components")):
                if "it" in args.targets:
                    it_image_embeds = image_embeds_t[..., :n_component]
                    it_ref_text_embeds = ref_text_embeds_t[..., :n_component]
                else:
                    it_image_embeds = image_embeds
                    it_ref_text_embeds = ref_text_embeds

                if "wt" in args.targets:
                    wt_window_embeds = [v[..., :n_component] for v in window_embeds_t]
                    wt_ref_text_embeds = ref_text_embeds_t[..., :n_component]
                else:
                    wt_window_embeds = window_embeds
                    wt_ref_text_embeds = ref_text_embeds

                scores, heatmaps = model.scoring(
                    images,
                    it_image_embeds=it_image_embeds,
                    it_ref_text_embeds=it_ref_text_embeds,
                    wt_window_embeds=wt_window_embeds,
                    wt_ref_text_embeds=wt_ref_text_embeds,
                )
                metrics["image"][f"{n_sample}/{n_component}"].append(laft.binary_metrics(scores, labels, types=types))
                metrics["pixel"][f"{n_sample}/{n_component}"].append(laft.binary_metrics(heatmaps, masks, types=types))

                table = laft.utils.build_table(metrics, group_headers=("#Shot", "#Comp."), types=types)
                tqdm.write(table if j == 0 else table.split("\n")[3 + j])

    table = laft.utils.build_table(metrics, group_headers=("#Shot", "#Comp."), types=types)
    print(table)
    laft.utils.save_table(table, args.output)
