from typing import Callable, overload

import torch
from torch import nn
from torch.nn import functional as F

import open_clip
from open_clip.model import CLIP
from open_clip.tokenizer import tokenize

from PIL.Image import Image

from laft.clip import encode_text_ensemble

from .winclip import create_prompts, harmonic_aggregation, make_masks


class TransformerBasicHead(nn.Module):
    def __init__(self, dim_in):
        super(TransformerBasicHead, self).__init__()
        self.projection1 = nn.Linear(dim_in, 128, bias=True)
        self.projection2 = nn.Linear(128, 64, bias=True)
        self.projection3 = nn.Linear(64, 1, bias=True)
        self.bn1 = nn.BatchNorm1d(dim_in)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.projection1(x)
        x = F.relu(x, inplace=True)
        x = self.bn2(x)
        x = self.projection2(x)
        x = F.relu(x, inplace=True)
        x = self.bn3(x)
        x = self.projection3(x)
        return torch.sigmoid(x).squeeze(1)


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.fc(x)


class InCTRL(nn.Module):
    def __init__(self, clip: CLIP):
        super().__init__()

        self.clip = clip
        self.adapter = Adapter(640, 4)
        self.diff_head = TransformerBasicHead(225)
        self.diff_head_ref = TransformerBasicHead(640)

        self.mask = make_masks((16, 16), 2)

        # Feature map hook
        self._outputs = []

        def _hook(_model, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor):
            self._outputs.append(output.detach())

        self.clip.visual.transformer.resblocks[6].register_forward_hook(_hook)
        self.clip.visual.transformer.resblocks[8].register_forward_hook(_hook)
        self.clip.visual.transformer.resblocks[10].register_forward_hook(_hook)

    @overload
    def setup_prompts(self, *, class_name: str):
        ...

    @overload
    def setup_prompts(self, normal_prompts: list[str] | list[list[str]], anomaly_prompts: list[str] | list[list[str]]):
        ...

    @torch.no_grad()
    def setup_prompts(
        self,
        normal_prompts: list[str] | list[list[str]] | None = None,
        anomaly_prompts: list[str] | list[list[str]] | None = None,
        *,
        class_name: str | None = None,
    ):
        if class_name is not None:
            normal_prompts, anomaly_prompts = create_prompts(class_name)
        elif normal_prompts is None or anomaly_prompts is None:
            raise ValueError("Either class_name or both normal_prompts and anomaly_prompts must be provided")

        pos_features = encode_text_ensemble(
            normal_prompts,
            encode_fn=self.clip.encode_text,
            tokenize_fn=tokenize,
            device=self.clip.logit_scale.device,
            normalize=True,
        ).mean(dim=0)

        neg_features = encode_text_ensemble(
            anomaly_prompts,
            encode_fn=self.clip.encode_text,
            tokenize_fn=tokenize,
            device=self.clip.logit_scale.device,
            normalize=True,
        ).mean(dim=0)

        pos_features = pos_features / pos_features.norm(dim=-1, keepdim=True)
        neg_features = neg_features / neg_features.norm(dim=-1, keepdim=True)
        self._text_embeds = torch.stack([pos_features, neg_features])

    @torch.no_grad()
    def setup_images(self, reference_images: torch.Tensor):
        image_embeds, patch_embeds = self.encode_image(reference_images)

        patch_embeds = patch_embeds.flatten(1, 2)
        patch_embeds = patch_embeds / patch_embeds.norm(dim=-1, keepdim=True)

        image_embeds = self.adapter(image_embeds)
        image_embeds = torch.mean(image_embeds, dim=0, keepdim=True)

        self._image_embeds_ref = image_embeds  # [1, 640]
        self._patch_embeds_ref = patch_embeds  # [3, 225 x shot, 896]

    def encode_image(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image_embeds = self.clip.visual(batch)
        patch_embeds = torch.stack(self._outputs)[:, :, 1:]
        self._outputs.clear()
        return image_embeds, patch_embeds

    def forward(self, image: torch.Tensor):
        image_embeds, patch_embeds = self.encode_image(image)
        # image_embeds: [B, 640]
        # patch_embeds: [3, B, 225, 896]

        patch_embeds = patch_embeds / patch_embeds.norm(dim=-1, keepdim=True)

        norm_image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_score = (100 * norm_image_embeds @ self._text_embeds.T).softmax(dim=-1)[:, 1]

        patch_sim = torch.matmul(patch_embeds, self._patch_embeds_ref.transpose(1, 2).unsqueeze(1))
        patch_ref_map = (1 - patch_sim).min(dim=3)[0] / 2  # [3, B, 225]
        patch_ref_map = torch.mean(patch_ref_map, dim=0)   # [B, 225]
        fg_score = patch_ref_map.max(dim=1)[0]             # [B]

        image_embeds_diff = self._image_embeds_ref - self.adapter(image_embeds)
        img_ref_score = self.diff_head_ref(image_embeds_diff)

        holistic_map = (text_score + img_ref_score).unsqueeze(1) + patch_ref_map
        hl_score = self.diff_head(holistic_map)
        final_score = (hl_score + fg_score) / 2

        heatmap = harmonic_aggregation(holistic_map, (16, 16), self.mask)
        holistic_map = F.interpolate(
            heatmap.unsqueeze(1),
            size=image.shape[2:],
            mode="bilinear",
        ).squeeze(1)

        return final_score, holistic_map


def load_inctrl(
    checkpoint: str,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    **kwargs,
) -> tuple[InCTRL, Callable[[Image], torch.Tensor]]:
    clip, _, transform = open_clip.create_model_and_transforms(
        "ViT-B-16-plus-240",
        device=device,
        # force_quick_gelu=True,  # XXX: check this (significantly affects performance)
        **kwargs,
    )

    model = InCTRL(clip)  # type: ignore

    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    state_dict = {
        k if k.startswith("adapter.") or k.startswith("diff_head.") or k.startswith("diff_head_ref.")
        else f"clip.{k}": v
        for k, v in state_dict.items()
    }
    state_dict["clip.logit_scale"] = torch.tensor(1.0)

    model.load_state_dict(state_dict, strict=True)
    model = model.eval().to(device)

    return model, transform  # type: ignore
