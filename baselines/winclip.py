# Code adapted from: https://github.com/openvinotoolkit/anomalib/blob/main/src/anomalib/models/image/winclip
# License: Apache-2.0

from typing import overload
from collections.abc import Callable, Sequence

import torch
from torch import nn
from torch.nn.functional import interpolate

import open_clip
from open_clip.model import CLIP, CustomTextCLIP
from open_clip.coca_model import CoCa
from open_clip.transformer import VisionTransformer
from open_clip.tokenizer import SimpleTokenizer, HFTokenizer

from PIL.Image import Image

from laft.clip import encode_text_ensemble


NORMAL_STATES = [
    "{}",
    "flawless {}",
    "perfect {}",
    "unblemished {}",
    "{} without flaw",
    "{} without defect",
    "{} without damage",
]

ANOMALOUS_STATES = [
    "damaged {}",
    "{} with flaw",
    "{} with defect",
    "{} with damage",
]

TEMPLATES = [
    "a cropped photo of the {}.",
    "a close-up photo of a {}.",
    "a close-up photo of the {}.",
    "a bright photo of a {}.",
    "a bright photo of the {}.",
    "a dark photo of the {}.",
    "a dark photo of a {}.",
    "a jpeg corrupted photo of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a blurry photo of the {}.",
    "a blurry photo of a {}.",
    "a photo of a {}.",
    "a photo of the {}.",
    "a photo of a small {}.",
    "a photo of the small {}.",
    "a photo of a large {}.",
    "a photo of the large {}.",
    "a photo of the {} for visual inspection.",
    "a photo of a {} for visual inspection.",
    "a photo of the {} for anomaly detection.",
    "a photo of a {} for anomaly detection.",
]


def create_prompts(class_name: str = "object") -> tuple[list[str], list[str]]:
    normal_states = [state.format(class_name) for state in NORMAL_STATES]
    normal_prompts = [template.format(state) for state in normal_states for template in TEMPLATES]

    anomaly_states = [state.format(class_name) for state in ANOMALOUS_STATES]
    anomaly_prompts = [template.format(state) for state in anomaly_states for template in TEMPLATES]
    return normal_prompts, anomaly_prompts


def cosine_similarity(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    ndim = input1.ndim
    input1 = input1.unsqueeze(0) if input1.ndim == 2 else input1
    input2 = input2.repeat(input1.shape[0], 1, 1) if input2.ndim == 2 else input2

    input1_norm = nn.functional.normalize(input1, p=2, dim=-1)
    input2_norm = nn.functional.normalize(input2, p=2, dim=-1)
    similarity = torch.bmm(input1_norm, input2_norm.transpose(-2, -1))
    if ndim == 2:
        return similarity.squeeze(0)
    return similarity


def harmonic_aggregation(window_scores: torch.Tensor, output_size: tuple, masks: torch.Tensor) -> torch.Tensor:
    batch_size = window_scores.shape[0]
    height, width = output_size

    scores = []
    for idx in range(height * width):
        patch_mask = torch.any(masks == idx, dim=0)  # boolean tensor indicating which masks contain the patch
        scores.append(sum(patch_mask) / (1 / window_scores.T[patch_mask]).sum(dim=0))

    return torch.stack(scores).T.reshape(batch_size, height, width).nan_to_num(posinf=0.0)


def visual_association_score(embeddings: torch.Tensor, reference_embeds: torch.Tensor) -> torch.Tensor:
    scores = cosine_similarity(embeddings, reference_embeds.reshape(-1, embeddings.shape[-1]))
    return (1 - scores).min(dim=-1)[0] / 2


def make_masks(grid_size: tuple[int, int], kernel_size: int) -> torch.Tensor:
    if any(dim < kernel_size for dim in grid_size):
        raise ValueError(
            "Each dimension of the grid size must be greater than or equal to "
            f"the kernel size. Got grid size {grid_size} and kernel size {kernel_size}."
        )
    height, width = grid_size
    grid = torch.arange(height * width).reshape(1, height, width)
    return nn.functional.unfold(grid.float(), kernel_size=kernel_size, stride=1).int()


class BufferListDescriptor:
    def __init__(self, name: str, length: int):
        self.name = name
        self.length = length

    def __get__(self, instance, object_type=None) -> list[torch.Tensor]:
        del object_type
        return [getattr(instance, f"_{self.name}_{i}") for i in range(self.length)]

    def __set__(self, instance, values: list[torch.Tensor]):
        for i, value in enumerate(values):
            setattr(instance, f"_{self.name}_{i}", value)


class WinCLIP(nn.Module):
    def __init__(
        self,
        clip: CLIP | CustomTextCLIP | CoCa,
        tokenizer: SimpleTokenizer | HFTokenizer,
        scales: Sequence[int] = (2, 3),
        temperature: float = 0.07,
    ):
        super().__init__()

        self.clip = clip
        self.tokenizer = tokenizer
        self.scales = tuple(scales)
        self.temperature = temperature
        self.k_shot = 0

        self.clip.visual.output_tokens = True  # type: ignore
        self.grid_size: tuple[int, int] = self.clip.visual.grid_size

        self.register_buffer_list("masks", [make_masks(self.grid_size, s) for s in self.scales], persistent=False)
        self.register_buffer("_text_embeds", torch.empty(0))
        self.register_buffer_list("_visual_embeds", [torch.empty(0) for _ in self.scales])
        self.register_buffer("_patch_embeds", torch.empty(0))

        # Feature map hook
        self._outputs = []

        def _hook(_module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor):
            self._outputs.append(output.detach())

        self.clip.visual.patch_dropout.register_forward_hook(_hook)

    def register_buffer_list(self, name: str, values: list[torch.Tensor], persistent: bool = True):
        for i, value in enumerate(values):
            self.register_buffer(f"_{name}_{i}", value, persistent=persistent)
        setattr(self.__class__, name, BufferListDescriptor(name, len(values)))

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

        self._text_embeds = torch.stack((
            encode_text_ensemble(
                normal_prompts,
                encode_fn=self.clip.encode_text,
                tokenize_fn=self.tokenizer,  # type: ignore
                device=self.clip.logit_scale.device,
            ).mean(dim=0),
            encode_text_ensemble(
                anomaly_prompts,
                encode_fn=self.clip.encode_text,
                tokenize_fn=self.tokenizer,  # type: ignore
                device=self.clip.logit_scale.device,
            ).mean(dim=0),
        ))

    @torch.no_grad()
    def setup_images(self, reference_images: torch.Tensor):
        self.k_shot = reference_images.shape[0]
        _, self._visual_embeds, self._patch_embeds = self.encode_image(reference_images)

    def encode_image(self, batch: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        image_embeds, patch_embeds = self.clip.encode_image(batch)
        feature_map = self._outputs.pop()
        window_embeds = [self._get_window_embeds(feature_map, masks) for masks in self.masks]
        self._outputs.clear()
        return image_embeds, window_embeds, patch_embeds

    def _get_window_embeds(self, feature_map: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        batch_size = feature_map.shape[0]
        n_masks = masks.shape[1]

        class_index = torch.zeros(1, n_masks, dtype=torch.int).to(feature_map.device)
        masks = torch.cat((class_index, masks + 1)).T  # +1 to account for class index
        masked = torch.cat([torch.index_select(feature_map, 1, mask) for mask in masks])

        masked = self.clip.visual.ln_pre(masked)
        masked = self.clip.visual.transformer(masked)
        masked = self.clip.visual.ln_post(masked)
        pooled, _ = self.clip.visual._global_pool(masked)

        if self.clip.visual.proj is not None:
            pooled = pooled @ self.clip.visual.proj

        return pooled.reshape((n_masks, batch_size, -1)).permute(1, 0, 2)

    @torch.no_grad()
    def forward(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image_embeds, window_embeds, patch_embeds = self.encode_image(batch)

        image_scores = (cosine_similarity(image_embeds, self._text_embeds) / self.temperature).softmax(dim=-1)[..., 1]
        multi_scale_scores = self._compute_zero_shot_scores(image_scores, window_embeds)

        if self.k_shot:
            few_shot_scores = self._compute_few_shot_scores(patch_embeds, window_embeds)
            multi_scale_scores = (multi_scale_scores + few_shot_scores) / 2
            image_scores = (image_scores + few_shot_scores.amax(dim=(-2, -1))) / 2

        # reshape to image dimensions
        pixel_scores = interpolate(
            multi_scale_scores.unsqueeze(1),
            size=batch.shape[-2:],
            mode="bilinear",
        ).squeeze(1)

        return image_scores, pixel_scores

    def _compute_zero_shot_scores(
        self,
        image_scores: torch.Tensor,
        window_embeds: list[torch.Tensor],
    ) -> torch.Tensor:
        # image scores are added to represent the full image scale
        multi_scale_scores = [image_scores.view(-1, 1, 1).repeat(1, self.grid_size[0], self.grid_size[1])]
        # add aggregated scores for each scale
        for window_embed, mask in zip(window_embeds, self.masks, strict=True):
            scores = (cosine_similarity(window_embed, self._text_embeds) / self.temperature).softmax(dim=-1)[..., 1]
            multi_scale_scores.append(harmonic_aggregation(scores, self.grid_size, mask))
        # aggregate scores across scales
        return (len(self.scales) + 1) / (1 / torch.stack(multi_scale_scores)).sum(dim=0)

    def _compute_few_shot_scores(
        self,
        patch_embeds: torch.Tensor,
        window_embeds: list[torch.Tensor],
    ) -> torch.Tensor:
        multi_scale_scores = [
            visual_association_score(patch_embeds, self._patch_embeds).reshape((-1, *self.grid_size)),
        ] + [
            harmonic_aggregation(
                visual_association_score(window_embed, reference_embed),
                self.grid_size, mask,
            )
            for window_embed, reference_embed, mask in zip(
                window_embeds, self._visual_embeds, self.masks, strict=True,
            )
        ]
        return torch.stack(multi_scale_scores).mean(dim=0)


def load_winclip(
    backbone: str = "ViT-B-16-plus-240:laion400m_e31",
    scales: Sequence[int] = (2, 3),
    temperature: float = 0.07,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    # jit: bool = False,
    download_root: str | None = "./checkpoints/open_clip",
    **kwargs,
) -> tuple[WinCLIP, Callable[[Image], torch.Tensor]]:
    if ":" in backbone:
        if backbone not in open_clip.list_pretrained(as_str=True):
            raise RuntimeError(f"Model {backbone} not found")

        backbone, pretrained = backbone.split(":")
    else:
        raise ValueError("WinCLIP only supports open_clip's pretrained models")

    clip, _, transform = open_clip.create_model_and_transforms(
        backbone,
        pretrained=pretrained,
        device=device,
        # jit=jit,  # TODO
        cache_dir=download_root,
        **kwargs,
    )

    if not isinstance(clip.visual, VisionTransformer):
        raise ValueError("WinCLIP only supports open_clip's VisionTransformer visual encoder")

    tokenizer = open_clip.get_tokenizer(backbone)

    model = WinCLIP(
        clip=clip,  # type: ignore
        tokenizer=tokenizer,
        scales=scales,
        temperature=temperature,
    ).to(device)

    return model, transform  # type: ignore
