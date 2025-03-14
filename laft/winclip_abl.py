from collections.abc import Callable, Sequence

import torch
from torch.nn.functional import interpolate

import open_clip
from open_clip.transformer import VisionTransformer

from PIL.Image import Image

from baselines.winclip import WinCLIP, cosine_similarity, visual_association_score, harmonic_aggregation


def inner(features: torch.Tensor, vectors: torch.Tensor):
    return features @ vectors.T


class WinCLIPwLAFT(WinCLIP):
    def get_ref_embeds(self):
        return (
            self._text_embeds.detach(),
            [v.detach() for v in self._visual_embeds] if self.k_shot else None,
            self._patch_embeds.detach() if self.k_shot else None,
        )

    @torch.no_grad()
    def scoring(
        self,
        batch: torch.Tensor,  # only used for shape information
        # image-text
        it_image_embeds: torch.Tensor,
        it_ref_text_embeds: torch.Tensor,
        # window-text
        wt_window_embeds: list[torch.Tensor],
        wt_ref_text_embeds: torch.Tensor,
        # window-window
        ww_window_embeds: list[torch.Tensor] | None = None,
        ww_ref_window_embeds: list[torch.Tensor] | None = None,
        # patch-patch
        pp_patch_embeds: torch.Tensor | None = None,
        pp_ref_patch_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        image_scores = (cosine_similarity(it_image_embeds, it_ref_text_embeds) / self.temperature).softmax(dim=-1)[..., 1]
        multi_scale_scores = self._compute_zero_shot_scores(image_scores, wt_window_embeds, wt_ref_text_embeds)

        if self.k_shot:
            assert pp_patch_embeds is not None and pp_ref_patch_embeds is not None and ww_window_embeds is not None and ww_ref_window_embeds is not None
            few_shot_scores = self._compute_few_shot_scores(pp_patch_embeds, ww_window_embeds, pp_ref_patch_embeds, ww_ref_window_embeds)
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
        ref_text_embeds: torch.Tensor,
    ) -> torch.Tensor:
        # image scores are added to represent the full image scale
        multi_scale_scores = [image_scores.view(-1, 1, 1).repeat(1, self.grid_size[0], self.grid_size[1])]
        # add aggregated scores for each scale
        for window_embed, mask in zip(window_embeds, self.masks, strict=True):
            scores = (cosine_similarity(window_embed, ref_text_embeds) / self.temperature).softmax(dim=-1)[..., 1]
            multi_scale_scores.append(harmonic_aggregation(scores, self.grid_size, mask))
        # aggregate scores across scales
        return (len(self.scales) + 1) / (1 / torch.stack(multi_scale_scores)).sum(dim=0)

    def _compute_few_shot_scores(
        self,
        patch_embeds: torch.Tensor,
        window_embeds: list[torch.Tensor],
        ref_patch_embeds: torch.Tensor,
        ref_window_embeds: list[torch.Tensor],
    ) -> torch.Tensor:
        multi_scale_scores = [
            visual_association_score(patch_embeds, ref_patch_embeds).reshape((-1, *self.grid_size)),
        ] + [
            harmonic_aggregation(
                visual_association_score(window_embed, reference_embed),
                self.grid_size, mask,
            )
            for window_embed, reference_embed, mask in zip(
                window_embeds, ref_window_embeds, self.masks, strict=True,
            )
        ]
        return torch.stack(multi_scale_scores).mean(dim=0)


def load_winclip_laft(
    backbone: str = "ViT-B-16-plus-240:laion400m_e31",
    scales: Sequence[int] = (2, 3),
    temperature: float = 0.07,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    # jit: bool = False,
    download_root: str | None = "./checkpoints/open_clip",
    **kwargs,
) -> tuple[WinCLIPwLAFT, Callable[[Image], torch.Tensor]]:
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
        raise ValueError("WinCLIPwLAFT only supports open_clip's VisionTransformer visual encoder")

    tokenizer = open_clip.get_tokenizer(backbone)

    model = WinCLIPwLAFT(
        clip=clip,  # type: ignore
        tokenizer=tokenizer,
        scales=scales,
        temperature=temperature,
    ).to(device)

    return model, transform  # type: ignore
