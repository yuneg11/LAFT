from collections.abc import Sequence

import torch
from torch.nn import functional as F

from laft.clip import CLIP, CustomTextCLIP, CoCa


__all__ = [
    "clip_score",
]



def _similarity(
    model: CLIP | CustomTextCLIP | CoCa,
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    *,
    temperature: float | None = None,
    softmax: bool = True,
) -> torch.Tensor:
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    logit_scale = model.logit_scale.exp() if temperature is None else (1 / temperature)
    image_logits = logit_scale * image_features @ text_features.t()

    if model.logit_bias is not None:
        image_logits += model.logit_bias

    if softmax:
        return image_logits.softmax(dim=-1)
    else:
        return image_logits


def clip_score(
    model: CLIP | CustomTextCLIP | CoCa,
    images: torch.Tensor,  # images or image_features
    normal_prompts: Sequence[str] | Sequence[Sequence[str]],
    anomaly_prompts: Sequence[str] | Sequence[Sequence[str]] | None = None,
    *,
    temperature: float | None = 1.0,
) -> torch.Tensor:
    image_features = model.encode_image(images) if images.dim() == 4 else images
    normal_text_features = model.encode_text(normal_prompts)

    if anomaly_prompts is None:
        # MCM scoring
        normal_outputs = _similarity(model, image_features, normal_text_features, softmax=False)
        scores = -torch.amax(normal_outputs, dim=1)
    else:
        # ZOE scoring
        anomaly_text_features = model.encode_text(anomaly_prompts)
        text_features = torch.cat((normal_text_features, anomaly_text_features), dim=0)

        normal_and_anomaly_outputs = _similarity(model, image_features, text_features, temperature=temperature)
        anomaly_outputs = normal_and_anomaly_outputs[:, normal_text_features.size(0):]
        scores = torch.sum(anomaly_outputs, dim=1)

    return scores
