from typing import cast
from collections.abc import Callable, Sequence

import torch
from torch import nn
from torch import Tensor, IntTensor, LongTensor

from PIL.Image import Image

import open_clip
from open_clip import tokenize
from open_clip.model import CLIP as _CLIP
from open_clip.model import CustomTextCLIP as _CustomTextCLIP
from open_clip.coca_model import CoCa as _CoCa
from open_clip.tokenizer import SimpleTokenizer, HFTokenizer


def encode_text_ensemble(
    text: IntTensor | LongTensor | Sequence[IntTensor | LongTensor] | Sequence[str] | Sequence[Sequence[str]],
    encode_fn: Callable[[Tensor, bool], Tensor],
    tokenize_fn: Callable[[str | list[str]], IntTensor | LongTensor] = tokenize,
    *,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    normalize: bool = False,
) -> torch.Tensor:
    if isinstance(text, Sequence):
        if isinstance(text[0], str):  # Sequence[str]
            text = tokenize_fn(cast(list[str], list(text)))
        elif isinstance(text[0], Sequence):  # Sequence[Sequence[str]]
            text = [tokenize_fn(cast(list[str], list(t))) for t in text]
        else:
            raise ValueError("Invalid input type")

    text = cast(IntTensor | LongTensor | Sequence[IntTensor | LongTensor], text)

    if isinstance(text, Sequence):  # Sequence[IntTensor | LongTensor]
        return torch.stack([encode_fn(t.to(device), normalize).mean(dim=0) for t in text], dim=0)
    else:  # IntTensor | LongTensor
        return encode_fn(text.to(device), normalize)


class CLIPMixin:
    tokenizer: SimpleTokenizer | HFTokenizer
    logit_scale: nn.Parameter

    def encode_image(
        self,
        image: torch.Tensor,
        normalize: bool = False,
    ) -> torch.Tensor:
        image = image.to(self.logit_scale.device)
        return super().encode_image(image, normalize=normalize)  # type: ignore

    def encode_text(
        self,
        text: IntTensor | LongTensor | Sequence[IntTensor | LongTensor] | Sequence[str] | Sequence[Sequence[str]],
        normalize: bool = False
    ) -> Tensor:
        return encode_text_ensemble(
            text,
            encode_fn=super().encode_text,  # type: ignore
            tokenize_fn=self.tokenizer,  # type: ignore
            device=self.logit_scale.device,
            normalize=normalize,
        )


class CLIP(CLIPMixin, _CLIP):
    ...


class CustomTextCLIP(CLIPMixin, _CustomTextCLIP):
    ...


class CoCa(CLIPMixin, _CoCa):
    ...


def load_clip(
    name: str,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    # jit: bool = False,
    download_root: str | None = "./checkpoints/open_clip",
    **kwargs,
) -> tuple[CLIP | CustomTextCLIP | CoCa, Callable[[Image], Tensor]]:
    if ":" in name:
        if name not in open_clip.list_pretrained(as_str=True):
            raise RuntimeError(f"Model {name} not found")

        backbone, pretrained = name.split(":")
    else:
        backbone, pretrained = name, None

        if name not in open_clip.list_models():
            raise RuntimeError(f"Model {name} not found")

    model, _, transform = open_clip.create_model_and_transforms(
        backbone,
        pretrained=pretrained,
        device=device,
        # jit=jit,  # TODO
        cache_dir=download_root,
        **kwargs,
    )

    if isinstance(model, _CLIP):
        model.__class__ = CLIP
        model = cast(CLIP, model)
        if hasattr(model.visual, "output_tokens"):
            model.visual.output_tokens = False  # type: ignore

    elif isinstance(model, _CustomTextCLIP):
        model.__class__ = CustomTextCLIP
        model = cast(CustomTextCLIP, model)

    elif isinstance(model, _CoCa):
        model.__class__ = CoCa
        model = cast(CoCa, model)

    else:
        raise NotImplementedError(f"Model {name} not supported")

    model.tokenizer = open_clip.get_tokenizer(backbone)

    return model, transform  # type: ignore
