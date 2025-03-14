# Code adapted from: https://github.com/xmed-lab/CLIPN
# License: MIT


from typing import Callable, Literal, overload
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from PIL.Image import Image

from open_clip.factory import create_model_and_transforms
from open_clip.tokenizer import tokenize
from open_clip.model import CLIP


PROMPT_LIST = [
    "'a bad photo of a {}.'",
    "'a photo of many {}.'",
    "'a sculpture of a {}.'",
    "'a low resolution photo of the {}.'",
    "'a rendering of a {}.'",
    "'graffiti of a {}.'",
    "'a bad photo of the {}.'",
    "'a cropped photo of the {}.'",
    "'a tattoo of a {}.'",
    "'a bright photo of a {}.'",
    "'a photo of a clean {}.'",
    "'a photo of a dirty {}.'",
    "'a dark photo of the {}.'",
    "'a drawing of a {}.'",
    "'a photo of my {}.'",
    "'a photo of the cool {}.'",
    "'a close-up photo of a {}.'",
    "'a black and white photo of the {}.'",
    "'a painting of the {}.'",
    "'a painting of a {}.'",
    "'a pixelated photo of the {}.'",
    "'a sculpture of the {}.'",
    "'a bright photo of the {}.'",
    "'a cropped photo of a {}.'",
    "'a photo of the dirty {}.'",
    "'a jpeg corrupted photo of a {}.'",
    "'a blurry photo of the {}.'",
    "'a photo of the {}.'",
    "'a good photo of the {}.'",
    "'a rendering of the {}.'",
    "'a {} in a video game.'",
    "'a photo of one {}.'",
    "'a doodle of a {}.'",
    "'a close-up photo of the {}.'",
    "'a photo of a {}.',",
    "'the {} in a video game.'",
    "'a sketch of a {}.'",
    "'a doodle of the {}.'",
    "'a low resolution photo of a {}.'",
    "'a rendition of the {}.'",
    "'a photo of the clean {}.'",
    "'a photo of a large {}.'",
    "'a rendition of a {}.'",
    "'a photo of a nice {}.'",
    "'a photo of a weird {}.'",
    "'a blurry photo of a {}.'",
    "'art of a {}.'",
    "'a sketch of the {}.'",
    "'a pixelated photo of a {}.'",
    "'itap of the {}.'",
    "'a jpeg corrupted photo of the {}.'",
    "'a good photo of a {}.'",
    "'a photo of the nice {}.'",
    "'a photo of the small {}.'",
    "'a photo of the weird {}.'",
    "'a drawing of the {}.'",
    "'a photo of the large {}.'",
    "'a black and white photo of a {}.'",
    "'a dark photo of a {}.'",
    "'itap of a {}.'",
    "'graffiti of the {}.'",
    "'a photo of a cool {}.'",
    "'a photo of a small {}.'",
    "'a tattoo of the {}.'",
    "'a bad photo with a {}.'",
    "'a photo with many {}.'",
    "'a sculpture with a {}.'",
    "'a low resolution photo with the {}.'",
    "'a rendering with a {}.'",
    "'graffiti with a {}.'",
    "'a bad photo with the {}.'",
    "'a cropped photo with the {}.'",
    "'a tattoo with a {}.'",
    "'a bright photo with a {}.'",
    "'a photo with a clean {}.'",
    "'a photo with a dirty {}.'",
    "'a dark photo with the {}.'",
    "'a drawing with a {}.'",
    "'a photo with my {}.'",
    "'a photo with the cool {}.'",
    "'a close-up photo with a {}.'",
    "'a black and white photo with the {}.'",
    "'a painting with the {}.'",
    "'a painting with a {}.'",
    "'a pixelated photo with the {}.'",
    "'a sculpture with the {}.'",
    "'a bright photo with the {}.'",
    "'a cropped photo with a {}.'",
    "'a photo with the dirty {}.'",
    "'a jpeg corrupted photo with a {}.'",
    "'a blurry photo with the {}.'",
    "'a photo with the {}.'",
    "'a good photo with the {}.'",
    "'a rendering with the {}.'",
    "'a {} appearing in a video game.'",
    "'a photo with one {}.'",
    "'a doodle with a {}.'",
    "'a close-up photo with the {}.'",
    "'a photo with a {}.',",
    "'the {} appearing in a video game.'",
    "'a sketch with a {}.'",
    "'a doodle with the {}.'",
    "'a low resolution photo with a {}.'",
    "'a rendition with the {}.'",
    "'a photo with the clean {}.'",
    "'a photo with a large {}.'",
    "'a rendition with a {}.'",
    "'a photo with a nice {}.'",
    "'a photo with a weird {}.'",
    "'a blurry photo with a {}.'",
    "'art with a {}.'",
    "'a sketch with the {}.'",
    "'a pixelated photo with a {}.'",
    "'itap with the {}.'",
    "'a jpeg corrupted photo with the {}.'",
    "'a good photo with a {}.'",
    "'a photo with the nice {}.'",
    "'a photo with the small {}.'",
    "'a photo with the weird {}.'",
    "'a drawing with the {}.'",
    "'a photo with the large {}.'",
    "'a black and white photo with a {}.'",
    "'a dark photo with a {}.'",
    "'itap with a {}.'",
    "'graffiti with the {}.'",
    "'a photo with a cool {}.'",
    "'a photo with a small {}.'",
    "'a tattoo with the {}.'",
]


class CLIPN(nn.Module):
    def __init__(self, clip: CLIP, logit_scale: float = 100.0):
        super().__init__()
        self.logit_scale = logit_scale

        # Visual
        self.visual = clip.visual

        # Text
        self.context_length = clip.context_length
        self.vocab_size = clip.vocab_size

        # Text (Yes)
        self.transformer = clip.transformer
        self.token_embedding = clip.token_embedding
        self.positional_embedding = clip.positional_embedding
        self.ln_final = clip.ln_final
        self.text_projection = clip.text_projection

        # Text (No)
        self.transformer_no = deepcopy(clip.transformer)
        self.token_embedding_no = deepcopy(clip.token_embedding)
        self.positional_embedding_no = deepcopy(clip.positional_embedding)
        self.ln_final_no = deepcopy(clip.ln_final)
        self.text_projection_no = deepcopy(clip.text_projection)

        self.prompt_no = nn.Parameter(torch.empty(self.context_length, 512))  # text_cfg.width = 512 (ViT-B-16/32)
        self.register_buffer("attn_mask", clip.attn_mask, persistent=False)

        self._weight_yes = None
        self._weight_no = None

    @property
    def device(self):
        return self.attn_mask.device

    def encode_text(self, text, yes: bool = True):
        text = text.to(self.device)

        if yes:
            x = self.token_embedding(text) + self.positional_embedding
            x = self.transformer(x, attn_mask=self.attn_mask)
            x = self.ln_final(x)
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        else:
            x = self.token_embedding_no(text) + self.positional_embedding_no + self.prompt_no
            x = self.transformer_no(x, attn_mask=self.attn_mask)
            x = self.ln_final_no(x)
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection_no

        x = F.normalize(F.normalize(x, dim=-1).sum(dim=0), dim=-1)
        return x

    @overload
    def setup_prompts(self, prompts: list[list[str]]):
        ...

    @overload
    def setup_prompts(self, *, words: list[str]):
        ...

    def setup_prompts(
        self,
        prompts: list[list[str]] | None = None,
        *,
        words: list[str] | None = None,
    ):
        if prompts is not None and words is None:
            texts = [
                tokenize(_prompts, self.context_length)
                for _prompts in prompts
            ]
        elif words is not None and prompts is None:
            texts = [
                tokenize([p.format(word) for p in PROMPT_LIST], self.context_length)
                for word in words
            ]
        else:
            raise ValueError("Only one of prompts or words should be provided")

        self._weight_yes = torch.stack([self.encode_text(text, yes=True) for text in texts])
        self._weight_no = torch.stack([self.encode_text(text, yes=False) for text in texts])

    def compute_logits(self, images: torch.Tensor):
        assert self._weight_yes is not None and self._weight_no is not None

        inputs = self.visual(images)
        inputs_norm = F.normalize(inputs, dim=-1)

        logits_yes = self.logit_scale * inputs_norm @ self._weight_yes.T
        logits_no = self.logit_scale * inputs_norm @ self._weight_no.T
        return logits_yes, logits_no

    def forward(self, images: torch.Tensor, score: Literal["ctw", "atd", "all"] = "atd"):
        logits_yes, logits_no = self.compute_logits(images)

        logits = torch.stack((logits_yes, logits_no))
        probs = torch.softmax(logits, dim=0)[0]

        idx = torch.argmax(logits_yes, dim=1, keepdim=True)
        ctw_score = 1 - torch.gather(probs, dim=1, index=idx).squeeze(1)
        atd_score = 1 - (probs * torch.softmax(logits_yes, dim=1)).sum(1)

        if score == "ctw":
            return ctw_score
        elif score == "atd":
            return atd_score
        elif score == "all":
            return ctw_score, atd_score
        else:
            raise ValueError(f"Unknown score type: {score}")


def load_clipn(
    checkpoint: str,
    logit_scale: float = 100.0,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    **kwargs,
) -> tuple[CLIPN, Callable[[Image], torch.Tensor]]:

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = {k.removeprefix("module."): v for k, v in ckpt["state_dict"].items()}
    state_dict["prompt_no"] = torch.mean(state_dict["prompt_no"], dim=0)
    del state_dict["logit_scale"]

    clip, _, transform = create_model_and_transforms(
        model_name="ViT-B-16",
        device=device,
        **kwargs,
    )

    model = CLIPN(clip, logit_scale=logit_scale)  # type: ignore
    model.load_state_dict(state_dict)
    model.to(device).eval()

    return model, transform  # type: ignore
