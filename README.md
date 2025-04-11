# LAFT

<br>

> [!WARNING]
> We pre-release the experiment code by request.
> Please use it for reference only.
> We are currently refactoring the code to make it usable for future work and will update the polished code at least _**within April 2025**_.
> README is also not fully updated.

<br>

**Language-Assisted Feature Transformation for Anomaly Detection** \
[EungGu Yun](https://github.com/yuneg11), Heonjin Ha, Yeongwoo Nam, Bryan Dongik Lee \
ICLR 2025

[[`Paper`](https://arxiv.org/abs/2503.01184)][[`ICLR`](https://openreview.net/forum?id=2p03KljxE9)][[`BibTeX`](#citation)]

<br>

## Installation

### Environment

```bash
conda create -n laft python=3.11
conda activate laft
```

### Dependencies

We recommend to install [PyTorch](https://pytorch.org/get-started/locally/) from the official website first.

```bash
pip install -r requirements.txt
```

We use:

- `torch==2.5.1`
- `torchvision==0.20.1`
- `torcheval==0.0.7`
- `open_clip_torch==2.29.0`

### Dataset

- MNIST: `data/MNIST`
- Waterbirds: `data/waterbirds_v1.0`
- CelebA: `data/celeba`
- [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad): `data/mvtec_anomaly_detection`
- [VisA](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar): `data/VisA_20220922`

### Checkpoints

[**CLIPN**](https://github.com/xmed-lab/CLIPN)

- [repeat1](https://drive.google.com/drive/folders/1CRIKr0vwrvK4Mc63zfhg2o8cbEGct4oF?usp=sharing): `checkpoints/clipn/repeat1.pt`
- [repeat2](https://drive.google.com/drive/folders/1eNaaPaRWz0La8_qQliX30A4I7Y44yDMY?usp=sharing): `checkpoints/clipn/repeat2.pt`
- [repeat3](https://drive.google.com/drive/folders/1qF4Pm1JSL3P0H4losPSmvldubFj91dew?usp=sharing): `checkpoints/clipn/repeat3.pt`

[**InCTRL**](https://github.com/mala-lab/InCTRL)

- [Google Drive](https://drive.google.com/drive/folders/1McmfxF8_H0BeRvcJ_poGIB-ATQCDDEIa?usp=sharing)
  - [MVTec AD](https://drive.google.com/file/d/1zEHsbbuUgBC4yuDu3g23wbUGmWmVyDRQ/view?usp=drive_link): `checkpoints/inctrl/mvtec{2,4,8}.pt`
  - [VisA](https://drive.google.com/file/d/1uDOnyRAlwtDukfhglR8YxidsnlezBD6I/view?usp=drive_link): `checkpoints/inctrl/visa{2,4,8}.pt`

## Usage

```python
import laft
import torch

torch.set_grad_enabled(False)  # disable Autograd (prevents OOM)

# assume image tensor is already loaded

# Load CLIP model and prompts
model, transform = laft.load_clip("ViT-B-16-quickgelu:dfn2b")
prompts = laft.prompts.get_prompts("color_mnist", "number")

# Encode image
image_features = model.encode_image(images)

# Construct concept subspace
text_features = model.encode_text(prompts["all"])
pair_diffs = laft.prompt_pair(features)
concept_basis = laft.pca(pair_diffs, n_components=24)

# Language-assisted feature transformation
guided_image_features = laft.inner(image_features, concept_basis)
ignored_image_features = laft.orthogonal(image_features, concept_basis)
```

See `runs/` directory for running scripts.

## Citation

```bibtex
@inproceedings{yun2025laft,
  title={Language-Assisted Feature Transformation for Anomaly Detection},
  author={EungGu Yun and Heonjin Ha and Yeongwoo Nam and Bryan Dongik Lee},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=2p03KljxE9}
}
```
