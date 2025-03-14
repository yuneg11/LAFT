import torch
from torch import Tensor
from torch.nn import functional as F


def inner(
    features: Tensor,  # [batch_size, feature_size]
    vectors: Tensor,   # [feature_size] or [num_vectors, feature_size]
    *,
    basis: bool = True,
) -> Tensor:           # [batch_size, feature_size]
    if vectors.dim() == 1:
        vectors = vectors.unsqueeze(dim=0)
        scales = torch.inner(features, vectors) / vectors.square().sum(dim=1)
        return scales @ vectors
    else:
        vector_basis = vectors if basis else torch.linalg.svd(vectors, full_matrices=False)[2]
        return (features @ vector_basis.T) @ vector_basis


def orthogonal(
    features: Tensor,  # [batch_size, feature_size]
    vectors: Tensor,   # [feature_size] or [num_vectors, feature_size]
    *,
    normalize: bool = False,
    basis: bool = True,
) -> Tensor:           # [batch_size, feature_size]
    if vectors.dim() == 1:
        vectors = vectors.unsqueeze(dim=0)

    vector_basis = vectors if basis else torch.linalg.svd(vectors, full_matrices=False)[2]
    proj = features - (features @ vector_basis.T) @ vector_basis

    if normalize:
        proj = F.normalize(proj, dim=-1)

    return proj


def cosine_similarity(x1: Tensor, x2: Tensor | None = None, eps: float = 1e-8) -> Tensor:
    x2 = x1 if x2 is None else x2
    x1_norm = x1 / torch.max(x1.norm(dim=-1, keepdim=True), eps * torch.ones_like(x1))
    x2_norm = x2 / torch.max(x2.norm(dim=-1, keepdim=True), eps * torch.ones_like(x2))
    if x1.ndim == 2:
        sim_matrix = torch.mm(x1_norm, x2_norm.transpose(-1, -2))
    else:
        sim_matrix = torch.bmm(x1_norm, x2_norm.transpose(-2, -1))
    return sim_matrix


def cosine_distance(x1: Tensor, x2: Tensor | None = None, eps: float = 1e-8) -> Tensor:
    return 1 - cosine_similarity(x1, x2, eps=eps)


def knn(
    train_features: Tensor,
    test_features: Tensor,
    *,
    n_neighbors: int = 30,
) -> Tensor:
    n_neighbors = min(n_neighbors, train_features.size(0))
    distance_matrix = cosine_distance(test_features, train_features)
    distances, _ = distance_matrix.topk(n_neighbors, largest=False)
    scores = distances.mean(dim=1)
    return scores


def pca(
    vectors: Tensor,
    n_components: int | None = None,
    *,
    center: bool = False,
    niter: int = 5,
) -> Tensor:
    min_d = min(vectors.size(0), vectors.size(1))
    d = min_d if n_components is None else min(n_components, min_d)
    components = torch.pca_lowrank(vectors, q=d, center=center, niter=niter)[2].T
    return components


def align_vectors(vectors: torch.Tensor, reference_idx: int = 0) -> Tensor:
    reference = vectors[reference_idx].half()
    sim = F.cosine_similarity(vectors.half(), reference)
    aligned = torch.where((sim < 0).unsqueeze(dim=1), -vectors, vectors)
    return aligned


def prompt_pair(*prompts_list: Tensor) -> Tensor:
    if len(prompts_list) == 1:
        prompts = prompts_list[0]
        length = prompts.size(0)
        idxs = torch.tensor([i * length + j for i in range(length) for j in range(i + 1, length)]).to(prompts.device)
        pairwise_diff = (prompts.unsqueeze(1) - prompts.unsqueeze(0)).flatten(0, 1).index_select(0, idxs)
        pairwise_diff = align_vectors(pairwise_diff)
        return pairwise_diff
    else:
        pairwise_diff = torch.cat([
            (prompts_list[i].unsqueeze(1) - prompts_list[j].unsqueeze(0)).flatten(0, 1)
            for i in range(len(prompts_list))
            for j in range(i + 1, len(prompts_list))
        ])
        pairwise_diff = align_vectors(pairwise_diff)
        return pairwise_diff
