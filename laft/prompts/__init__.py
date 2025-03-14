def get_labels(dataset_name, attrs, guidance: str):
    if dataset_name == "color_mnist":
        from .color_mnist import get_labels
    elif dataset_name == "waterbirds":
        from .waterbirds import get_labels
    elif dataset_name == "celeba":
        from .celeba import get_labels
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'")

    return get_labels(attrs, guidance)


def get_prompts(dataset_name, guidance: str):
    if not guidance.startswith("guide_") or not guidance.startswith("ignore_"):
        guidance = f"guide_{guidance}"

    if dataset_name == "color_mnist":
        from .color_mnist import get_prompts
    elif dataset_name == "waterbirds":
        from .waterbirds import get_prompts
    elif dataset_name == "celeba":
        from .celeba import get_prompts
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'")

    return get_prompts(guidance)


def get_words(dataset_name, guidance: str):
    if dataset_name == "color_mnist":
        from .color_mnist import get_words
    elif dataset_name == "waterbirds":
        from .waterbirds import get_words
    elif dataset_name == "celeba":
        from .celeba import get_words
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'")

    return get_words(guidance)
