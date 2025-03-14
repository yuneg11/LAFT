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
NORMAL_STATES = [
    "{}",
    "flawless {}",
    "perfect {}",
    "unblemished {}",
    "{} without flaw",
    "{} without defect",
    "{} without damage",
]
ANOMALY_STATES = [
    "damaged {}",
    "{} with flaw",
    "{} with defect",
    "{} with damage",
]
NORMAL_TEMPLATES = [f.format(s) for f in TEMPLATES for s in NORMAL_STATES]
ANOMALY_TEMPLATES = [f.format(s) for f in TEMPLATES for s in ANOMALY_STATES]


def get_prompts(category: str):
    category = category.rstrip("0123456789").replace("_", " ")
    normal_prompts = [f.format(category) for f in NORMAL_TEMPLATES]
    anomaly_prompts = [f.format(category) for f in ANOMALY_TEMPLATES]
    return normal_prompts, anomaly_prompts
