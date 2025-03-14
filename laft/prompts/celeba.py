DEFAULT_CONFIG = {
    "Blond_Hair": False,  # Blonde hair is normal (Blond: normal, No blond: anomaly)
    "Eyeglasses": True,   # Eyeglasses is anomaly (No eyeglasses: normal, Eyeglasses: anomaly)
}

BLOND_TEMPLATES = [
    "a photo of a person with {} hair",
    "a bad photo of a person with {} hair",
    "a good photo of a person with {} hair",
    "a small photo of a person with {} hair",
    "a big photo of a person with {} hair",
    "a photo of a potrait with {} hair",
    "a bad photo of a potrait with {} hair",
    "a good photo of a potrait with {} hair",
    "a small photo of a potrait with {} hair",
    "a big photo of a potrait with {} hair",
    "a photo of a face with {} hair",
    "a bad photo of a face with {} hair",
    "a good photo of a face with {} hair",
    "a small photo of a face with {} hair",
    "a big photo of a face with {} hair",
    "a photo of a man with {} hair",
    "a bad photo of a man with {} hair",
    "a good photo of a man with {} hair",
    "a small photo of a man with {} hair",
    "a big photo of a man with {} hair",
    "a photo of a woman with {} hair",
    "a bad photo of a woman with {} hair",
    "a good photo of a woman with {} hair",
    "a small photo of a woman with {} hair",
    "a big photo of a woman with {} hair",
]
GLASS_TEMPLATES = [
    "a photo of a person {}",
    "a bad photo of a person {}",
    "a good photo of a person {}",
    "a small photo of a person {}",
    "a big photo of a person {}",
    "a photo of a potrait {}",
    "a bad photo of a potrait {}",
    "a good photo of a potrait {}",
    "a small photo of a potrait {}",
    "a big photo of a potrait {}",
    "a photo of a face {}",
    "a bad photo of a face {}",
    "a good photo of a face {}",
    "a small photo of a face {}",
    "a big photo of a face {}",
    "a photo of a man {}",
    "a bad photo of a man {}",
    "a good photo of a man {}",
    "a small photo of a man {}",
    "a big photo of a man {}",
    "a photo of a woman {}",
    "a bad photo of a woman {}",
    "a good photo of a woman {}",
    "a small photo of a woman {}",
    "a big photo of a woman {}",
]

BLOND_WORDS = [
    ["blond", "blonde", "auburn"],
    ["black", "jet black"],
    ["brown", "ash brown", "dark brown"],
    ["gray", "grey", "silver", "ash gray", "silver gray"],
    ["red", "ginger", "burgundy"],
    ["white"],
    ["green", "mint"],
    ["blue", "cyan", "turquoise", "aquamarine", "teal"],
    ["purple", "lavender", "violet", "indigo"],
    ["pink", "magenta", "violet", "pastel pink"],
    ["orange", "yellow"],
]
GLASS_WORDS = [
    ["without glasses", "not wearing glasses"],
    ["without eyeglasses", "not wearing eyeglasses"],
    ["without sunglasses", "not wearing sunglasses"],
    ["with glasses", "wearing glasses", "glasses on"],
    ["with eyeglasses", "wearing eyeglasses", "eyeglasses on"],
    ["with sunglasses", "wearing sunglasses", "sunglasses on"],
]

WORD_BLOND_NORMAL = [v for w in BLOND_WORDS[:1] for v in w]
WORD_BLOND_ANOMALY = [v for w in BLOND_WORDS[1:] for v in w]
PROMPT_BLOND_NORMAL = [[f.format(v) for f in BLOND_TEMPLATES] for v in WORD_BLOND_NORMAL]
PROMPT_BLOND_ANOMALY = [[f.format(v) for f in BLOND_TEMPLATES] for v in WORD_BLOND_ANOMALY]

WORD_GLASS_NORMAL = [v for w in GLASS_WORDS[:3] for v in w]
WORD_GLASS_ANOMALY = [v for w in GLASS_WORDS[3:] for v in w]
PROMPT_GLASS_NORMAL = [[f.format(v) for f in GLASS_TEMPLATES] for v in WORD_GLASS_NORMAL]
PROMPT_GLASS_ANOMALY = [[f.format(v) for f in GLASS_TEMPLATES] for v in WORD_GLASS_ANOMALY]


def get_labels(attrs, guidance: str):
    if guidance == "guide_blond" or guidance == "ignore_glass":
        attend_name, ignore_name = "blond", "glass"
        attend_labels, ignore_labels = attrs[:, 0], attrs[:, 1]
    elif guidance == "guide_glass" or guidance == "ignore_blond":
        attend_name, ignore_name = "glass", "blond"
        attend_labels, ignore_labels = attrs[:, 1], attrs[:, 0]
    else:
        raise ValueError(f"Invalid guidance: {guidance}")

    return attend_name, ignore_name, attend_labels, ignore_labels


def get_prompts(guidance: str):
    if guidance == "guide_blond" or guidance == "ignore_blond":
        prompts = {
            "normal": PROMPT_BLOND_NORMAL,
            "anomaly": PROMPT_BLOND_ANOMALY,
            "half": None,
            "exact": None,
            "all": PROMPT_BLOND_NORMAL + PROMPT_BLOND_ANOMALY,
        }
    elif guidance == "guide_glass" or guidance == "ignore_glass":
        prompts = {
            "normal": PROMPT_GLASS_NORMAL,
            "anomaly": PROMPT_GLASS_ANOMALY,
            "half": None,
            "exact": None,
            "all": PROMPT_GLASS_NORMAL + PROMPT_GLASS_ANOMALY,
        }
    else:
        raise ValueError(f"Invalid guidance: {guidance}")

    return prompts


def get_words(guidance: str):
    if guidance == "guide_blond" or guidance == "ignore_blond":
        prompts = {
            "normal": WORD_BLOND_NORMAL,
            "anomaly": WORD_BLOND_ANOMALY,
            "half": None,
            "exact": None,
            "all": WORD_BLOND_NORMAL + WORD_BLOND_ANOMALY,
        }
    elif guidance == "guide_glass" or guidance == "ignore_glass":
        prompts = {
            "normal": WORD_GLASS_NORMAL,
            "anomaly": WORD_GLASS_ANOMALY,
            "half": None,
            "exact": None,
            "all": WORD_GLASS_NORMAL + WORD_GLASS_ANOMALY,
        }
    else:
        raise ValueError(f"Invalid guidance: {guidance}")

    return prompts
