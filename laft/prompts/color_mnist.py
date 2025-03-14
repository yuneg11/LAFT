DATASET_CONFIG = {
    "number": {
        0: False,  # Normal
        1: False,  # Normal
        2: False,  # Normal
        3: False,  # Normal
        4: False,  # Normal
        5: True,   # Anomaly
        6: True,   # Anomaly
        7: True,   # Anomaly
        8: True,   # Anomaly
        9: True,   # Anomaly
    },
    "color": {
        "red": False,   # Normal
        "green": True,  # Anomaly
        "blue": True,   # Anomaly
    },
}

TEMPLATES = [
    "{}",
    "a number {}",
    "an image of {}",
    "a picture of {}",
    "a photo of {}",
    "a drawing of {}",
    "a sketch of {}",
    "a figure of {}",
    "{} letter",
    "a number {} letter",
    "a image of {} letter",
    "a picture of {} letter",
    "a photo of {} letter",
    "a drawing of {} letter",
    "a sketch of {} letter",
    "a figure of {} letter",
    "a letter of {}",
    "a letter of number {}",
    "a photo of the number: '{}'",
]

NUMBER_WORDS = [
    ["zero", "0"],
    ["one", "1"],
    ["two", "2"],
    ["three", "3"],
    ["four", "4"],
    ["five", "5"],
    ["six", "6"],
    ["seven", "7"],
    ["eight", "8"],
    ["nine", "9"],
    ["ten", "10"],
    ["eleven", "11"],
    ["twelve", "12"],
    ["thirteen", "13"],
    ["fourteen", "14"],
    ["fifteen", "15"],
    ["sixteen", "16"],
    ["seventeen", "17"],
    ["eighteen", "18"],
    ["nineteen", "19"],
    ["twenty", "20"],
]
WORD_NUMBER_NORMAL = [v for w in NUMBER_WORDS[:5] for v in w]
WORD_NUMBER_ANOMALY = [v for w in NUMBER_WORDS[5:10] for v in w]
WORD_NUMBER_AUXILIARY = [v for w in NUMBER_WORDS[10:] for v in w]
PROMPT_NUMBER_NORMAL = [[f.format(v) for f in TEMPLATES] for v in WORD_NUMBER_NORMAL]
PROMPT_NUMBER_ANOMALY = [[f.format(v) for f in TEMPLATES] for v in WORD_NUMBER_ANOMALY]
PROMPT_NUMBER_AUXILIARY = [[f.format(v) for f in TEMPLATES] for v in WORD_NUMBER_AUXILIARY]

COLOR_WORDS = [
    ["red", "ruby", "scarlet", "crimson", "maroon", "carmine", "vermilion"],
    ["green", "lime", "olive", "jade"],
    ["blue", "azure", "sky blue", "navy"],
    ["yellow", "gold", "amber", "lemon"],
    ["orange", "titian", "coral"],
    ["purple", "violet", "lavender", "lilac", "mauve", "plum"],
    ["pink", "rose", "magenta", "fuchsia"],
    ["brown", "tan", "sepia", "beige"],
    ["black", "ebony", "sable", "jet"],
    ["white", "ivory", "snow", "chalk", "pearl", "cream"],
]
WORD_COLOR_NORMAL = [v for w in COLOR_WORDS[:1] for v in w]
WORD_COLOR_ANOMALY = [v for w in COLOR_WORDS[1:3] for v in w]
WORD_COLOR_AUXILIARY = [v for w in COLOR_WORDS[3:] for v in w]
PROMPT_COLOR_NORMAL = [[f.format(v) for f in TEMPLATES] for v in WORD_COLOR_NORMAL]
PROMPT_COLOR_ANOMALY = [[f.format(v) for f in TEMPLATES] for v in WORD_COLOR_ANOMALY]
PROMPT_COLOR_AUXILIARY = [[f.format(v) for f in TEMPLATES] for v in WORD_COLOR_AUXILIARY]


def get_labels(attrs, guidance: str):
    if guidance == "guide_number" or guidance == "ignore_color":
        attend_name, ignore_name = "number", "color"
        attend_labels, ignore_labels = attrs[:, 0], attrs[:, 1]
    elif guidance == "guide_color" or guidance == "ignore_number":
        attend_name, ignore_name = "color", "number"
        attend_labels, ignore_labels = attrs[:, 1], attrs[:, 0]
    else:
        raise ValueError(f"Invalid guidance: {guidance}")

    return attend_name, ignore_name, attend_labels, ignore_labels


def get_prompts(guidance: str):
    if guidance == "guide_number" or guidance == "ignore_number":
        prompts = {
            "normal": PROMPT_NUMBER_NORMAL,
            "anomaly": PROMPT_NUMBER_ANOMALY + PROMPT_NUMBER_AUXILIARY,
            "half": PROMPT_NUMBER_NORMAL + PROMPT_NUMBER_ANOMALY[:len(PROMPT_NUMBER_ANOMALY) // 2],
            "exact": PROMPT_NUMBER_NORMAL + PROMPT_NUMBER_ANOMALY,
            "all": PROMPT_NUMBER_NORMAL + PROMPT_NUMBER_ANOMALY + PROMPT_NUMBER_AUXILIARY,
        }
    elif guidance == "guide_color" or guidance == "ignore_color":
        prompts = {
            "normal": PROMPT_COLOR_NORMAL,
            "anomaly": PROMPT_COLOR_ANOMALY + PROMPT_COLOR_AUXILIARY,
            "half": PROMPT_COLOR_NORMAL + PROMPT_COLOR_ANOMALY[:len(PROMPT_COLOR_ANOMALY) // 2],
            "exact": PROMPT_COLOR_NORMAL + PROMPT_COLOR_ANOMALY,
            "all": PROMPT_COLOR_NORMAL + PROMPT_COLOR_ANOMALY + PROMPT_COLOR_AUXILIARY,
        }
    else:
        raise ValueError(f"Invalid guidance: {guidance}")

    if guidance.startswith("ignore"):
        prompts = {k: [v for w in prompts[k] for v in w] for k in prompts}

    return prompts


def get_words(guidance: str):
    if guidance == "guide_number" or guidance == "ignore_number":
        prompts = {
            "normal": WORD_NUMBER_NORMAL,
            "anomaly": WORD_NUMBER_ANOMALY + WORD_NUMBER_AUXILIARY,
            "half": WORD_NUMBER_NORMAL + WORD_NUMBER_ANOMALY[:len(WORD_NUMBER_ANOMALY) // 2],
            "exact": WORD_NUMBER_NORMAL + WORD_NUMBER_ANOMALY,
            "all": WORD_NUMBER_NORMAL + WORD_NUMBER_ANOMALY + WORD_NUMBER_AUXILIARY,
        }
    elif guidance == "guide_color" or guidance == "ignore_color":
        prompts = {
            "normal": WORD_COLOR_NORMAL,
            "anomaly": WORD_COLOR_ANOMALY + WORD_COLOR_AUXILIARY,
            "half": WORD_COLOR_NORMAL + WORD_COLOR_ANOMALY[:len(WORD_COLOR_ANOMALY) // 2],
            "exact": WORD_COLOR_NORMAL + WORD_COLOR_ANOMALY,
            "all": WORD_COLOR_NORMAL + WORD_COLOR_ANOMALY + WORD_COLOR_AUXILIARY,
        }
    else:
        raise ValueError(f"Invalid guidance: {guidance}")

    if guidance.startswith("ignore"):
        prompts = {k: [v for w in prompts[k] for v in w] for k in prompts}

    return prompts
