# Prompts adapted from: https://github.com/FuNz-0/PromptAD/blob/master/PromptAD/ad_prompts.py

TEMPLATES = [
    "an image of a {}",
    "an image of the {}",
    "a photo of the {}.",
    "a photo of a {}.",
    "a bright photo of a {}.",
    "a bright photo of the {}.",
    "a dark photo of a {}.",
    "a dark photo of the {}.",
    "a jpeg corrupted photo of a {}.",
    "a jpeg corrupted photo of the {}.",
    "a blurry photo of the {}.",
    "a blurry photo of a {}.",
    "a photo of the {}.",
    "a photo of a {}.",
    "a photo of a {} for visual inspection.",
    "a photo of the {} for visual inspection.",
    "a photo of a {} for anomaly detection.",
    "a photo of the {} for anomaly detection.",
    "a detailed photo of a {}.",
    "a detailed photo of the {}.",
    "a zoomed-in photo of a {}.",
    "a zoomed-in photo of the {}.",
    "a side view of the {}.",
    "a top-down view of a {}.",
    "a high-resolution image of a {}.",
    "a high-resolution image of the {}.",
    "a close-up photo of a {}.",
    "a close-up photo of the {}.",
    "an inspection image of a {}.",
    "an inspection image of the {}.",
    "a grayscale photo of a {}.",
    "a grayscale photo of the {}.",
    "an overexposed photo of a {}.",
    "an overexposed photo of the {}.",
    "an underexposed photo of a {}.",
    "an underexposed photo of the {}.",
    "an artificially generated photo of a {}.",
    "an artificially generated photo of the {}.",
    "a distorted image of a {}.",
    "a distorted image of the {}.",
    "a noisy photo of a {}.",
    "a noisy photo of the {}.",
]
NORMAL_STATES = [
    "{}",
    "flawless {}",
    "perfect {}",
    "unblemished {}",
    "{} without flaw",
    "{} without defect",
    "{} without damage",
    "intact {}",
    "{} in perfect condition",
    "{} in excellent condition",
    "{} with no visible issues",
    "{} with no imperfections",
    "spotless {}",
    "clean and undamaged {}",
    "{} in optimal condition",
    "{} in ideal condition",
]
ANOMALY_STATES = [
    "damaged {}",
    "{} with flaw",
    "{} with defect",
    "{} with damage",
    "cracked {}",
    "scratched {}",
    "distorted {}",
    "{} with imperfections",
    "malformed {}",
    "{} with a tear",
    "faulty {}",
    "broken {}",
    "{} with visible damage",
    "{} with discoloration",
    "{} showing irregularities",
    "fractured {}",
    "{} with a scratch",
    "flawed {}",
    "abnormal {}",
    "imperfect {}",
    "blemished {}",
]
NORMAL_TEMPLATES = [f.format(s) for f in TEMPLATES for s in NORMAL_STATES]
ANOMALY_TEMPLATES = [f.format(s) for f in TEMPLATES for s in ANOMALY_STATES]

CATEGORY_MAPPING = {
    "metal_nut": "metal nut",
    "macaroni1": "macaroni",
    "macaroni2": "macaroni",
    "pcb1": "printed circuit board",
    "pcb2": "printed circuit board",
    "pcb3": "printed circuit board",
    "pcb4": "printed circuit board",
    "pipe_fryum": "pipe fryum",
    "chewinggum": "chewing gum",
}

CATEGORY_ANOMALY_STATES = {
    # MVTec AD
    "bottle": ["{} with large breakage", "{} with small breakage", "{} with contamination"],
    "cable": ["{} with bent wire", "{} with missing part", "{} with missing wire", "{} with cut", "{} with poke"],
    "capsule": ["{} with crack", "{} with faulty imprint", "{} with poke", "{} with scratch", "{} squeezed with compression"],
    "carpet": ["{} with hole", "{} with color stain", "{} with metal contamination", "{} with thread residue", "{} with thread", "{} with cut"],
    "grid": ["{} with breakage",  "{} with thread residue", "{} with thread", "{} with metal contamination", "{} with glue", "{} with a bent shape"],
    "hazelnut": ["{} with crack", "{} with cut", "{} with hole", "{} with print"],
    "leather": ["{} with color stain", "{} with cut", "{} with fold", "{} with glue", "{} with poke"],
    "metal_nut": ["{} with a bent shape ", "{} with color stain", "{} with a flipped orientation", "{} with scratch"],
    "pill": ["{} with color stain", "{} with contamination", "{} with crack", "{} with faulty imprint", "{} with scratch", "{} with abnormal type"],
    "screw": ["{} with manipulated front",  "{} with scratch neck", "{} with scratch head"],
    "tile": ["{} with crack", "{} with glue strip", "{} with gray stroke", "{} with oil", "{} with rough surface"],
    "toothbrush": ["{} with defect", "{} with anomaly"],
    "transistor": ["{} with bent lead", "{} with cut lead", "{} with damage", "{} with misplaced transistor"],
    "wood": ["{} with color stain", "{} with hole", "{} with scratch", "{} with liquid"],
    "zipper": ["{} with broken teeth", "{} with fabric border", "{} with defect fabric", "{} with broken fabric", "{} with split teeth", "{} with squeezed teeth"],
    # VisA
    "candle": ["{} with melded wax", "{} with foreign particals", "{} with extra wax", "{} with chunk of wax missing", "{} with weird candle wick", "{} with damaged corner of packaging", "{} with different colour spot"],
    "capsules": ["{} with scratch", "{} with discolor", "{} with misshape", "{} with leak", "{} with bubble"],
    "cashew": ["{} with breakage", "{} with small scratches", "{} with burnt", "{} with stuck together", "{} with spot"],
    "chewinggum": ["{} with corner missing", "{} with scratches", "{} with chunk of gum missing", "{} with colour spot", "{} with cracks"],
    "fryum": ["{} with breakage", "{} with scratches", "{} with burnt", "{} with colour spot", "{} with fryum stuck together", "{} with colour spot"],
    "macaroni1": ["{} with color spot", "{} with small chip around edge", "{} with small scratches", "{} with breakage", "{} with cracks"],
    "macaroni2": ["{} with color spot", "{} with small chip around edge", "{} with small scratches", "{} with breakage", "{} with cracks"],
    "pcb1": ["{} with bent", "{} with scratch", "{} with missing", "{} with melt"],
    "pcb2": ["{} with bent", "{} with scratch", "{} with missing", "{} with melt"],
    "pcb3": ["{} with bent", "{} with scratch", "{} with missing", "{} with melt"],
    "pcb4": ["{} with scratch", "{} with extra", "{} with missing", "{} with wrong place", "{} with damage", "{} with burnt", "{} with dirt"],
    "pipe_fryum": ["{} with breakage", "{} with small scratches", "{} with burnt", "{} with stuck together", "{} with colour spot", "{} with cracks"],
}


def get_prompts(category: str):
    category_templates = [f.format(s) for f in TEMPLATES for s in CATEGORY_ANOMALY_STATES[category]]
    category = CATEGORY_MAPPING.get(category, category)
    category = category.rstrip("0123456789").replace("_", " ")
    normal_prompts = [f.format(category) for f in NORMAL_TEMPLATES]
    anomaly_prompts = [f.format(category) for f in ANOMALY_TEMPLATES + category_templates]

    return normal_prompts, anomaly_prompts
