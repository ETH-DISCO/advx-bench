import csv
import itertools
import json
import random
from pathlib import Path
from typing import Generator, Optional

import torch
import torchvision.transforms as transforms
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from advx.masks import get_circle_mask, get_diamond_mask, get_knit_mask, get_square_mask, get_word_mask
from advx.utils import add_overlay
from metrics.metrics import get_cosine_similarity, get_fid, get_inception_features, get_kid, get_psnr, get_ssim
from models.cls import classify_clip


def get_imagenet_generator(size: int, seed: Optional[int] = None) -> Generator:
    if seed is None:
        seed = random.randint(0, 1000)
    subset = load_dataset("visual-layer/imagenet-1k-vl-enriched", split="validation", streaming=True).take(size).shuffle(seed=seed)  # type: ignore
    for elem in subset:
        yield elem["image_id"], elem["image"].convert("RGB"), elem["label"], elem["caption_enriched"]  # type: ignore


def get_imagenet_label(idx: int) -> str:
    datapath = Path.cwd() / "data" / "imagenet_labels.json"
    data = json.loads(datapath.read_text())
    return data[str(idx)]


def get_imagenet_labels() -> list[str]:
    datapath = Path.cwd() / "data" / "imagenet_labels.json"
    data = json.loads(datapath.read_text())
    return list(data.values())


def get_advx(img: Image.Image, combination: dict) -> Image.Image:
    if combination["mask"] == "diamond":
        img = add_overlay(img, get_diamond_mask(), opacity=combination["opacity"])
    elif combination["mask"] == "word":
        img = add_overlay(img, get_word_mask(words=get_imagenet_labels()), opacity=combination["opacity"])
    elif combination["mask"] == "circle":
        img = add_overlay(img, get_circle_mask(), opacity=combination["opacity"])
    elif combination["mask"] == "knit":
        img = add_overlay(img, get_knit_mask(), opacity=combination["opacity"])
    elif combination["mask"] == "square":
        img = add_overlay(img, get_square_mask(), opacity=combination["opacity"])
    else:
        raise ValueError(f"Unknown mask: {combination['mask']}")
    return img


"""
config
"""


CONFIG = {
    "outpath": Path.cwd() / "data" / "eval" / "eval_cls.csv",
    "fidkidpath": Path.cwd() / "data" / "eval" / "eval_cls_fidkid.csv",
    "subset_size": 5,
}
CONFIG["outpath"].unlink(missing_ok=True)
CONFIG["fidkidpath"].unlink(missing_ok=True)


COMBINATIONS = {
    "mask": ["circle", "square", "diamond", "knit", "word"],
    "opacity": [0, 64, 128, 192, 255],  # range from 0 (transparent) to 255 (opaque)
}


"""
eval loop
"""


random_combinations = list(itertools.product(*COMBINATIONS.values()))
random.shuffle(random_combinations)
print(f"total iterations: {len(random_combinations)} * {CONFIG['subset_size']} = {len(random_combinations) * CONFIG['subset_size']}")

for i, combination in enumerate(random_combinations):
    print(f">>> iteration {i+1}/{len(random_combinations)}")
    combination = dict(zip(COMBINATIONS.keys(), combination))

    dataset = get_imagenet_generator(size=CONFIG["subset_size"]) # note: you shouldn't use different subsets for each combination
    labels = get_imagenet_labels()

    x_features = []
    advx_features = []

    # example subset for this combination --------------------

    for id, x_image, label_id, caption in tqdm(dataset, total=CONFIG["subset_size"]):
        # get advx --------------------------------------------

        advx_image = get_advx(x_image, combination)

        # compare x and advx ----------------------------------

        transform = transforms.Compose([transforms.Resize((256, 256)), transforms.Grayscale(num_output_channels=3), transforms.ToTensor()])
        x: torch.Tensor = transform(x_image).unsqueeze(0)
        advx_x: torch.Tensor = transform(advx_image).unsqueeze(0)

        x_features.append(get_inception_features(x))
        advx_features.append(get_inception_features(advx_x))

        def get_acc_boolmask(img: Image.Image) -> list[bool]:
            preds = list(zip(range(len(labels)), classify_clip(img, labels)))  # most adversarially robust model model based on the RoZ paper
            preds.sort(key=lambda x: x[1], reverse=True)
            top5_keys, top5_vals = zip(*preds[:5])
            top5_mask = [label_id == key for key in top5_keys]
            return top5_mask

        x_acc5 = get_acc_boolmask(x_image)
        advx_acc5 = get_acc_boolmask(advx_image)

        results = {
            # settings
            **combination,
            # "subset_size": CONFIG["subset_size"],
            # semantic similarity
            "cosine_sim": get_cosine_similarity(x, advx_x),
            "psnr": get_psnr(x, advx_x),
            "ssim": get_ssim(x, advx_x),
            # accuracy
            "img_id": id,
            "label": get_imagenet_label(label_id),
            # "caption": caption.replace("\n", ""),
            "x_acc1": 1 if x_acc5[0] else 0,
            "advx_acc1": 1 if advx_acc5[0] else 0,
            "x_acc5": 1 if any(x_acc5) else 0,
            "advx_acc5": 1 if any(advx_acc5) else 0,
        }

        with open(CONFIG["outpath"], mode="a") as f:
            writer = csv.DictWriter(f, fieldnames=results.keys())
            if CONFIG["outpath"].stat().st_size == 0:
                writer.writeheader()
            writer.writerow(results)

    # get fid/kid for this combination ------------------------

    with open(CONFIG["fidkidpath"], mode="a") as f:
        metrics = {
            **combination,
            "fid": get_fid(x_features, x_features),
            "kid": get_kid(advx_features, advx_features, CONFIG["subset_size"]),
        }
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        if CONFIG["fidkidpath"].stat().st_size == 0:
            writer.writeheader()
        writer.writerow(metrics)
