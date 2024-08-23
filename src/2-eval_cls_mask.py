import csv
import itertools
import json
import random
from pathlib import Path

import torch
import torchvision.transforms as transforms
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from advx.masks import get_circle_mask, get_diamond_mask, get_knit_mask, get_square_mask, get_word_mask
from advx.utils import add_overlay
from metrics.metrics import get_cosine_similarity, get_psnr, get_ssim
from models.cls import classify_clip


def get_imagenet_label(idx: int) -> str:
    datapath = Path.cwd() / "data" / "imagenet_labels.json"
    data = json.loads(datapath.read_text())
    return data[str(idx)]


def get_imagenet_labels() -> list[str]:
    datapath = Path.cwd() / "data" / "imagenet_labels.json"
    data = json.loads(datapath.read_text())
    return list(data.values())


def get_advx(img: Image.Image, combination: dict) -> Image.Image:
    density = combination["density"]
    if combination["mask"] == "diamond":
        density = int(density / 10)  # 1 -> 10 (count per row)
        img = add_overlay(img, get_diamond_mask(), opacity=combination["opacity"])

    elif combination["mask"] == "circle":
        density = int(density / 10)  # 1 -> 10 (count per row)
        img = add_overlay(img, get_circle_mask(), opacity=combination["opacity"])

    elif combination["mask"] == "square":
        density = int(density / 10)  # 1 -> 10 (count per row)
        img = add_overlay(img, get_square_mask(), opacity=combination["opacity"])

    elif combination["mask"] == "knit":
        density = int(density * 10)  # 10 -> 1000 (iterations)
        img = add_overlay(img, get_knit_mask(), opacity=combination["opacity"])

    elif combination["mask"] == "word":
        density = int(density * 2)  # 2 -> 200 (words)
        img = add_overlay(img, get_word_mask(words=get_imagenet_labels()), opacity=combination["opacity"])

    else:
        raise ValueError(f"Unknown mask: {combination['mask']}")

    return img


"""
config
"""


CONFIG = {
    "outpath": Path.cwd() / "data" / "eval" / "eval_cls.csv",
    "subset_size": 5,
}
CONFIG["outpath"].unlink(missing_ok=True)
CONFIG["fidkidpath"].unlink(missing_ok=True)


COMBINATIONS = {
    "mask": ["circle", "square", "diamond", "knit", "word"],
    "opacity": [0, 64, 128, 192, 255],  # range from 0 (transparent) to 255 (opaque)
    "density": [1, 25, 50, 75, 100],  # percentage of the image covered by the mask
}


"""
eval loop
"""


random_combinations = list(itertools.product(*COMBINATIONS.values()))
random.shuffle(random_combinations)
print(f"total iterations: {len(random_combinations)} * {CONFIG['subset_size']} = {len(random_combinations) * CONFIG['subset_size']}")

dataset = load_dataset("visual-layer/imagenet-1k-vl-enriched", split="validation", streaming=True).take(CONFIG["subset_size"]).shuffle(seed=random.randint(0, 1000))
dataset = list(map(lambda x: (x["image_id"], x["image"].convert("RGB"), x["label"], x["caption_enriched"]), dataset))  # same subset for all, for fair comparison
labels = get_imagenet_labels()

for combination in tqdm(random_combinations, total=len(random_combinations)):
    combination = dict(zip(COMBINATIONS.keys(), combination))

    for id, x_image, label_id, caption in dataset:
        advx_image = get_advx(x_image, combination)

        transform = transforms.Compose([transforms.Resize((256, 256)), transforms.Grayscale(num_output_channels=3), transforms.ToTensor()])
        x: torch.Tensor = transform(x_image).unsqueeze(0)
        advx_x: torch.Tensor = transform(advx_image).unsqueeze(0)

        def get_acc_boolmask(img: Image.Image) -> list[bool]:
            preds = list(zip(range(len(labels)), classify_clip(img, labels)))
            preds.sort(key=lambda x: x[1], reverse=True)
            top5_keys, top5_vals = zip(*preds[:5])
            top5_mask = [label_id == key for key in top5_keys]
            return top5_mask

        x_acc5 = get_acc_boolmask(x_image)
        advx_acc5 = get_acc_boolmask(advx_image)

        results = {
            # settings
            **combination,
            # semantic similarity
            "cosine_sim": get_cosine_similarity(x_image, advx_image),
            "psnr": get_psnr(x, advx_x),
            "ssim": get_ssim(x, advx_x),
            # accuracy
            "img_id": id,
            "label": get_imagenet_label(label_id),
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
