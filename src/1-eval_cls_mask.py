import csv
import gc
import itertools
import json
import os
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
from utils import get_device

torch.backends.cuda.matmul.allow_tf32 = True  # allow TF32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow TF32 on cudnn
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def is_cached(path: Path, entry_id: dict) -> bool:
    if not path.exists():
        return False

    with open(path, mode="r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if all(row[key] == str(value) for key, value in entry_id.items()):
                return True
    return False


def get_imagenet_label(idx: int) -> str:
    datapath = Path.cwd() / "data" / "imagenet_labels.json"
    data = json.loads(datapath.read_text())
    return data[str(idx)]


def get_imagenet_labels() -> list[str]:
    datapath = Path.cwd() / "data" / "imagenet_labels.json"
    data = json.loads(datapath.read_text())
    return list(data.values())


def get_advx(img: Image.Image, label_id: int, combination: dict) -> Image.Image:
    if combination["mask"] == "diamond":
        img = add_overlay(img, get_diamond_mask(), opacity=combination["opacity"])

    elif combination["mask"] == "circle":
        img = add_overlay(img, get_circle_mask(), opacity=combination["opacity"])

    elif combination["mask"] == "square":
        img = add_overlay(img, get_square_mask(), opacity=combination["opacity"])

    elif combination["mask"] == "knit":
        img = add_overlay(img, get_knit_mask(), opacity=combination["opacity"])

    elif combination["mask"] == "word":
        img = add_overlay(img, get_word_mask(words=get_imagenet_labels(), avoid_center=False), opacity=combination["opacity"])

    return img


"""
config
"""

CONFIG = {
    "outpath": Path.cwd() / "data" / "eval" / "eval_cls.csv",
    "subset_size": 5,
}
COMBINATIONS = {
    "mask": ["circle", "square", "diamond", "knit", "word"],
    "opacity": [0, 64, 128, 192, 255],  # 0;255
}

"""
eval loop
"""

random_combinations = list(itertools.product(*COMBINATIONS.values()))
random.shuffle(random_combinations)
print(f"total iterations: {len(random_combinations)} * {CONFIG['subset_size']} = {len(random_combinations) * CONFIG['subset_size']}")

dataset = load_dataset("visual-layer/imagenet-1k-vl-enriched", split="validation", streaming=True).take(CONFIG["subset_size"]).shuffle(seed=random.randint(0, 1000))
dataset = list(map(lambda x: (x["image_id"], x["image"].convert("RGB"), x["label"], x["caption_enriched"]), dataset))
labels = get_imagenet_labels()

if get_device() == "cuda":
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()

for combination in tqdm(random_combinations, total=len(random_combinations)):
    combination = dict(zip(COMBINATIONS.keys(), combination))

    for id, x_image, label_id, caption in dataset:
        entry_id = {
            **combination,
            "img_id": id,
        }
        if is_cached(CONFIG["outpath"], entry_id):
            print(f"skipping {entry_id}")
            continue

        if get_device() == "cuda":
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=True):
                advx_image = get_advx(x_image, label_id, combination)

                transform = transforms.Compose([transforms.Resize((256, 256)), transforms.Grayscale(num_output_channels=3), transforms.ToTensor()])
                x: torch.Tensor = transform(x_image).unsqueeze(0)
                advx_x: torch.Tensor = transform(advx_image).unsqueeze(0)

                def get_acc_boolmask(img: Image.Image) -> list[bool]:
                    preds = list(zip(range(len(labels)), classify_clip(img, labels)))
                    preds.sort(key=lambda x: x[1], reverse=True)
                    top5_keys, top5_vals = zip(*preds[:5])
                    top5_mask = [label_id == key for key in top5_keys]
                    return top5_mask

        else:
            with torch.no_grad():
                advx_image = get_advx(x_image, label_id, combination)

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
            **entry_id,
            # semantic similarity
            "cosine_sim": get_cosine_similarity(x_image, advx_image),
            "psnr": get_psnr(x, advx_x),
            "ssim": get_ssim(x, advx_x),
            # accuracy
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

        torch.cuda.empty_cache()
        gc.collect()
