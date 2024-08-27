import csv
import gc
import json
import os
from pathlib import Path
from typing import Callable

import torch
import torchvision.transforms as transforms
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from advx.masks import get_diamond_mask
from advx.utils import add_overlay
from metrics.metrics import get_cosine_similarity, get_psnr, get_ssim
from models.cls import classify_clip, classify_robustified_clip
from utils import get_device, set_seed

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def is_cached(path: Path, entry_id: dict) -> bool:
    entry_id = entry_id.copy()

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


"""
validation
"""

seed = 41
set_seed(seed=seed)
device = get_device()

# config
subset = 5_000

# data
outpath = Path.cwd() / "data" / "eval" / "eval_cls.csv"
dataset = load_dataset("visual-layer/imagenet-1k-vl-enriched", split="validation", streaming=False).shuffle(seed=seed).take(subset)
dataset = list(map(lambda x: (x["image_id"], x["image"].convert("RGB"), x["label"], x["caption_enriched"]), dataset))
overlay = get_diamond_mask(diamond_count=15, diamonds_per_row=10)
labels = get_imagenet_labels()

if device == "cuda":
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()

for id, img, label_id, caption in tqdm(dataset):
    entry_id = {
        "img_id": id,
    }
    if is_cached(outpath, entry_id):
        print(f"skipping {entry_id}")
        continue

    adv_img = add_overlay(img.convert("RGB"), overlay=overlay, opacity=160).convert("RGB")
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.Grayscale(num_output_channels=3), transforms.ToTensor()])
    x: torch.Tensor = transform(img).unsqueeze(0)
    advx_x: torch.Tensor = transform(adv_img).unsqueeze(0)

    def get_acc_boolmask(img: Image.Image, model: Callable) -> list[bool]:
        preds = list(zip(range(len(labels)), model(img, labels)))
        preds.sort(key=lambda x: x[1], reverse=True)
        top5_keys, top5_vals = zip(*preds[:5])
        top5_mask = [label_id == key for key in top5_keys]
        return top5_mask

    boolmask = get_acc_boolmask(img, classify_clip)
    adv_boolmask = get_acc_boolmask(adv_img, classify_robustified_clip)

    results = {
        **entry_id,
        # semantic similarity
        "cosine_sim": get_cosine_similarity(img, adv_img),
        "psnr": get_psnr(x, advx_x),
        "ssim": get_ssim(x, advx_x),
        # accuracy
        "label": get_imagenet_label(label_id),
        "original_acc1": 1 if boolmask[0] else 0,
        "robustified_acc1": 1 if adv_boolmask[0] else 0,
        "original_acc5": 1 if any(boolmask) else 0,
        "robustified_acc5": 1 if any(adv_boolmask) else 0,
    }

    with open(outpath, mode="a") as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        if outpath.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(results)

    torch.cuda.empty_cache()
    gc.collect()
