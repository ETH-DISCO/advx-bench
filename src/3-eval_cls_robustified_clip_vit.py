import gc
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from PIL import Image

from advx.masks import get_diamond_mask
from advx.utils import add_overlay
from models.cls import classify_clip, classify_robustified_clip
from utils import get_device, set_seed

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


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
num_epochs = 20
subset = 1

# data
outpath = Path.cwd() / "data" / "eval" / "eval_cls.csv"
dataset = load_dataset("visual-layer/imagenet-1k-vl-enriched", split="validation", streaming=True).take(subset).shuffle(seed=seed)
dataset = list(map(lambda x: (x["image_id"], x["image"].convert("RGB"), x["label"], x["caption_enriched"]), dataset))
overlay = get_diamond_mask(diamond_count=15, diamonds_per_row=10)
labels = get_imagenet_labels()

if device == "cuda":
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()

for id, img, label_id, caption in dataset:
    adv_img = add_overlay(img.convert("RGB"), overlay=overlay, opacity=160).convert("RGB")

    def get_acc_boolmask(img: Image.Image, model: torch.nn.Module) -> list[bool]:
        preds = list(zip(range(len(labels)), model(img, labels)))
        preds.sort(key=lambda x: x[1], reverse=True)
        top5_keys, top5_vals = zip(*preds[:5])
        top5_mask = [label_id == key for key in top5_keys]
        return top5_mask

    original_top5_boolmask = classify_clip(adv_img, labels)
    robustified_top5_boolmask = classify_robustified_clip(adv_img, labels)

    print(f"original: {original_top5_boolmask}")
    print(f"robustified: {robustified_top5_boolmask}")

    torch.cuda.empty_cache()
    gc.collect()
