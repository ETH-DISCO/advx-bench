import gc
import json
import os
from pathlib import Path

import torch
import torchvision.transforms as transforms
from datasets import load_dataset
from PIL import Image

from advx.masks import get_diamond_mask
from advx.utils import add_overlay
from utils import get_device, set_seed

torch.backends.cuda.matmul.allow_tf32 = True  # allow TF32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow TF32 on cudnn
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def get_advx(img: Image.Image) -> Image.Image:
    # optimal settings: `diamond mask`, `perturb=False`, `density=50`, `opacity=150;170`
    density = 50
    opacity = (150 + 170) / 2
    img = add_overlay(img, get_diamond_mask(diamond_count=(density / 10 + 10), diamonds_per_row=(density / 5)), opacity=opacity)
    return img


def get_imagenet_label(idx: int) -> str:
    datapath = Path.cwd() / "data" / "imagenet_labels.json"
    data = json.loads(datapath.read_text())
    return data[str(idx)]


def get_imagenet_labels() -> list[str]:
    datapath = Path.cwd() / "data" / "imagenet_labels.json"
    data = json.loads(datapath.read_text())
    return list(data.values())


"""
run
"""

CONFIG = {
    "outpath": Path.cwd() / "data" / "eval" / "eval_cls.csv",
}

seed = 41
set_seed(seed=seed)
dataset = load_dataset("visual-layer/imagenet-1k-vl-enriched", split="validation", streaming=True).take(CONFIG["subset_size"]).shuffle(seed=seed)
dataset = list(map(lambda x: (x["image_id"], x["image"].convert("RGB"), x["label"], x["caption_enriched"]), dataset))
labels = get_imagenet_labels()

if get_device() == "cuda":
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()

for id, x_image, label_id, caption in dataset:
    with torch.no_grad(), torch.amp.autocast(device_type=get_device(disable_mps=True), enabled="cuda" == get_device()):
        advx_image = get_advx(x_image, label_id)

        transform = transforms.Compose([transforms.Resize((256, 256)), transforms.Grayscale(num_output_channels=3), transforms.ToTensor()])
        x: torch.Tensor = transform(x_image).unsqueeze(0)
        advx_x: torch.Tensor = transform(advx_image).unsqueeze(0)

    torch.cuda.empty_cache()
    gc.collect()
