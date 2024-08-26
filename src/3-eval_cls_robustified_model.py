import json
import os
from pathlib import Path

import torch
from datasets import load_dataset

from advx.masks import get_diamond_mask
from advx.utils import add_overlay
from utils import set_seed

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
adversarial training
"""

seed = 41
set_seed(seed=seed)

dataset = load_dataset("visual-layer/imagenet-1k-vl-enriched", split="train", streaming=True).shuffle(seed=seed)

overlay = get_diamond_mask(diamond_count=15, diamonds_per_row=10)

for elem in dataset:
    elem = {
        "image_id": elem["image_id"],
        "image": elem["image"].convert("RGB"),
        "advx_image": add_overlay(elem["image"].convert("RGB"), overlay=overlay, opacity=160),
        "label": elem["label"],
        "label_word": get_imagenet_label(elem["label"]),
        "caption_enriched": elem["caption_enriched"],
    }
    print(elem)

    elem["image"].show()
    elem["advx_image"].show()
    break

outpath = (Path.cwd() / "data" / "eval" / "eval_cls.csv",)

# if get_device() == "cuda":
#     torch.cuda.empty_cache()
#     torch.cuda.reset_peak_memory_stats()
#     torch.cuda.reset_accumulated_memory_stats()

# for id, x_image, label_id, caption in dataset:
#     with torch.no_grad(), torch.amp.autocast(device_type=get_device(disable_mps=True), enabled="cuda" == get_device()):

#         transform = transforms.Compose([transforms.Resize((256, 256)), transforms.Grayscale(num_output_channels=3), transforms.ToTensor()])
#         x: torch.Tensor = transform(x_image).unsqueeze(0)
#         advx_x: torch.Tensor = transform(advx_image).unsqueeze(0)

#     torch.cuda.empty_cache()
#     gc.collect()
