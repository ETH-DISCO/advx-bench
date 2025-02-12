import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file
from tqdm import tqdm

from models.caption import caption_blip
from models.cls import classify_metaclip
from models.det import detect_vit
from models.seg import segment_sam1

datapath = Path.cwd() / "data" / "hcaptcha" / "seg" / "data"
outputpath = Path.cwd() / "data" / "hcaptcha" / "seg" / "eval"
assert datapath.exists()
files = [x for x in datapath.iterdir() if x.is_file()]

for file in tqdm(files):
    if (outputpath / f"{file.stem}.safetensors").exists():
        print(f"skipping: `{file.stem}.safetensors`")
        continue

    img = Image.open(file)
    threshold = 0.1

    img = img.convert("RGB")
    assert img.mode == "RGB" and len(np.array(img).shape) == 3

    captions: list[str] = caption_blip(img)
    probs: list[float] = classify_metaclip(img, captions)
    detection: tuple[list[list[float]], list[float], list[str]] = detect_vit(img, captions, threshold)
    boxes, scores, labels = detection
    masks: torch.Tensor = segment_sam1(img, boxes)

    def ensure_contiguous(tensor):
        if len(tensor) == 0:
            return torch.empty(0)
        elif tensor.is_contiguous():
            return tensor
        else:
            return tensor.contiguous()

    img_tensor = ensure_contiguous(torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0)

    data_dict = {
        "image": img_tensor,
        "captions": ensure_contiguous(torch.tensor([ord(c) for c in "|".join(captions)], dtype=torch.int32)),
        "classification_probs": ensure_contiguous(torch.tensor(probs, dtype=torch.float32)),
        "detection_boxes": ensure_contiguous(torch.tensor(boxes, dtype=torch.float32)),
        "detection_scores": ensure_contiguous(torch.tensor(scores, dtype=torch.float32)),
        "detection_labels": ensure_contiguous(torch.tensor([ord(c) for c in "|".join(labels)], dtype=torch.int32)),
        "segmentation_masks": ensure_contiguous(masks),
    }

    save_file(data_dict, outputpath / f"{file.stem}.safetensors")
    os.system(f"git add . && git commit -m 'autocommit' && git push")
