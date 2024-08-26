import gc
import json
import os
from pathlib import Path

import clip
import torch
import torch.nn.functional as F
from datasets import load_dataset

from advx.masks import get_diamond_mask
from advx.utils import add_overlay
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

# decide whether to train or validate based on the presence of the model .pth file

"""
training
"""

seed = 41
set_seed(seed=seed)

# config
num_epochs = 10 # increase for better performance
lr = 1e-5 # common for CLIP

# data
dataset = load_dataset("visual-layer/imagenet-1k-vl-enriched", split="train", streaming=True).shuffle(seed=seed)
overlay = get_diamond_mask(diamond_count=15, diamonds_per_row=10)
labels = get_imagenet_labels()

# model
model, preprocess = clip.load("ViT-L/14@336px", device=get_device())
model.train()
for param in model.parameters():
    param.requires_grad = True
model = model.float()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
if get_device() == "cuda":
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()


for epoch in range(num_epochs):
    for elem in dataset:
        image = preprocess(elem["image"].convert("RGB")).unsqueeze(0).to(get_device(), dtype=torch.float32)
        adv_img = preprocess(add_overlay(elem["image"].convert("RGB"), overlay=overlay, opacity=160)).unsqueeze(0).to(get_device(), dtype=torch.float32)

        text = clip.tokenize([get_imagenet_label(elem["label"])]).to(get_device())

        with torch.amp.autocast(device_type=get_device(disable_mps=True), enabled="cuda" == get_device()):
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)

            # loss for original image
            logits_per_image = image_features @ text_features.t()
            original_loss = F.cross_entropy(logits_per_image, torch.arange(len(text), device=get_device(), dtype=torch.long))

            # loss for adversarial image
            adv_image_features = model.encode_image(adv_img)
            adv_logits_per_image = adv_image_features @ text_features.t()
            adv_loss = F.cross_entropy(adv_logits_per_image, torch.arange(len(text), device=get_device(), dtype=torch.long))

            # combine losses
            similarity_loss = 1 - F.cosine_similarity(image_features, adv_image_features).mean()
            total_loss = original_loss + adv_loss + similarity_loss

        # backprop
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss.item()}")

    torch.cuda.empty_cache()
    gc.collect()


torch.save(model.state_dict(), "adversarially_trained_clip.pth")


"""
validation
"""

outpath = Path.cwd() / "data" / "eval" / "eval_cls.csv"
dataset = load_dataset("visual-layer/imagenet-1k-vl-enriched", split="validation", streaming=True).shuffle(seed=seed)
