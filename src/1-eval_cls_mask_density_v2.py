import csv
import gc
import itertools
import json
import random
from functools import lru_cache
from pathlib import Path

import open_clip
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from advx.masks import get_circle_mask, get_diamond_mask, get_knit_mask, get_square_mask, get_word_mask
from advx.utils import add_overlay
from metrics.metrics import get_cosine_similarity, get_psnr, get_ssim
from utils import get_device, set_env


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
    density = combination["density"]
    if combination["mask"] == "diamond":
        density = int(density / 10)  # 1 -> 10 (count per row)
        img = add_overlay(img, get_diamond_mask(diamonds_per_row=density), opacity=combination["opacity"])

    elif combination["mask"] == "circle":
        density = int(density / 10)  # 1 -> 10 (count per row)
        img = add_overlay(img, get_circle_mask(row_count=density), opacity=combination["opacity"])

    elif combination["mask"] == "square":
        density = int(density / 10)  # 1 -> 10 (count per row)
        img = add_overlay(img, get_square_mask(row_count=density), opacity=combination["opacity"])

    elif combination["mask"] == "knit":
        density = int(density * 10)  # 100 -> 1000 (iterations)
        img = add_overlay(img, get_knit_mask(step=density), opacity=combination["opacity"])

    elif combination["mask"] == "word":
        density = int(density * 2)  # 20 -> 200 (words)
        img = add_overlay(img, get_word_mask(num_words=density, words=get_imagenet_labels(), avoid_center=False), opacity=combination["opacity"])

    return img


"""
config
"""


device = get_device(disable_mps=False)
seed = 42
set_env(seed=seed)

CONFIG = {
    "outpath": Path.cwd() / "data" / "eval" / "eval_cls.csv",
    "subset_size": 10_000,
}
COMBINATIONS = {
    "model": ["vit", "eva02", "eva01", "convnext", "resnet"],
    "mask": ["circle", "square", "diamond", "knit", "word"],
    "opacity": [0, 64, 128, 192, 255],  # 0;255
    "density": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],  # 1;100
}

random_combinations = list(itertools.product(*COMBINATIONS.values()))
random.shuffle(random_combinations)
print(f"total iterations: {len(random_combinations)} * {CONFIG['subset_size']} = {len(random_combinations) * CONFIG['subset_size']}")


"""
eval loop
"""

# data
dataset = load_dataset("visual-layer/imagenet-1k-vl-enriched", split="validation", streaming=False).take(CONFIG["subset_size"]).shuffle(seed=seed)
dataset = list(map(lambda x: (x["image_id"], x["image"].convert("RGB"), x["label"], x["caption_enriched"]), dataset))
labels = get_imagenet_labels()
print("loaded dataset: imagenet-1k-vl-enriched")

# models
# see: https://github.com/mlfoundations/open_clip/blob/main/docs/openclip_results.csv
model_vit, _, preprocess_vit = open_clip.create_model_and_transforms("ViT-H-14-378-quickgelu", pretrained="dfn5b")
model_vit = torch.jit.script(model_vit)
model_vit.eval()
tokenizer_vit = open_clip.get_tokenizer("ViT-H-14-378-quickgelu")
text_vit = tokenizer_vit(labels)
print("loaded model: ViT-H-14-378-quickgelu")

model_eva02, _, preprocess_eva02 = open_clip.create_model_and_transforms("EVA02-E-14-plus", pretrained="laion2b_s9b_b144k")
model_eva02 = torch.jit.script(model_eva02)
model_eva02.eval()
tokenizer_eva02 = open_clip.get_tokenizer("EVA02-E-14-plus")
text_eva02 = tokenizer_eva02(labels)
print("loaded model: EVA02-E-14-plus")

model_eva01, _, preprocess_eva01 = open_clip.create_model_and_transforms("EVA01-g-14-plus", pretrained="merged2b_s11b_b114k")
model_eva01 = torch.jit.script(model_eva01)
model_eva01.eval()
tokenizer_eva01 = open_clip.get_tokenizer("EVA01-g-14-plus")
text_eva01 = tokenizer_eva01(labels)
print("loaded model: EVA01-g-14-plus")

model_convnext, _, preprocess_convnext = open_clip.create_model_and_transforms("convnext_xxlarge", pretrained="laion2b_s34b_b82k_augreg_soup")
model_convnext = torch.jit.script(model_convnext)
model_convnext.eval()
tokenizer_convnext = open_clip.get_tokenizer("convnext_xxlarge")
text_convnext = tokenizer_convnext(labels)
print("loaded model: convnext_xxlarge")

model_resnet, _, preprocess_resnet = open_clip.create_model_and_transforms("RN50x64", pretrained="openai")
model_resnet = torch.jit.script(model_resnet)
model_resnet.eval()
tokenizer_resnet = open_clip.get_tokenizer("RN50x64")
text_resnet = tokenizer_resnet(labels)
print("loaded model: RN50x64")


for combination in tqdm(random_combinations, total=len(random_combinations)):
    combination = dict(zip(COMBINATIONS.keys(), combination))

    for img_id, image, label_id, caption in dataset:
        entry_ids = {
            **combination,
            "img_id": img_id,
        }
        if is_cached(CONFIG["outpath"], entry_ids):
            print(f"skipping {entry_ids}")
            continue

        model, preprocess, text = None, None, None
        if combination["model"] == "vit":
            model, preprocess, text = model_vit, preprocess_vit, text_vit
        elif combination["model"] == "eva02":
            model, preprocess, text = model_eva02, preprocess_eva02, text_eva02
        elif combination["model"] == "eva01":
            model, preprocess, text = model_eva01, preprocess_eva01, text_eva01
        elif combination["model"] == "convnext":
            model, preprocess, text = model_convnext, preprocess_convnext, text_convnext
        elif combination["model"] == "resnet":
            model, preprocess, text = model_resnet, preprocess_resnet, text_resnet
        print(f"Model: {combination['model']}")
        print(f"model: {model}, preprocess: {preprocess}, text: {text}")
        assert model is not None and preprocess is not None and text is not None

        advx_image = get_advx(image, label_id, combination)
        x: torch.Tensor = preprocess(image).unsqueeze(0)
        advx_x: torch.Tensor = preprocess(advx_image).unsqueeze(0)

        def get_acc_boolmask(img: Image.Image) -> list[bool]:
            with torch.no_grad(), torch.amp.autocast(device_type=device, enabled="cuda" == device):
                image_features = model.encode_image(img)
                text_features = model.encode_text(text)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            preds = list(zip(range(len(labels)), text_probs.squeeze().tolist()))
            preds.sort(key=lambda x: x[1], reverse=True)
            top5_keys, top5_vals = zip(*preds[:5])
            top5_mask = [label_id == key for key in top5_keys]
            return top5_mask

        x_acc5 = get_acc_boolmask(image)
        advx_acc5 = get_acc_boolmask(advx_image)

        results = {
            **entry_ids,
            # semantic similarity
            "cosine_sim": get_cosine_similarity(image, advx_image),
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
