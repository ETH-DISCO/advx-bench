import csv
import gc
import itertools
import json
import os
import random
from pathlib import Path

import lpips
import numpy as np
import open_clip
import torch
import torchvision.transforms as transforms
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTModel

from advx.masks import get_circle_mask, get_diamond_mask, get_knit_mask, get_square_mask, get_word_mask
from advx.utils import add_overlay
from metrics.metrics import get_psnr, get_ssim


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


CONFIG = {
    "outpath": Path.cwd() / "data" / "eval" / "eval_cls_generalizability.csv",
    "subset_size": 500,
}
COMBINATIONS = {
    "model": ["vit", "eva02", "eva01", "convnext", "resnet"],
    "mask": ["circle", "square", "diamond", "knit"],
    "opacity": [50, 80, 110, 140, 170],  # 0;255
    "density": [70],  # 1;100
}

random_combinations = list(itertools.product(*COMBINATIONS.values()))
print(f"total iterations: {len(random_combinations)} * {CONFIG['subset_size']} = {len(random_combinations) * CONFIG['subset_size']}")

if CONFIG["outpath"].exists():
    filecontent = list(csv.DictReader(CONFIG["outpath"].open(mode="r")))
    filecontent: list[dict] = [{k: v for k, v in entry.items() if k in list(COMBINATIONS.keys())} for entry in filecontent]  # remove cols
    filecontent = [dict(t) for t in {tuple(d.items()) for d in filecontent}]  # remove duplicates
    filecontent: list[tuple] = [tuple(entry.values()) for entry in filecontent]
    for fc_entry in filecontent:
        fc_entry = tuple(map(lambda x: int(x) if isinstance(x, str) and x.isdigit() else x, fc_entry))
        random_combinations.remove(fc_entry)

print(f"remaining iterations: {len(random_combinations)} * {CONFIG['subset_size']} = {len(random_combinations) * CONFIG['subset_size']}")


"""
environment
"""


device = "cuda" if torch.cuda.is_available() else "cpu"

seed = 42

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if device == "cuda":
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()


"""
eval loop
"""


# data
dataset = load_dataset("visual-layer/imagenet-1k-vl-enriched", split="validation", streaming=False).take(CONFIG["subset_size"]).shuffle(seed=seed)
dataset = list(map(lambda x: (x["image_id"], x["image"].convert("RGB"), x["label"], x["caption_enriched"]), dataset))
labels = get_imagenet_labels()
print("loaded dataset: imagenet-1k-vl-enriched")


def load_model(model_name, pretrained, labels):
    # load to cpu first to avoid cuda out of memory
    # see: https://github.com/mlfoundations/open_clip/blob/main/docs/openclip_results.csv
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device="cpu")
    model = model.to("cpu")
    model.eval()
    model = torch.compile(model, mode="reduce-overhead")

    tokenizer = open_clip.get_tokenizer(model_name)
    text = tokenizer(labels).to("cpu")

    torch.cuda.empty_cache()
    gc.collect()
    print(f"loaded model: {model_name}")
    return model, preprocess, text


# models
lpips_model = lpips.LPIPS(net="vgg")
feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
cosine_sim_model = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device)
model_vit, preprocess_vit, text_vit = load_model("ViT-H-14-378-quickgelu", "dfn5b", labels)
model_eva02, preprocess_eva02, text_eva02 = load_model("EVA02-E-14-plus", "laion2b_s9b_b144k", labels)
model_eva01, preprocess_eva01, text_eva01 = load_model("EVA01-g-14-plus", "merged2b_s11b_b114k", labels)
model_convnext, preprocess_convnext, text_convnext = load_model("convnext_xxlarge", "laion2b_s34b_b82k_augreg_soup", labels)
model_resnet, preprocess_resnet, text_resnet = load_model("RN50x64", "openai", labels)


for combination in tqdm(random_combinations, total=len(random_combinations)):
    combination = dict(zip(COMBINATIONS.keys(), combination))

    model, preprocess, text, transform = None, None, None, None
    if combination["model"] == "vit":
        model, preprocess, text = model_vit, preprocess_vit, text_vit
        transform = transforms.Compose([transforms.Lambda(lambda x: x.convert("RGB")), transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    elif combination["model"] == "eva02":
        model, preprocess, text = model_eva02, preprocess_eva02, text_eva02
        required_size = 224
        transform = transforms.Compose([transforms.Lambda(lambda x: x.convert("RGB")), transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    elif combination["model"] == "eva01":
        model, preprocess, text = model_eva01, preprocess_eva01, text_eva01
        transform = transforms.Compose([transforms.Lambda(lambda x: x.convert("RGB")), transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    elif combination["model"] == "convnext":
        model, preprocess, text = model_convnext, preprocess_convnext, text_convnext
        transform = transforms.Compose([transforms.Lambda(lambda x: x.convert("RGB")), transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    elif combination["model"] == "resnet":
        model, preprocess, text = model_resnet, preprocess_resnet, text_resnet
        transform = transforms.Compose([transforms.Lambda(lambda x: x.convert("RGB")), transforms.Resize(448), transforms.CenterCrop(448), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    assert model is not None and preprocess is not None and text is not None and transform is not None
    model = model.to(device)
    text = text.to(device)

    for img_id, image, label_id, caption in tqdm(dataset, total=len(dataset)):
        print(f"processing {combination} - {img_id}")
        entry_ids = {
            **combination,
            "img_id": img_id,
        }
        if is_cached(CONFIG["outpath"], entry_ids):
            print(f"skipping {entry_ids}")
            continue

        def get_boolmask(img: Image.Image) -> Image.Image:
            img = img.convert("RGB")
            img = preprocess(img).unsqueeze(0)
            with torch.cuda.amp.autocast(enabled=False):
                image_features = model.encode_image(img.to(device))
                text_features = model.encode_text(text)

            image_features = image_features.cpu()
            text_features = text_features.cpu()

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

            probs = text_probs[0].numpy().tolist()
            assert all(isinstance(prob, float) for prob in probs)
            preds = list(zip(range(len(labels)), probs))
            preds.sort(key=lambda x: x[1], reverse=True)
            top5_keys, top5_vals = zip(*preds)
            boolmask = [label_id == key for key in top5_keys]
            return boolmask

        with torch.no_grad(), torch.amp.autocast(device_type=device, enabled=("cuda" in str(device))), torch.inference_mode():

            def get_cosine_similarity(x: Image.Image, y: Image.Image) -> float:
                x = x.convert("RGB")
                y = y.convert("RGB")
                inputs1 = feature_extractor(images=x, return_tensors="pt")
                inputs2 = feature_extractor(images=y, return_tensors="pt")
                inputs1 = {k: v.to(device) for k, v in inputs1.items()}
                inputs2 = {k: v.to(device) for k, v in inputs2.items()}
                outputs1 = cosine_sim_model(**inputs1)
                outputs2 = cosine_sim_model(**inputs2)

                latents1 = outputs1.last_hidden_state[:, 0, :]
                latents2 = outputs2.last_hidden_state[:, 0, :]
                similarity = torch.nn.functional.cosine_similarity(latents1, latents2).item()
                return similarity

            adv_image = get_advx(image, label_id, combination)

            boolmask = get_boolmask(image)
            acc_rank = boolmask.index(True) if True in boolmask else -1
            adv_boolmask = get_boolmask(adv_image)
            adv_acc_rank = adv_boolmask.index(True) if True in adv_boolmask else -1

            x: torch.Tensor = transform(image).unsqueeze(0)
            adv_x: torch.Tensor = transform(adv_image).unsqueeze(0)

            results = {
                **entry_ids,
                # perceptual quality
                "cosine_sim": get_cosine_similarity(image, adv_image),
                "psnr": get_psnr(x, adv_x),
                "ssim": get_ssim(x, adv_x),
                "lpips": lpips_model(x, adv_x).item(),
                # difference in prediction with vs. without mask
                "acc_rank": acc_rank,
                "adv_acc_rank": adv_acc_rank,
            }

        with open(CONFIG["outpath"], mode="a") as f:
            writer = csv.DictWriter(f, fieldnames=results.keys())
            if CONFIG["outpath"].stat().st_size == 0:
                writer.writeheader()
            writer.writerow(results)

        del image
        del adv_image
        del x
        del adv_x
        del boolmask
        del adv_boolmask
        del results
        gc.collect()
        torch.cuda.empty_cache()
