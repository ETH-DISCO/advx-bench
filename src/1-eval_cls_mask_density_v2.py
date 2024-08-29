import csv
import gc
import itertools
import json
from pathlib import Path

import lpips
import open_clip
import torch
import torchvision.transforms as transforms
from datasets import load_dataset
from filelock import FileLock
from PIL import Image
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTModel

from advx.masks import get_circle_mask, get_diamond_mask, get_knit_mask, get_square_mask, get_word_mask
from advx.utils import add_overlay
from metrics.metrics import get_psnr, get_ssim
from utils import set_env


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


def load_model(model_name, pretrained, device, labels):
    # load to cpu first to avoid cuda out of memory
    # see: https://github.com/mlfoundations/open_clip/blob/main/docs/openclip_results.csv
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device="cpu")
    model = model.to("cpu")
    model.eval()

    model = torch.compile(model, mode="reduce-overhead")
    model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    tokenizer = open_clip.get_tokenizer(model_name)
    text = tokenizer(labels).to("cpu")

    torch.cuda.empty_cache()
    gc.collect()
    print(f"loaded model: {model_name}")
    return model, preprocess, text


def eval(device: str, seed: int, config: dict, random_combinations: list[tuple]):
    # data
    dataset = load_dataset("visual-layer/imagenet-1k-vl-enriched", split="validation", streaming=False).take(CONFIG["subset_size"]).shuffle(seed=seed)
    dataset = list(map(lambda x: (x["image_id"], x["image"].convert("RGB"), x["label"], x["caption_enriched"]), dataset))
    labels = get_imagenet_labels()
    print("loaded dataset: imagenet-1k-vl-enriched")

    # models
    lpips_model = lpips.LPIPS(net="vgg")
    feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    cosine_sim_model = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device)
    model_vit, preprocess_vit, text_vit = load_model("ViT-H-14-378-quickgelu", "dfn5b", device, labels)
    model_eva02, preprocess_eva02, text_eva02 = load_model("EVA02-E-14-plus", "laion2b_s9b_b144k", device, labels)
    model_eva01, preprocess_eva01, text_eva01 = load_model("EVA01-g-14-plus", "merged2b_s11b_b114k", device, labels)
    model_convnext, preprocess_convnext, text_convnext = load_model("convnext_xxlarge", "laion2b_s34b_b82k_augreg_soup", device, labels)
    model_resnet, preprocess_resnet, text_resnet = load_model("RN50x64", "openai", device, labels)

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
            entry_ids = {
                **combination,
                "img_id": img_id,
            }
            if is_cached(CONFIG["outpath"], entry_ids):
                print(f"skipping {entry_ids}")
                continue

            with torch.no_grad(), torch.amp.autocast(device_type=device, enabled=("cuda" in str(device))), torch.inference_mode():

                def get_boolmask(img: Image.Image) -> Image.Image:
                    img = img.convert("RGB")
                    img = preprocess(img).unsqueeze(0).to(device)

                    image_features = model.encode_image(img)
                    text_features = model.encode_text(text)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

                    probs = text_probs[0].cpu().numpy().tolist()
                    assert all(isinstance(prob, float) for prob in probs)
                    preds = list(zip(range(len(labels)), probs))
                    preds.sort(key=lambda x: x[1], reverse=True)
                    top5_keys, top5_vals = zip(*preds[:5])
                    boolmask = [label_id == key for key in top5_keys]
                    return boolmask

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

                x_acc5 = get_boolmask(image)
                advx_acc5 = get_boolmask(adv_image)

                x: torch.Tensor = transform(image).unsqueeze(0)
                advx_x: torch.Tensor = transform(adv_image).unsqueeze(0)

                results = {
                    **entry_ids,
                    # perceptual quality
                    "cosine_sim": get_cosine_similarity(image, adv_image),
                    "psnr": get_psnr(x, advx_x),
                    "ssim": get_ssim(x, advx_x),
                    "lpips": lpips_model(x, advx_x).item(),
                    # adversarial accuracy disadvantage
                    "x_acc1": 1 if x_acc5[0] else 0,
                    "advx_acc1": 1 if advx_acc5[0] else 0,
                    "x_acc5": 1 if any(x_acc5) else 0,
                    "advx_acc5": 1 if any(advx_acc5) else 0,
                }

            lock = FileLock(CONFIG["outpath"].with_suffix(".lock"), timeout=1)
            with lock:
                with open(CONFIG["outpath"], mode="a") as f:
                    writer = csv.DictWriter(f, fieldnames=results.keys())
                    if CONFIG["outpath"].stat().st_size == 0:
                        writer.writeheader()
                    writer.writerow(results)

            torch.cuda.empty_cache()
            gc.collect()


"""
config
"""

seed = 42
set_env(seed=seed)

CONFIG = {
    "outpath": Path.cwd() / "data" / "eval" / "eval_cls.csv",
    "subset_size": 1_000,
}
COMBINATIONS = {
    "model": ["vit", "eva02", "eva01", "convnext", "resnet"],
    "mask": ["circle", "square", "diamond", "knit", "word"],
    "opacity": [0, 64, 128, 192, 255],  # 0;255
    "density": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],  # 1;100
}

random_combinations = list(itertools.product(*COMBINATIONS.values()))
random_combinations.sort(key=lambda x: x[0])  # sort list by model to reduce model loading
print(f"total iterations: {len(random_combinations)} * {CONFIG['subset_size']} = {len(random_combinations) * CONFIG['subset_size']}")

assert torch.cuda.is_available()
devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
for device in devices:
    eval(device, seed, CONFIG, random_combinations)
