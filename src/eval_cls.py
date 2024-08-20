import csv
import json
from pathlib import Path

import torch
import torchvision.transforms as transforms
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from metrics.metrics import get_cosine_similarity, get_fid, get_inception_features, get_kid, get_psnr, get_ssim
from models.cls import classify_clip
from utils import set_seed


def get_imagenet_generator(size: int, seed: int = 41):
    subset = load_dataset("visual-layer/imagenet-1k-vl-enriched", split="validation", streaming=True).take(size).shuffle(seed=seed)  # type: ignore
    for elem in subset:
        yield elem["image_id"], elem["image"].convert("RGB"), elem["label"], elem["caption_enriched"]  # type: ignore


def get_imagenet_label(idx: int) -> str:
    datapath = Path.cwd() / "data" / "imagenet_labels.json"
    data = json.loads(datapath.read_text())
    return data[str(idx)]


def get_imagenet_labels() -> list[str]:
    datapath = Path.cwd() / "data" / "imagenet_labels.json"
    data = json.loads(datapath.read_text())
    return list(data.values())


def get_advx(image: Image.Image, config: dict) -> Image.Image:
    # apply
    return image


if __name__ == "__main__":
    set_seed(41)

    config = {
        "outpath": Path.cwd() / "data" / "eval" / "eval_cls.json",
        "fidkidpath": Path.cwd() / "data" / "eval" / "eval_cls_fidkid.json",
        "subset_size": 25,
        # advx config
        # ...
    }
    config["outpath"].unlink(missing_ok=True)
    config["fidkidpath"].unlink(missing_ok=True)

    dataset = get_imagenet_generator(size=config["subset_size"])
    labels = get_imagenet_labels()

    x_features = []
    advx_features = []

    for id, x_image, label_id, caption in tqdm(dataset, total=config["subset_size"]):
        advx_image = get_advx(x_image, config)

        transform = transforms.Compose([transforms.Resize((256, 256)), transforms.Grayscale(num_output_channels=3), transforms.ToTensor()])
        x: torch.Tensor = transform(x_image).unsqueeze(0)
        advx_x: torch.Tensor = transform(advx_image).unsqueeze(0)

        x_features.append(get_inception_features(x))
        advx_features.append(get_inception_features(advx_x))

        def get_acc_boolmask(img: Image.Image) -> list[bool]:
            preds = list(zip(range(len(labels)), classify_clip(img, labels)))
            preds.sort(key=lambda x: x[1], reverse=True)
            top5_keys, top5_vals = zip(*preds[:5])
            top5_mask = [label_id == key for key in top5_keys]
            return top5_mask

        x_acc5 = get_acc_boolmask(x_image)
        advx_acc5 = get_acc_boolmask(advx_image)

        results = {
            **config,
            # semantic similarity
            "cosine_sim": get_cosine_similarity(x, advx_x),
            "psnr": get_psnr(x, advx_x),
            "ssim": get_ssim(x, advx_x),
            # accuracy
            "img_id": id,
            "label": get_imagenet_label(label_id),
            "caption": caption,
            "x_acc1": x_acc5[0],
            "advx_acc1": advx_acc5[0],
            "x_acc5": any(x_acc5),
            "advx_acc5": any(advx_acc5),
        }

        with open(config["outpath"], mode="a") as f:
            writer = csv.DictWriter(f, fieldnames=results.keys())
            if config["outpath"].stat().st_size == 0:
                writer.writeheader()
            writer.writerow(results)

    with open(config["fidkidpath"], mode="a") as f:
        metrics = {
            "fid": get_fid(x_features, x_features),
            "kid": get_kid(advx_features, advx_features, config["subset_size"]),
        }
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        if config["fidkidpath"].stat().st_size == 0:
            writer.writeheader()
        writer.writerow(metrics)
